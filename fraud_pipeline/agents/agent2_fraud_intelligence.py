"""
Agent 2 — Fraud Intelligence Agent
=====================================
Multi-signal fraud scoring using three layers of feature engineering:

  Layer 1 — Raw claim features        (policy, financial, demographic)
  Layer 2 — Document quality features (OCR, metadata, verification)
  Layer 3 — Cross-agent signal features (reads Agents 1,3,4,5,7 outputs)

Combined → 32-dimensional feature vector fed into an Isolation Forest
trained on 1,000 synthetic claims (800 normal + 150 suspicious + 50 fraud).

Rule-based signals (10 rules + compound amplifier) provide an explainable,
deterministic baseline. The ML model adds statistical anomaly detection on
patterns rules miss.

Final fraud_risk_score = rule_score × 0.60 + anomaly_score × 0.40

No hardcoded field names for diseases or conditions.
All signal thresholds are externalized in SIGNAL_WEIGHTS.
"""

import logging
import os
import numpy as np
from typing import Dict, List, Tuple

from sklearn.ensemble import IsolationForest

from ..schemas import ClaimState, FraudIntelligenceOutput
from ..services.llm_service import LLMService
from ..utils.runtime_queries import (
    get_nominee_claim_count,
    get_bank_account_claim_count,
)

logger = logging.getLogger(__name__)


# ── Risk level thresholds ─────────────────────────────────────────────────────
# (min_score, level) — checked top-to-bottom, first match wins
RISK_LEVELS: List[Tuple[float, str]] = [
    (0.80, "CRITICAL"),
    (0.60, "HIGH"),
    (0.35, "MEDIUM"),
    (0.00, "LOW"),
]

RECOMMENDED_ACTIONS = {
    "CRITICAL": "ESCALATE",
    "HIGH":     "ESCALATE",
    "MEDIUM":   "REVIEW",
    "LOW":      "APPROVE",
}


# ── Rule-based signal weights ─────────────────────────────────────────────────
# Each fired signal adds its weight to rule_score independently.
# Weights reflect relative insurance fraud signal strength.
SIGNAL_WEIGHTS: Dict[str, float] = {
    # Policy timeline signals
    "policy_very_early":         0.35,   # < 30 days — extremely suspicious
    "policy_early":              0.20,   # 30-180 days — early claim indicator
    "policy_revival":            0.22,   # policy revived just before death
    "policy_lapsed":             0.10,   # had lapse periods (minor signal alone)

    # Document forensics signals
    "metadata_tampered":         0.30,   # exif/creation timestamp manipulation
    "image_manipulation":        0.35,   # pixel-level image tampering
    "very_low_ocr":              0.28,   # avg OCR < 0.45 — likely forged doc
    "low_ocr_confidence":        0.15,   # avg OCR 0.45-0.60 — quality issue

    # External verification failures
    "hospital_not_verified":     0.18,   # hospital not in national registry
    "doctor_not_verified":       0.15,   # doctor credentials unconfirmed
    "fir_not_verified":          0.12,   # FIR authenticity unconfirmed
    "geo_mismatch":              0.15,   # location inconsistency in metadata

    # Cross-agent confirmed signals (highest weight — already validated)
    "fraud_ring_detected":       0.45,   # Agent 7: connected to fraud network
    "non_disclosure_detected":   0.35,   # Agent 4: withheld medical history
    "genuine_conflict":          0.25,   # Agent 5: GENUINE_CONFLICT_DETECTED flag
    "repeated_nominee":          0.30,   # same nominee across multiple claims
    "repeated_bank_account":     0.30,   # same bank account across claims
    "shared_hospital":           0.12,   # same hospital in multiple claims

    # Financial motivation signals
    "very_high_sum_assured":     0.18,   # > ₹1 crore — high financial motive
    "high_sum_assured":          0.08,   # > ₹50 lakh — elevated motive
    "premium_irregularity":      0.18,   # payment gaps / arrears

    # Demographic signals
    "very_young_death":          0.18,   # age < 25 — statistically rare
    "young_death":               0.08,   # age 25-35 — warrants scrutiny

    # Compound amplifier — multiple signals firing together
    "compound_4plus":            0.18,   # 4+ signals active simultaneously
    "compound_2plus":            0.05,   # 2-3 signals active simultaneously
}


# ── Isolation Forest hyper-parameters ────────────────────────────────────────
_IF_N_ESTIMATORS  = 300     # more trees = more stable scores
_IF_CONTAMINATION = 0.10    # assume ~10% of claims are anomalous
_IF_RANDOM_STATE  = 42
_IF_MAX_SAMPLES   = "auto"

# Number of features — MUST match _FeatureExtractor.extract() output length
_N_FEATURES = 32


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR — 3-layer, 32-dimensional
# ═══════════════════════════════════════════════════════════════════════════════

class _FeatureExtractor:
    """
    Extracts a fixed-length 32-dimensional feature vector from ClaimState.
    All features are normalized to [0, 1] for Isolation Forest compatibility.

    Feature layout (indices 0-31):
      [ 0] policy_age_norm         — normalized policy age (0=very new, 1=5+ years)
      [ 1] policy_age_early_flag   — 1 if policy_age < 180 days
      [ 2] policy_age_veryyoung    — 1 if policy_age < 30 days
      [ 3] policy_revival_flag     — 1 if revival detected
      [ 4] sum_assured_norm        — normalized (0 to 1 crore+)
      [ 5] premium_irregular_flag  — 1 if premium irregularities detected
      [ 6] death_age_norm          — normalized age at death
      [ 7] young_death_flag        — 1 if age < 35
      [ 8] avg_ocr_confidence      — average OCR across all documents
      [ 9] min_ocr_confidence      — worst-case OCR (catches single forged doc)
      [10] low_ocr_flag            — 1 if avg_ocr < 0.60
      [11] very_low_ocr_flag       — 1 if min_ocr < 0.45
      [12] doc_count_norm          — normalized document count
      [13] insufficient_docs_flag  — 1 if fewer than 2 documents
      [14] metadata_tampered_flag  — from Agent 1 external_verification
      [15] image_manipulated_flag  — from Agent 1 external_verification
      [16] hospital_verified       — 1=verified, 0=failed
      [17] doctor_verified         — 1=verified, 0=failed
      [18] fir_verified            — 1=verified, 0=failed
      [19] geo_mismatch_flag       — 1 if location inconsistency detected
      [20] verification_confidence — overall verification confidence score
      [21] early_claim_risk_norm   — Agent 3 risk: LOW=0.1 MED=0.5 HIGH=0.75 VERY_HIGH=1.0
      [22] agent3_revival_flag     — from Agent 3 early_claim_analysis output
      [23] non_disclosure_flag     — from Agent 4: contradiction_detected
      [24] non_disclosure_score    — from Agent 4: non_disclosure_score (0-1)
      [25] conflict_detected_flag  — from Agent 5: conflict_detected
      [26] genuine_conflict_flag   — from Agent 5: GENUINE_CONFLICT_DETECTED in flags
      [27] conflict_severity_norm  — ESCALATE=1.0, MANUAL_REVIEW=0.5, ACCEPT=0.0
      [28] fraud_ring_flag         — from Agent 7: fraud_ring_detected
      [29] graph_risk_score        — from Agent 7: graph_risk_score (0-1)
      [30] connected_claims_norm   — from Agent 7: connected_claims / 10
      [31] validation_flag_count   — normalized count of pipeline validation flags
    """

    _EARLY_RISK_MAP = {
        "VERY_HIGH": 1.00,
        "HIGH":      0.75,
        "MEDIUM":    0.50,
        "LOW":       0.10,
    }
    _CONFLICT_SEVERITY_MAP = {
        "ESCALATE":      1.00,
        "MANUAL_REVIEW": 0.50,
        "ACCEPT":        0.00,
    }

    def extract(self, state: ClaimState) -> np.ndarray:
        """Return a float64 ndarray of shape (32,). Never raises — defaults to 0."""
        ev = state.external_verification or {}
        ec = state.early_claim_analysis   or {}
        nd = state.non_disclosure_analysis or {}
        cr = state.conflict_resolution    or {}
        ga = state.graph_analysis         or {}

        # ── OCR helpers ────────────────────────────────────────────────────────
        ocr_scores = list(state.ocr_confidence_scores.values())
        if not ocr_scores:
            ocr_scores = [
                d.ocr_confidence for d in state.submitted_documents
                if d.ocr_confidence > 0
            ]
        avg_ocr = float(sum(ocr_scores) / len(ocr_scores)) if ocr_scores else 0.85
        min_ocr = float(min(ocr_scores)) if ocr_scores else 0.85

        # ── Age helper ─────────────────────────────────────────────────────────
        age = int(
            state.life_assured.age
            or state.life_assured.age_at_death
            or 0
        )

        # ── Conflict flags ─────────────────────────────────────────────────────
        cr_flags          = cr.get("validation_flags", [])
        genuine_conflict  = 1.0 if "GENUINE_CONFLICT_DETECTED" in cr_flags else 0.0

        # ── Build vector ───────────────────────────────────────────────────────
        policy_age = max(0, state.policy_age_days or 0)

        vec = np.array([
            # Layer 1 — Raw claim features (indices 0-7)
            min(1.0, policy_age / (365.0 * 5)),          # [0]  policy_age_norm (5yr max)
            1.0 if policy_age < 180 else 0.0,            # [1]  early_flag
            1.0 if policy_age < 30  else 0.0,            # [2]  very_early_flag
            1.0 if state.policy_revival_detected else 0.0, # [3] revival_flag
            min(1.0, float(state.policy_sum_assured or 0) / 10_000_000.0), # [4] sum_assured_norm
            1.0 if ec.get("premium_irregularities") else 0.0, # [5] premium_irregular
            min(1.0, age / 100.0),                       # [6]  age_norm
            1.0 if 0 < age < 35 else 0.0,               # [7]  young_death_flag

            # Layer 2 — Document quality features (indices 8-20)
            avg_ocr,                                     # [8]  avg_ocr
            min_ocr,                                     # [9]  min_ocr
            1.0 if avg_ocr < 0.60 else 0.0,             # [10] low_ocr_flag
            1.0 if min_ocr < 0.45 else 0.0,             # [11] very_low_ocr_flag
            min(1.0, len(state.submitted_documents) / 10.0), # [12] doc_count_norm
            1.0 if len(state.submitted_documents) < 2 else 0.0, # [13] insufficient_docs
            1.0 if ev.get("metadata_tampered") else 0.0, # [14] metadata_tampered
            1.0 if ev.get("image_manipulation_detected") else 0.0, # [15] image_manip
            0.0 if ev.get("hospital_verified") is False else 1.0,  # [16] hosp_verified
            0.0 if ev.get("doctor_verified")   is False else 1.0,  # [17] doc_verified
            0.0 if ev.get("fir_verified")       is False else 1.0,  # [18] fir_verified
            1.0 if ev.get("geo_location_match") is False else 0.0, # [19] geo_mismatch
            float(ev.get("verification_confidence", 0.80)),         # [20] verif_conf

            # Layer 3 — Cross-agent signal features (indices 21-31)
            self._EARLY_RISK_MAP.get(ec.get("early_claim_risk", "LOW"), 0.10), # [21]
            1.0 if ec.get("policy_revival_detected") else 0.0,  # [22] agent3_revival
            1.0 if nd.get("contradiction_detected") else 0.0,   # [23] non_disclosure
            float(nd.get("non_disclosure_score", 0.0)),          # [24] nd_score
            1.0 if cr.get("conflict_detected") else 0.0,        # [25] conflict_flag
            genuine_conflict,                                     # [26] genuine_conflict
            self._CONFLICT_SEVERITY_MAP.get(                     # [27] conflict_severity
                cr.get("resolved_action", "ACCEPT"), 0.0),
            1.0 if ga.get("fraud_ring_detected") else 0.0,      # [28] fraud_ring
            float(ga.get("graph_risk_score", 0.0)),              # [29] graph_risk
            min(1.0, float(ga.get("connected_claims", 0)) / 10.0), # [30] connected_norm
            min(1.0, float(len(state.validation_flags)) / 5.0), # [31] flag_count_norm
        ], dtype=np.float64)

        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC TRAINING DATA — 1,000 samples across 3 tiers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_training_data() -> np.ndarray:
    """
    Generate synthetic Isolation Forest training data.
    Must match _FeatureExtractor.extract() exactly (32 features).

    Three tiers:
      800 clean     — normal claims with old policies, high OCR, no flags
      150 suspicious — borderline claims with some elevated signals
       50 fraud     — clear fraud patterns: multiple simultaneous flags
    """
    rng = np.random.default_rng(42)

    # ── Clean claims (indices 0-799) ──────────────────────────────────────────
    # Characteristic: old policy, high OCR, verified, no cross-agent flags
    clean = rng.uniform(0.00, 0.12, (800, _N_FEATURES))
    # policy_age_norm [0] — old policies (0.5-1.0)
    clean[:, 0]  = rng.uniform(0.50, 1.00, 800)
    # early_flag [1] and very_early_flag [2] — no early claims
    clean[:, 1]  = np.zeros(800)
    clean[:, 2]  = np.zeros(800)
    # avg_ocr [8] and min_ocr [9] — high confidence docs
    clean[:, 8]  = rng.uniform(0.82, 0.99, 800)
    clean[:, 9]  = rng.uniform(0.80, 0.99, 800)
    # low_ocr [10], very_low_ocr [11] — clear
    clean[:, 10] = np.zeros(800)
    clean[:, 11] = np.zeros(800)
    # hospital_verified [16], doctor_verified [17], fir_verified [18] — all pass
    clean[:, 16] = rng.uniform(0.90, 1.00, 800)
    clean[:, 17] = rng.uniform(0.90, 1.00, 800)
    clean[:, 18] = rng.uniform(0.90, 1.00, 800)
    # verification_confidence [20] — high
    clean[:, 20] = rng.uniform(0.80, 0.99, 800)
    # early_claim_risk_norm [21] — LOW = 0.10
    clean[:, 21] = np.full(800, 0.10)
    # All cross-agent flags [23-31] — zero (no issues)
    clean[:, 23:] = rng.uniform(0.00, 0.05, (800, 9))

    # ── Suspicious claims (indices 800-949) ───────────────────────────────────
    # Characteristic: 1-2 elevated signals but not full fraud
    susp = rng.uniform(0.05, 0.35, (150, _N_FEATURES))
    # Some have early policy (not all)
    early_mask = rng.random(150) < 0.5
    susp[early_mask, 0] = rng.uniform(0.05, 0.30, early_mask.sum())  # younger policy
    susp[early_mask, 1] = 1.0  # early_flag
    # Some have medium OCR
    susp[:, 8]  = rng.uniform(0.55, 0.80, 150)
    susp[:, 9]  = rng.uniform(0.45, 0.75, 150)
    # Some verification failures
    susp[:, 16] = rng.uniform(0.40, 1.00, 150)
    susp[:, 17] = rng.uniform(0.50, 1.00, 150)
    susp[:, 18] = rng.uniform(0.50, 1.00, 150)
    susp[:, 21] = rng.uniform(0.10, 0.75, 150)  # mixed early risk

    # ── Fraud claims (indices 950-999) ────────────────────────────────────────
    # Characteristic: very early policy + multiple verification failures +
    #                 tampered docs + cross-agent positive signals
    fraud = rng.uniform(0.00, 0.15, (50, _N_FEATURES))
    # Very early policy
    fraud[:, 0]  = rng.uniform(0.00, 0.15, 50)  # new policy
    fraud[:, 1]  = np.ones(50)                   # early_flag
    fraud[:, 2]  = rng.uniform(0.50, 1.00, 50)   # many are very early
    # Low OCR
    fraud[:, 8]  = rng.uniform(0.25, 0.55, 50)
    fraud[:, 9]  = rng.uniform(0.20, 0.50, 50)
    fraud[:, 10] = np.ones(50)
    fraud[:, 11] = rng.uniform(0.50, 1.00, 50)
    # Tampered docs
    fraud[:, 14] = rng.uniform(0.60, 1.00, 50)   # metadata_tampered
    fraud[:, 15] = rng.uniform(0.40, 1.00, 50)   # image_manipulation
    # Verification failures
    fraud[:, 16] = rng.uniform(0.00, 0.30, 50)   # hospital failed
    fraud[:, 17] = rng.uniform(0.00, 0.30, 50)   # doctor failed
    fraud[:, 18] = rng.uniform(0.00, 0.40, 50)   # fir failed
    fraud[:, 19] = rng.uniform(0.60, 1.00, 50)   # geo_mismatch
    fraud[:, 20] = rng.uniform(0.00, 0.30, 50)   # low verif_confidence
    # High cross-agent signals
    fraud[:, 21] = rng.uniform(0.75, 1.00, 50)   # early_claim_risk HIGH/VERY_HIGH
    fraud[:, 23] = rng.uniform(0.70, 1.00, 50)   # non_disclosure
    fraud[:, 24] = rng.uniform(0.60, 1.00, 50)   # non_disclosure_score
    fraud[:, 25] = rng.uniform(0.60, 1.00, 50)   # conflict_detected
    fraud[:, 26] = rng.uniform(0.70, 1.00, 50)   # genuine_conflict
    fraud[:, 28] = rng.uniform(0.70, 1.00, 50)   # fraud_ring
    fraud[:, 29] = rng.uniform(0.60, 1.00, 50)   # graph_risk_score
    fraud[:, 30] = rng.uniform(0.40, 1.00, 50)   # connected_claims_norm
    fraud[:, 31] = rng.uniform(0.60, 1.00, 50)   # validation_flag_count

    return np.vstack([clean, susp, fraud])


# ── Build and train model at module load time ─────────────────────────────────
_training_data = _build_training_data()

_isolation_forest = IsolationForest(
    n_estimators=_IF_N_ESTIMATORS,
    contamination=_IF_CONTAMINATION,
    random_state=_IF_RANDOM_STATE,
    max_samples=_IF_MAX_SAMPLES,
)
_isolation_forest.fit(_training_data)

logger.info(
    "[Agent 2] Isolation Forest trained: %d samples, %d features",
    len(_training_data),
    _N_FEATURES,
)

_extractor = _FeatureExtractor()


def _compute_anomaly_score(features: np.ndarray) -> float:
    """
    Maps IsolationForest.decision_function → [0, 1].
    decision_function returns negative for anomalies.
    Typical range: [-0.5, +0.5]
    Map: -0.5 → 1.0 (maximum anomaly), +0.5 → 0.0 (perfectly normal)
    """
    try:
        raw = _isolation_forest.decision_function(features.reshape(1, -1))[0]
        score = float(np.clip((0.5 - raw), 0.0, 1.0))
        return round(score, 4)
    except Exception as exc:
        logger.warning("[Agent 2] Anomaly score failed: %s", exc)
        return 0.40   # neutral fallback — does not penalise clean claims


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class FraudIntelligenceAgent:
    """
    Fraud scoring engine for insurance claims.

    Scoring architecture:
      1. Rule-based layer  — 10 domain-knowledge rules + compound amplifier
         Explainable, deterministic, audit-friendly.
      2. ML anomaly layer  — Isolation Forest on 32-dimensional feature vector
         Catches statistical patterns no rule explicitly covers.

    Final score = rule_score × 0.60 + anomaly_score × 0.40
    """
    def __init__(self, redis_client=None):
        self.llm = LLMService(redis_client=redis_client)
        
        gemini_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'agent2_gemini.txt')
        with open(gemini_path, 'r') as f:
            self.gemini_prompt = f.read()
            
        local_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'agent2_local.txt')
        with open(local_path, 'r') as f:
            self.local_prompt = f.read()

    def run(self, state: ClaimState) -> ClaimState:
        logger.info("[Agent 2] Fraud Intelligence → %s", state.claim_case_id)

        # ── Step 1: Extract features (for ML) ─────────────────────────────────
        features = _extractor.extract(state)

        # ── Step 2: Extract rule signals (for rules + reasons) ────────────────
        signals, reasons = self._extract_signals(state)

        # ── Step 3: ML anomaly score ───────────────────────────────────────────
        anomaly_score = _compute_anomaly_score(features)

        # ── Step 4: Rule-based score ───────────────────────────────────────────
        rule_score = sum(
            SIGNAL_WEIGHTS.get(sig, 0.10)
            for sig, fired in signals.items()
            if fired
        )
        rule_score = round(min(0.99, rule_score), 4)

        # ── Step 5: Combined fraud score ───────────────────────────────────────
        fraud_score = round(min(0.99, rule_score * 0.60 + anomaly_score * 0.40), 4)

        # ── Step 6: Risk level ─────────────────────────────────────────────────
        risk_level = "LOW"
        for threshold, level in RISK_LEVELS:
            if fraud_score >= threshold:
                risk_level = level
                break

        # Override: critical combination check regardless of blended score
        if rule_score >= 0.60 and sum(signals.values()) >= 3:
            risk_level = "CRITICAL"

        action = RECOMMENDED_ACTIONS[risk_level]
        trust  = round(max(0.05, 1.0 - fraud_score), 4)

        # ── Step 7: Add ML context to reasons ────────────────────────────────
        if anomaly_score >= 0.72:
            reasons.append(
                f"ML anomaly detection: claim profile is highly unusual "
                f"compared to population baseline "
                f"(anomaly_score={anomaly_score:.2f})"
            )
        elif anomaly_score >= 0.50:
            reasons.append(
                f"ML anomaly detection: claim shows unusual patterns "
                f"across {_N_FEATURES} signals "
                f"(anomaly_score={anomaly_score:.2f})"
            )

        logger.info(
            "[Agent 2] %s | rule=%.3f anomaly=%.3f final=%.3f "
            "level=%s signals=%d",
            state.claim_case_id,
            rule_score, anomaly_score, fraud_score,
            risk_level, sum(signals.values()),
        )

        # ── Step 8: Gemini API Syndicate Detection ───────────────────────────
        age_str = str(int(state.life_assured.age or state.life_assured.age_at_death or 0))
        g_prompt = self.gemini_prompt.replace("{age}", age_str)\
            .replace("{occupation}", state.life_assured.occupation or "Unknown")\
            .replace("{policy_duration}", str(max(0, state.policy_age_days or 0)))\
            .replace("{death_location}", state.death_information.place_of_death or "Unknown")\
            .replace("{cause_of_death}", state.death_information.cause_of_death or "Unknown")\
            .replace("{nominee_relationship}", state.claimant.relationship_to_life_assured or "Unknown")
        syndicate_detected = False
        
        logger.info("[Agent 2] Calling Gemini for Syndicate Detection...")
        g_result = self.llm.route_to_gemini(g_prompt, enforce_json=True)
        if g_result and g_result.get("syndicate_pattern_detected"):
            syndicate_detected = True
            reasons.append(f"Gemini AI detected potential Syndicate: {g_result.get('pattern_description')}")

        # ── Step 9: Local LLM Human Explanation ──────────────────────────────
        local_prompt = self.local_prompt.replace("{list_of_anomalies}", ", ".join(reasons))
        logger.info("[Agent 2] Calling Local LLM for Anomaly Summary...")
        explanation = self.llm.route_to_local(local_prompt)
        if explanation:
            state.anomaly_explanation = explanation

        output = FraudIntelligenceOutput(
            fraud_risk_score   = fraud_score,
            fraud_risk_level   = risk_level,
            fraud_reasons      = reasons,
            anomaly_score      = anomaly_score,
            suspicious_nominee = signals.get("repeated_nominee", False),
            recommended_action = action,
            confidence_score   = round(
                0.85 if sum(signals.values()) >= 3 else 0.70, 4
            ),
            trust_score        = trust,
            validation_flags   = (
                [f"FRAUD_RISK:{risk_level}"] if risk_level != "LOW" else []
            ),
        )

        state.fraud_analysis = output.model_dump()

        if risk_level in ("HIGH", "CRITICAL"):
            flag = f"FRAUD_RISK:{risk_level}"
            if flag not in state.validation_flags:
                state.validation_flags.append(flag)

        return state

    # ══════════════════════════════════════════════════════════════════════════
    # RULE SIGNAL EXTRACTION — deterministic, explainable
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_signals(
        self,
        state: ClaimState,
    ) -> Tuple[Dict[str, bool], List[str]]:
        """
        Extract boolean fraud signals and human-readable reasons.
        Signals map directly to SIGNAL_WEIGHTS keys for scoring.
        Every reason appears verbatim in the output fraud_reasons list.
        """
        signals: Dict[str, bool] = {}
        reasons: List[str]       = []

        ev = state.external_verification    or {}
        nd = state.non_disclosure_analysis  or {}
        cr = state.conflict_resolution      or {}
        ga = state.graph_analysis           or {}

        policy_age = max(0, state.policy_age_days or 0)

        # ── Signal 1: Policy age — timeline anomaly ────────────────────────────
        if 0 < policy_age < 30:
            signals["policy_very_early"] = True
            reasons.append(
                f"Policy issued only {policy_age} day(s) before death "
                f"(EXTREMELY suspicious — under 30 days)"
            )
        elif 0 < policy_age < 180:
            signals["policy_early"] = True
            reasons.append(
                f"Policy issued {policy_age} days before death "
                f"(early claim indicator — under 6 months)"
            )

        # ── Signal 2: Policy revival ───────────────────────────────────────────
        if state.policy_revival_detected:
            signals["policy_revival"] = True
            reasons.append(
                "Policy was revived before death — "
                "revival-then-early-death is a known fraud pattern"
            )

        # ── Signal 3: Metadata tampered ────────────────────────────────────────
        if ev.get("metadata_tampered"):
            signals["metadata_tampered"] = True
            reasons.append(
                "Document metadata shows signs of tampering — "
                "creation/modification timestamps are inconsistent"
            )

        # ── Signal 4: Image manipulation ───────────────────────────────────────
        if ev.get("image_manipulation_detected"):
            signals["image_manipulation"] = True
            reasons.append(
                "Image forensics detected possible pixel-level manipulation — "
                "document authenticity is compromised"
            )

        # ── Signal 5: OCR confidence ────────────────────────────────────────────
        ocr_scores = list(state.ocr_confidence_scores.values())
        if not ocr_scores:
            ocr_scores = [
                d.ocr_confidence for d in state.submitted_documents
                if d.ocr_confidence > 0
            ]
        if ocr_scores:
            avg_ocr = sum(ocr_scores) / len(ocr_scores)
            min_ocr = min(ocr_scores)
            if min_ocr < 0.45:
                signals["very_low_ocr"] = True
                reasons.append(
                    f"Critical OCR confidence on at least one document "
                    f"(min={min_ocr:.2f}) — document may be forged or severely degraded"
                )
            elif avg_ocr < 0.60:
                signals["low_ocr_confidence"] = True
                reasons.append(
                    f"Low average OCR confidence ({avg_ocr:.2f}) — "
                    f"document quality raises authenticity concerns"
                )

        # ── Signal 6: Hospital verification failed ─────────────────────────────
        if ev.get("hospital_verified") is False:
            signals["hospital_not_verified"] = True
            reasons.append(
                "Hospital could not be verified in national registry — "
                "may be unregistered or fictitious facility"
            )

        # ── Signal 7: Doctor verification failed ───────────────────────────────
        if ev.get("doctor_verified") is False:
            signals["doctor_not_verified"] = True
            reasons.append(
                "Attending doctor could not be verified — "
                "medical credentials are unconfirmed"
            )

        # ── Signal 8: FIR + geo failures ───────────────────────────────────────
        if ev.get("fir_verified") is False:
            signals["fir_not_verified"] = True
            reasons.append("FIR record could not be verified with police records")

        if ev.get("geo_location_match") is False:
            signals["geo_mismatch"] = True
            reasons.append(
                "Geographic location mismatch — "
                "document metadata location inconsistent with claimed death location"
            )

        # ── Signal 9: Repeated nominee (PostgreSQL runtime check) ─────────────
        nominee_id = state.claimant.nominee_id or ""
        nominee_freq_threshold = int(os.environ.get("NOMINEE_FREQ_THRESHOLD", 2))
        graph_nom  = int(ga.get("shared_nominees", 0))
        try:
            nom_count, nom_claims = get_nominee_claim_count(nominee_id)
            repeat_nom = bool(nominee_id and nom_count >= nominee_freq_threshold)
        except Exception:
            nom_count, nom_claims, repeat_nom = 0, [], False
        if repeat_nom or graph_nom >= 1:
            signals["repeated_nominee"] = True
            if graph_nom >= 1:
                reasons.append(
                    f"Nominee confirmed across {graph_nom} other claim(s) "
                    f"by graph analysis — potential fraud ring participant"
                )
            else:
                reasons.append(
                    f"Nominee '{nominee_id}' appears in {nom_count} prior claim(s) "
                    f"(DB frequency check — threshold {nominee_freq_threshold})"
                )

        # ── Signal 10: Repeated bank account (PostgreSQL runtime check) ────────
        account = state.claimant.bank_account or ""
        bank_freq_threshold = int(os.environ.get("BANK_ACCOUNT_FREQ_THRESHOLD", 2))
        graph_bank = int(ga.get("shared_bank_accounts", 0))
        try:
            bank_count, bank_claims = get_bank_account_claim_count(account)
            repeat_bank = bool(account and bank_count >= bank_freq_threshold)
        except Exception:
            bank_count, bank_claims, repeat_bank = 0, [], False
        if repeat_bank or graph_bank >= 1:
            signals["repeated_bank_account"] = True
            if graph_bank >= 1:
                reasons.append(
                    f"Bank account confirmed across {graph_bank} other claim(s) "
                    f"by graph analysis — shared financial channel"
                )
            else:
                reasons.append(
                    f"Bank account '{account}' linked to {bank_count} prior claim(s) "
                    f"(DB frequency check — threshold {bank_freq_threshold})"
                )

        # ── Signal 11: Fraud ring (Agent 7) ───────────────────────────────────
        if ga.get("fraud_ring_detected"):
            signals["fraud_ring_detected"] = True
            reasons.append(
                "Fraud ring confirmed by graph engine (Agent 7) — "
                "claim is part of a suspicious network of connected claims"
            )

        # ── Signal 12: Non-disclosure (Agent 4) ───────────────────────────────
        if nd.get("contradiction_detected"):
            signals["non_disclosure_detected"] = True
            nd_score = float(nd.get("non_disclosure_score", 0.0))
            reasons.append(
                f"Non-disclosure confirmed by Agent 4 "
                f"(score={nd_score:.2f}) — "
                f"medical records contradict proposal form declarations"
            )

        # ── Signal 13: Genuine conflict (Agent 5) ─────────────────────────────
        cr_flags = cr.get("validation_flags", [])
        if cr.get("conflict_detected") and "GENUINE_CONFLICT_DETECTED" in cr_flags:
            signals["genuine_conflict"] = True
            reasons.append(
                "Genuine document conflict confirmed by Agent 5 — "
                "irreconcilable inconsistency found across submitted documents"
            )

        # ── Signal 14: Financial motivation ────────────────────────────────────
        sum_assured = float(state.policy_sum_assured or state.life_assured.sum_assured or 0)
        if sum_assured >= 10_000_000:
            signals["very_high_sum_assured"] = True
            reasons.append(
                f"Very high sum assured (₹{sum_assured:,.0f}) — "
                f"financial motive significantly elevated"
            )
        elif sum_assured >= 5_000_000:
            signals["high_sum_assured"] = True
            reasons.append(
                f"High sum assured (₹{sum_assured:,.0f}) — warrants additional scrutiny"
            )

        # ── Signal 15: Demographics ────────────────────────────────────────────
        age = int(state.life_assured.age or state.life_assured.age_at_death or 0)
        if 0 < age < 25:
            signals["very_young_death"] = True
            reasons.append(
                f"Very young age at death ({age} years) — "
                f"statistically unusual for natural causes"
            )
        elif 0 < age < 35:
            signals["young_death"] = True
            reasons.append(
                f"Young age at death ({age} years) — additional verification warranted"
            )

        # ── Signal 16: Compound amplifier ─────────────────────────────────────
        # Counts all non-compound signals that fired
        base_signal_count = sum(
            1 for k, v in signals.items()
            if v and not k.startswith("compound")
        )
        if base_signal_count >= 4:
            signals["compound_4plus"] = True
            reasons.append(
                f"{base_signal_count} independent fraud signals active simultaneously — "
                f"compound risk amplifier triggered (scores multiply)"
            )
        elif base_signal_count >= 2:
            signals["compound_2plus"] = True

        return signals, reasons
