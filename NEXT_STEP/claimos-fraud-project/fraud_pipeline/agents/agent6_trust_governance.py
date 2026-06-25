"""
Agent 6 — Trust Governance Agent

Responsibilities:
  - Aggregate trust signals from all upstream agents
  - Compute overall trust score and automation eligibility
  - Propagate confidence across the pipeline
  - Flag claims requiring human review
"""

import logging
import os
from ..schemas import ClaimState, TrustGovernanceOutput
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class TrustGovernanceAgent:
    """
    Aggregates all upstream trust signals into a single governance decision.
    Enriches ClaimState.trust_analysis.
    """

    def __init__(self, redis_client=None):
        self.llm = LLMService(redis_client=redis_client)
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'agent6_summary.txt')
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

    # Thresholds
    AUTO_APPROVE_THRESHOLD = 0.75      # Overall trust above this → auto approve eligible
    HUMAN_REVIEW_THRESHOLD = 0.55      # Below this → human review required
    OCR_MIN_TRUST = 0.55               # OCR trust below this reduces overall trust

    # Weights for each trust component
    WEIGHTS = {
        "ocr":      0.20,
        "external": 0.25,
        "fraud":    0.35,
        "conflict": 0.10,
        "nondisclosure": 0.10,
    }
    CRITICAL_FIELD_PATTERNS = (
        "causeofdeath",
        "dateofdeath",
        "registrationnumber",
        "aadhaarnumber",
        "pannumber",
        "firnumber",
    )

    def run(self, state: ClaimState) -> ClaimState:
        logger.info("[Agent 6] Trust Governance → claim %s", state.claim_case_id)

        reasons: list = []

        # ── 1. OCR Trust ──────────────────────────────────────────────────────
        ocr_trust = self._compute_ocr_trust(state, reasons)
        critical_field_trust = self._compute_critical_field_trust(state, reasons)
        if critical_field_trust is not None:
            ocr_trust = round((0.70 * ocr_trust) + (0.30 * critical_field_trust), 4)

        # ── 2. External Verification Trust ───────────────────────────────────
        ext = state.external_verification or {}
        external_trust = ext.get("trust_score", 0.80)
        if not ext.get("hospital_verified", True):
            external_trust *= 0.80
            reasons.append("Hospital verification failed — external trust reduced")
        if ext.get("metadata_tampered"):
            external_trust *= 0.60
            reasons.append("Metadata tampering detected — external trust heavily penalised")

        # ── 3. Fraud Confidence ───────────────────────────────────────────────
        fraud = state.fraud_analysis or {}
        fraud_risk_score  = fraud.get("fraud_risk_score", 0.0)
        fraud_confidence  = fraud.get("confidence_score", 0.5)
        fraud_trust_component = round(1.0 - fraud_risk_score, 4)
        if fraud_risk_score > 0.60:
            reasons.append(
                f"High fraud risk score ({fraud_risk_score:.2f}) — trust significantly reduced"
            )

        # ── 4. Conflict Trust — reads Agent 5 output correctly ────────────────
        conflict_trust = self._compute_conflict_trust(state)
        cr = state.conflict_resolution or {}
        if cr.get("conflict_detected") and \
                "GENUINE_CONFLICT_DETECTED" in cr.get("validation_flags", []):
            reasons.append("Genuine document conflict confirmed — trust reduced")
        elif cr.get("resolved_action") == "MANUAL_REVIEW":
            reasons.append("Conflict requires manual review — trust moderately reduced")

        # ── 5. Non-Disclosure Trust ──────────────────────────────────────────
        nd = state.non_disclosure_analysis or {}
        nd_trust = 0.90 if not nd.get("contradiction_detected") else 0.45
        if nd.get("contradiction_detected"):
            reasons.append("Non-disclosure contradiction detected — trust reduced")

        # ── Weighted aggregation ─────────────────────────────────────────────
        overall = (
            0.25 * ocr_trust            +
            0.25 * external_trust       +
            0.30 * fraud_trust_component +
            0.20 * conflict_trust
        )
        overall = round(min(0.99, max(0.01, overall)), 4)

        # ── Automation decision — reads flags not just action string ──────────
        automation_decision, human_review_required = \
            self._compute_automation_decision(overall, state, reasons)

        automation_eligible = automation_decision == "APPROVE" and not human_review_required

        if human_review_required:
            reasons.append(
                f"Overall trust {overall:.2f} — human review required"
            )

        output = TrustGovernanceOutput(
            ocr_trust_score=round(ocr_trust, 4),
            external_trust_score=round(external_trust, 4),
            fraud_confidence_score=round(fraud_confidence, 4),
            overall_trust_score=overall,
            human_review_required=human_review_required,
            automation_eligible=automation_eligible,
            automation_decision=automation_decision,
            trust_reduction_reasons=reasons,
            confidence_score=overall,
            trust_score=overall,
            validation_flags=["HUMAN_REVIEW_REQUIRED"] if human_review_required else [],
        )

        # ── 6. Generate LLM Executive Summary (Local LLM) ──────────────────────
        graph_data = state.graph_analysis or {}
        graph_links = f"{graph_data.get('connected_claims', 0)} claims, {graph_data.get('shared_bank_accounts', 0)} banks"
        
        ml_anomalies_str = ", ".join(state.fraud_analysis.get("fraud_reasons", [])) if state.fraud_analysis else "None"
        prompt = self.prompt_template\
            .replace("{trust_score}", str(int(overall * 100)))\
            .replace("{geo_flag}", "Suspicious Geo" if "GEO_MISMATCH" in state.validation_flags else "Clear")\
            .replace("{ml_anomalies}", ml_anomalies_str)\
            .replace("{medical_contradiction}", "Yes" if state.non_disclosure_analysis and state.non_disclosure_analysis.get("contradiction_detected") else "No")\
            .replace("{graph_links}", graph_links)
        
        logger.info("[Agent 6] Requesting Local LLM Executive Summary...")
        summary = self.llm.route_to_local(prompt)
        if summary:
            output.executive_summary = summary
        else:
            output.executive_summary = "LLM Summary unavailable. Proceed with raw numerical scores."

        state.trust_analysis = output.model_dump()
        if human_review_required:
            if "HUMAN_REVIEW_REQUIRED" not in state.validation_flags:
                state.validation_flags.append("HUMAN_REVIEW_REQUIRED")
            state.escalation_required = True
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_automation_decision(
        self,
        overall_trust:     float,
        state:             ClaimState,
        reduction_reasons: list,
    ):
        """
        Compute automation_decision and human_review_required.

        Reads Agent 5 output correctly:
          - Checks validation_flags for GENUINE_CONFLICT_DETECTED
            not just resolved_action string
          - Clinically valid relationships (layer 1/2/3 resolved)
            do NOT trigger HOLD
          - Only genuine confirmed conflicts trigger ESCALATE
          - Layer 4 fallback triggers MANUAL_REVIEW not ESCALATE
        """
        fraud_analysis      = state.fraud_analysis      or {}
        conflict_resolution = state.conflict_resolution or {}
        graph_analysis      = state.graph_analysis      or {}

        # Read fraud signals
        fraud_level  = fraud_analysis.get("fraud_risk_level", "LOW")
        fraud_score  = float(fraud_analysis.get("fraud_risk_score", 0.0))
        fraud_action = fraud_analysis.get("recommended_action", "APPROVE")
        fraud_ring   = graph_analysis.get("fraud_ring_detected", False)

        # Read conflict resolution output correctly
        # Check FLAGS not just resolved_action string
        conflict_flags       = conflict_resolution.get("validation_flags", [])
        conflict_detected    = conflict_resolution.get("conflict_detected", False)
        conflict_action      = conflict_resolution.get("resolved_action", "ACCEPT")
        layer_resolved       = conflict_resolution.get("resolution_layer", 0)
        medical_relationship = conflict_resolution.get(
            "medical_relationship_found", False
        )

        # ── Priority 1 — Hard ESCALATE signals ────────────────────────────────
        if fraud_ring:
            return "ESCALATE", True

        if fraud_level == "CRITICAL" or fraud_score >= 0.90:
            return "ESCALATE", True

        # Only escalate conflict when Agent 5 confirmed it is genuine
        # GENUINE_CONFLICT_DETECTED flag is only set when:
        #   - terms are genuinely unrelated (not just different names)
        #   - confidence >= 0.80
        #   - temporal impossibility detected
        #   - age discrepancy > 10 years
        if (conflict_detected
                and "GENUINE_CONFLICT_DETECTED" in conflict_flags):
            return "ESCALATE", True

        # ── Priority 2 — MANUAL_REVIEW signals ────────────────────────────────
        if "FRAUD_RING_DETECTED" in state.validation_flags:
            return "ESCALATE", True

        if fraud_level == "HIGH" or fraud_score >= 0.70:
            return "MANUAL_REVIEW", True

        if "NON_DISCLOSURE_DETECTED" in state.validation_flags:
            return "MANUAL_REVIEW", True

        # Layer 4 means ontology could not resolve — uncertain
        if layer_resolved == 4 and conflict_detected:
            return "MANUAL_REVIEW", True

        if conflict_action == "MANUAL_REVIEW":
            return "MANUAL_REVIEW", True

        if overall_trust < self.HUMAN_REVIEW_THRESHOLD:
            return "MANUAL_REVIEW", True

        # ── Priority 3 — Trust-based approval ─────────────────────────────────
        if overall_trust >= self.AUTO_APPROVE_THRESHOLD:
            return "APPROVE", False

        return "MANUAL_REVIEW", True

    def _compute_conflict_trust(self, state: ClaimState) -> float:
        """
        Compute trust contribution from Agent 5 conflict resolution output.

        Key fix:
          Previous: any ESCALATE action → 0.25 trust → tanks overall score
          Fixed:    read medical_relationship_found and layer_resolved
                    clinically valid relationships → high trust maintained

        Trust mapping:
          ACCEPT + no conflict                → 0.95  clean claim
          Clinically valid relationship found → 0.90  Agent 5 confirmed valid
          Layer 4 fired but no conflict       → 0.75  uncertain but not flagged
          MANUAL_REVIEW                       → 0.65  some uncertainty
          Genuine conflict confirmed          → 0.25  real problem found
        """
        cr = state.conflict_resolution or {}
        if not cr:
            return 0.90

        resolved_action      = cr.get("resolved_action", "ACCEPT")
        conflict_detected    = cr.get("conflict_detected", False)
        flags                = cr.get("validation_flags", [])
        medical_relationship = cr.get("medical_relationship_found", False)
        layer_resolved       = cr.get("resolution_layer", 0)

        # Clean claim — no conflicts of any kind
        if resolved_action == "ACCEPT" and not conflict_detected:
            return 0.95

        # Agent 5 confirmed a clinically valid medical relationship
        if medical_relationship and not conflict_detected:
            return 0.90

        # Layer 4 fired but no conflict flagged
        if layer_resolved == 4 and not conflict_detected:
            return 0.75

        # Uncertain cases — manual review but not confirmed conflict
        if resolved_action == "MANUAL_REVIEW" and not conflict_detected:
            return 0.65

        # Genuine confirmed conflict — strong trust penalty
        if conflict_detected and "GENUINE_CONFLICT_DETECTED" in flags:
            return 0.25

        # Default moderate trust
        return 0.70

    def _compute_ocr_trust(self, state: ClaimState, reasons: list) -> float:
        """Average OCR confidence across all submitted documents."""
        scores = list(state.ocr_confidence_scores.values())
        if not scores:
            scores = [d.ocr_confidence for d in state.submitted_documents if d.ocr_confidence > 0]

        if not scores:
            return 0.80   # Default when no OCR data

        avg = sum(scores) / len(scores)
        low_docs = [s for s in scores if s < self.OCR_MIN_TRUST]

        if low_docs:
            reasons.append(
                f"{len(low_docs)} document(s) with OCR confidence below {self.OCR_MIN_TRUST} "
                f"(lowest: {min(low_docs):.2f})"
            )
            # Penalise proportionally
            penalty = len(low_docs) / len(scores) * 0.30
            avg = max(0.05, avg - penalty)

        return round(avg, 4)

    def _confidence_to_ratio(self, value) -> float | None:
        try:
            val = float(value)
        except Exception:
            return None
        if val > 1.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))

    def _compute_critical_field_trust(self, state: ClaimState, reasons: list) -> float | None:
        """
        Compute OCR trust from critical extracted fields when field-level metadata is available.
        """
        scores = []
        for doc in state.submitted_documents:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            field_map = metadata.get("ocr_field_map")
            if not isinstance(field_map, dict):
                continue
            for field_path, field_data in field_map.items():
                key = str(field_path).replace(".", "").replace("_", "").lower()
                if not any(pattern in key for pattern in self.CRITICAL_FIELD_PATTERNS):
                    continue
                if not isinstance(field_data, dict):
                    continue
                conf = self._confidence_to_ratio(field_data.get("ocr_confidence"))
                if conf is not None:
                    scores.append(conf)

        if not scores:
            return None

        avg = round(sum(scores) / len(scores), 4)
        if avg < self.OCR_MIN_TRUST:
            reasons.append(
                f"Critical field OCR confidence is low ({avg:.2f}) across {len(scores)} extracted fields"
            )
        return avg
