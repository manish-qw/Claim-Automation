# =============================================================================
# orchestrator.py
# Chief Claim Processing Manager — ICICI Pru iProtect Smart
# Phase 3: Policy Reasoning & Autonomous Decision Layer
#
# Upgrades over original:
#   1. Fraud layer fully integrated into policy reasoning (non-disclosure,
#      suicide timing, hospital verification, early-claim flags)
#   2. Policy agent explicitly handles iProtect Smart clause routing
#      (death type → clause mapping, suicide 12-month window, WoP-PD
#      exclusions, benefit multiplier logic)
#   3. Non-disclosure detection gates policy coverage before amount calc
#   4. Review agent generates a structured ClaimDecisionReport (10 sections)
#   5. Decision router has 4 paths: APPROVED / PARTIALLY_APPROVED /
#      ESCALATE_HUMAN / REJECTED — replacing the vague approve/partial/reject
#   6. Escalation node packages a full context bundle for human reviewer
#   7. Every node appends a rich timestamped audit entry
#   8. Final print block renders a formatted, section-by-section report
# =============================================================================

import os
import json
from datetime import datetime, timezone
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, field_validator

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Project-local imports (unchanged filenames)
from state import ClaimState
from mock_data import get_initial_state
from rag_pipeline import retrieve_policy_clauses

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
load_dotenv()
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# ---------------------------------------------------------------------------
# LLM — use env-configured local model (defaults to llama3.2)
# ---------------------------------------------------------------------------
POLICY_LLM_MODEL = os.getenv(
    "POLICY_LLM_MODEL",
    os.getenv("LOCAL_LLM_MODEL", "llama3.2"),
)
llm = ChatOllama(model=POLICY_LLM_MODEL, temperature=0)


# =============================================================================
# SECTION 1 — PYDANTIC OUTPUT SCHEMAS
# =============================================================================

class PolicyOutput(BaseModel):
    """Structured output from the Policy Interpretation Agent."""
    is_covered: bool = Field(
        description="True if the policy explicitly covers this claim. False if excluded or unresolvable."
    )
    coverage_verdict: str = Field(
        description="One of: COVERED / PARTIALLY_COVERED / NOT_COVERED / REQUIRES_ESCALATION"
    )
    gross_covered_amount: float = Field(
        description="Gross benefit amount before deductibles. 0.0 if not covered."
    )
    deductible_applied: float = Field(
        description="Deductible or reduction applied per policy terms. 0.0 if none."
    )
    net_coverage_amount: float = Field(
        description="gross_covered_amount minus deductible_applied."
    )
    cited_policy_section: str = Field(
        description="Exact clause/section of the policy relied upon. Must not be blank if is_covered=True."
    )
    reasoning: str = Field(
        description="Step-by-step reasoning grounded in the retrieved policy text."
    )
    exclusions_triggered: List[str] = Field(
        default_factory=list,
        description="List of policy exclusions that apply, each with clause reference."
    )
    uncertainty_flags: List[str] = Field(
        default_factory=list,
        description="Unresolvable ambiguities that must be surfaced to the reviewer."
    )
    confidence_score: float = Field(
        description="Policy agent's own confidence in this determination. 0.0–1.0."
    )


class ClaimDecisionReport(BaseModel):
    """
    Full structured decision report produced by the Review Agent.
    This is the terminal deliverable of the pipeline — no payment is triggered here.
    """
    # ── Identity & Overview ──────────────────────────────────────────────────
    claim_id: str = Field(description="Claim identifier passed through from state.")
    policy_id: str = Field(description="Policy number.")
    claimant_name: str = Field(description="Name of the claimant.")
    life_assured_name: str = Field(description="Name of the life assured.")
    incident_type: str = Field(description="NATURAL / ACCIDENTAL / SUICIDE / TERMINAL_ILLNESS")
    incident_date: str = Field(description="Date of the insured event.")

    # ── Documents ────────────────────────────────────────────────────────────
    documents_reviewed: List[str] = Field(
        description="List of document types reviewed and their OCR confidence."
    )
    document_flags: List[str] = Field(
        default_factory=list,
        description="Any inconsistencies or low-confidence flags from Stage 2."
    )

    # ── Fraud & Risk ─────────────────────────────────────────────────────────
    fraud_risk_level: str = Field(description="LOW / MEDIUM / HIGH from Stage 3.")
    fraud_risk_score: float = Field(description="Numeric fraud score 0.0–1.0.")
    fraud_reasoning_summary: str = Field(
        description="Plain-language summary of the fraud agent's key findings."
    )
    non_disclosure_detected: bool = Field(
        description="True if Stage 3 detected a material non-disclosure."
    )
    non_disclosure_detail: str = Field(
        default="",
        description="Description of the non-disclosure if detected."
    )
    hospital_verified: bool = Field(
        description="Whether the treating hospital was verified against the registry."
    )
    early_claim_flag: bool = Field(
        description="True if the policy was less than 180 days old at time of claim."
    )
    suicide_window_flag: bool = Field(
        description="True if death occurred within 12 months of policy inception/revival."
    )

    # ── Policy Coverage ──────────────────────────────────────────────────────
    policy_coverage_verdict: str = Field(
        description="COVERED / PARTIALLY_COVERED / NOT_COVERED / REQUIRES_ESCALATION"
    )
    gross_covered_amount: float = Field(description="Gross benefit before deductions.")
    deductible_applied: float = Field(description="Deductions applied per policy.")
    net_coverage_amount: float = Field(description="Final net payable amount.")
    policy_clauses_cited: List[str] = Field(
        description="All policy clause references used in this determination."
    )
    exclusions_applied: List[str] = Field(
        default_factory=list,
        description="Policy exclusions triggered, each with clause reference."
    )

    # ── Areas of Uncertainty ─────────────────────────────────────────────────
    uncertainty_flags: List[str] = Field(
        default_factory=list,
        description="Unresolved ambiguities that a human reviewer must address."
    )

    # ── Final Verdict ────────────────────────────────────────────────────────
    final_decision: str = Field(
        description="APPROVED / PARTIALLY_APPROVED / ESCALATE_HUMAN / REJECTED"
    )
    decision_reasoning: str = Field(
        description=(
            "Full plain-language reasoning chain linking fraud findings, "
            "policy clauses, and the final verdict. IRDAI-defensible."
        )
    )
    confidence_score: float = Field(
        description="Reviewer's composite confidence in this automated decision. 0.0–1.0."
    )
    escalation_reason: str = Field(
        default="",
        description="If final_decision = ESCALATE_HUMAN, the precise reason for escalation."
    )

    @staticmethod
    def _coerce_string_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"none", "null", "n/a", "na", "not specified"}:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
            if ";" in text:
                return [part.strip() for part in text.split(";") if part.strip()]
            return [text]
        return [str(value).strip()] if str(value).strip() else []

    @field_validator(
        "documents_reviewed",
        "document_flags",
        "policy_clauses_cited",
        "exclusions_applied",
        "uncertainty_flags",
        mode="before",
    )
    @classmethod
    def _normalize_list_fields(cls, value):
        return cls._coerce_string_list(value)


# =============================================================================
# SECTION 2 — PROMPT TEMPLATES
# =============================================================================

# ── Policy Interpretation Prompt ─────────────────────────────────────────────
POLICY_PROMPT = """
You are a strict, IRDAI-compliant Insurance Policy Adjuster AI for ICICI Pru iProtect Smart.
You resolve claim coverage questions using ONLY the retrieved policy text below.

=== CLAIM DETAILS ===
Claim ID          : {claim_id}
Claimant          : {claimant_name}
Life Assured      : {life_assured_name}
Incident Type     : {incident_type}
Incident Date     : {incident_date}
Cause of Death    : {cause_of_death}
Death Type        : {death_type}
Months Since Inception: {months_since_inception}
Policy In Force   : {is_policy_in_force}
Premium Pay Type  : {premium_payment_type}
Age at Entry      : {age_at_entry}
Sum Assured       : {sum_assured}
Annualised Premium: {annualised_premium}
Payout Option     : {payout_option}

=== FRAUD CONTEXT (from Stage 3) ===
Fraud Risk Level  : {fraud_risk_level}
Fraud Risk Score  : {fraud_risk_score}
Non-Disclosure    : {non_disclosure_detected}
Non-Disclosure Detail: {non_disclosure_detail}
Hospital Verified : {hospital_verified}
Suicide Window    : {suicide_window_flag}
Early Claim (<180d): {early_claim_flag}

=== CONFLICT TO RESOLVE ===
{conflict_reason}

=== RETRIEVED POLICY TEXT ===
{retrieved_clauses}

=== YOUR MANDATORY INSTRUCTIONS ===

STEP 1 — POLICY IN FORCE CHECK
If is_policy_in_force is False, set coverage_verdict = NOT_COVERED.
Cite Part C, Clause 1.1: "provided the Policy is in force as on the date of death."

STEP 2 — DEATH TYPE ROUTING
Route based on death_type:
  a) NATURAL   → check Part C Clause 1.1 (Death Benefit). 
                 The system has pre-calculated the maximum eligible gross_covered_amount as: {system_calculated_gross}.
                 You MUST use {system_calculated_gross} as the gross_covered_amount. Do not calculate it yourself.
  b) SUICIDE   → If months_since_inception <= 12: apply Part F Clause 11.
                 Return 80% of premiums paid only. NOT full death benefit.
                 If months_since_inception > 12: treat as natural death under Clause 1.1.
  c) TERMINAL_ILLNESS → Part C Clause 1.1 TI sub-clause: two independent specialist
                 certifications required. Flag as REQUIRES_ESCALATION if not confirmed.
  d) ACCIDENTAL + PERMANENT_DISABILITY → Part C Clause 1.2 (WoP-PD).
                 BLOCKED if premium_payment_type = SINGLE (policy explicitly excludes Single Pay).
                 Check exclusions: suicide attempt, war, hazardous sports, criminal act, aerial flights.

STEP 3 — NON-DISCLOSURE CHECK
If non_disclosure_detected = True:
  Flag this as an uncertainty_flag.
  Reference Part F Clause 5 and Annexure IV (Section 45).
  If months_since_inception <= 36 months: insurer may repudiate on fraud grounds.
  Coverage verdict should be REQUIRES_ESCALATION — do not auto-approve.

STEP 4 — HOSPITAL VERIFICATION
If hospital_verified = False: add to uncertainty_flags. Note this does not block coverage
but must be flagged for manual verification.

STEP 5 — CONFLICT RESOLUTION
Resolve the specific conflict_reason using retrieved policy text only.
If the policy text semantically links the conflicting terms, state so with clause reference.
If not resolvable from text: add to uncertainty_flags.

STEP 6 — FINAL DETERMINATION
Set coverage_verdict, compute gross/net amounts, list all clauses cited.
Set confidence_score: reduce by 0.15 for each uncertainty_flag, by 0.20 for non-disclosure,
by 0.10 for unverified hospital, by 0.25 for suicide window.

You must cite a specific policy section for EVERY coverage determination.
Never invent clauses. If text does not support a claim, say so.
"""


# ── Review / Decision Prompt ──────────────────────────────────────────────────
REVIEW_PROMPT = """
You are the Lead Claim Decision Officer AI for ICICI Pru iProtect Smart.
You receive the full claim context and the Policy Agent's findings.
You produce the final ClaimDecisionReport — the terminal deliverable of this pipeline.
No payment is triggered by this report. Your job ends at the decision document.

=== COMPLETE CLAIM CONTEXT ===

CLAIM IDENTITY
  Claim ID         : {claim_id}
  Policy ID        : {policy_id}
  Claimant         : {claimant_name}
  Life Assured     : {life_assured_name}
  Incident Type    : {incident_type}
  Incident Date    : {incident_date}

DOCUMENTS REVIEWED
  {documents_reviewed}
  Document Flags   : {document_flags}

FRAUD INTELLIGENCE (Stage 3 Output)
  Fraud Risk Level : {fraud_risk_level}
  Fraud Risk Score : {fraud_risk_score}
  Fraud Reasoning  : {fraud_reasoning_summary}
  Non-Disclosure   : {non_disclosure_detected}
  Non-Disclosure Detail: {non_disclosure_detail}
  Hospital Verified: {hospital_verified}
  Early Claim Flag : {early_claim_flag}
  Suicide Window   : {suicide_window_flag}
  Stage 3 Confidence: {stage3_confidence}

POLICY AGENT FINDINGS (Stage 4 Output)
  Coverage Verdict : {policy_coverage_verdict}
  Gross Amount     : {gross_covered_amount}
  Deductible       : {deductible_applied}
  Net Amount       : {net_coverage_amount}
  Clauses Cited    : {policy_clauses_cited}
  Exclusions       : {exclusions_applied}
  Uncertainty Flags: {policy_uncertainty_flags}
  Policy Confidence: {policy_confidence}

=== YOUR MANDATORY DECISION RULES ===

RULE 1 — AUTOMATIC REJECTION (set final_decision = REJECTED):
  - suicide_window_flag = True AND months_since_inception <= 12
    (Per Part F Clause 11: only 80% premiums returned, not full benefit)
  - fraud_risk_score >= 0.85 AND fraud_ring_detected = True
  - policy_coverage_verdict = NOT_COVERED with a valid policy exclusion cited

RULE 2 — MANDATORY HUMAN ESCALATION (set final_decision = ESCALATE_HUMAN):
  - non_disclosure_detected = True (Section 45 review required within 3 years)
  - hospital_verified = False AND fraud_risk_score > 0.50
  - policy_coverage_verdict = REQUIRES_ESCALATION
  - More than 2 uncertainty_flags present
  - stage3_confidence < 0.75 (upstream confidence too low to auto-decide)

RULE 3 — PARTIAL APPROVAL (set final_decision = PARTIALLY_APPROVED):
  - policy_coverage_verdict = PARTIALLY_COVERED
  - OR fraud_risk_score between 0.30 and 0.60 with no non-disclosure
  - State exactly which portion is approved and which requires further review

RULE 4 — FULL APPROVAL (set final_decision = APPROVED):
  - policy_coverage_verdict = COVERED
  - fraud_risk_score < 0.30
  - non_disclosure_detected = False
  - hospital_verified = True OR early_claim_flag = False
  - No uncertainty_flags OR only minor flags that do not affect coverage

CONFIDENCE SCORING:
  Start at 1.0. Deduct:
  -0.25 for non_disclosure_detected = True
  -0.20 for fraud_risk_score > 0.50
  -0.10 for hospital_verified = False
  -0.15 for each uncertainty_flag (max 3 deductions)
  -0.10 for early_claim_flag = True
  -0.30 for suicide_window_flag = True
  Floor at 0.0.

DECISION REASONING must:/
  1. Reference specific fraud findings from Stage 3
  2. Reference specific policy clauses from Stage 4
  3. State what the net payable amount is (or why it cannot be determined)
  4. NEVER use the '$' symbol. You MUST use '₹' or 'INR' for all currency.
  5. Be written in plain language suitable for IRDAI regulatory review

Populate ALL fields of the ClaimDecisionReport schema completely.
"""


# =============================================================================
# SECTION 3 — HELPER: AUDIT ENTRY
# =============================================================================

def audit_entry(step: str, details: str, confidence: float = None, flags: list = None) -> dict:
    """Creates a rich, timestamped audit log entry."""
    entry = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }
    if confidence is not None:
        entry["confidence"] = round(confidence, 4)
    if flags:
        entry["flags"] = flags
    return entry


# =============================================================================
# SECTION 4 — HELPER: SAFE STATE EXTRACTION
# =============================================================================

def _safe(state: dict, *keys, default="N/A"):
    """Safely traverses nested dict keys. Returns default if any key is missing."""
    val = state
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
        if val == default:
            return default
    return val if val is not None else default


def _extract_fraud_fields(state: dict) -> dict:
    """
    Pulls all fraud-related fields from state.

    FIXED: reads non_disclosure_analysis, external_verification, and
    early_claim_analysis directly from top-level state — these are placed
    there by mock_data.get_initial_state() (or your live fraud pipeline
    adapter). No nested fallback into fraud_analysis — that fallback was
    always returning {} silently and masking real fraud signals.
    """
    fa        = state.get("fraud_analysis") or {}
    non_disc  = state.get("non_disclosure_analysis") or {}
    ext_verif = state.get("external_verification") or {}
    early     = state.get("early_claim_analysis") or {}

    # trust_analysis lives at top-level too if you choose to add it to state;
    # otherwise fall back gracefully to fraud_analysis confidence fields.
    trust = state.get("trust_analysis") or {}

    return {
        "fraud_risk_score":        fa.get("fraud_risk_score", 0.0),
        "fraud_risk_level":        fa.get("fraud_risk_level", "LOW"),
        "fraud_reasoning_summary": fa.get("fraud_reasons") or ["No specific reasons provided."],
        "non_disclosure_detected": non_disc.get("contradiction_detected", False),
        "non_disclosure_detail":   "; ".join(non_disc.get("non_disclosure_findings", [])),
        "hospital_verified":       ext_verif.get("hospital_verified", True),
        "early_claim_flag":        early.get("early_claim_risk", "LOW") in ("HIGH", "VERY_HIGH"),
        "stage3_confidence":       trust.get("confidence_score", fa.get("confidence_score", 0.85)),
        "overall_trust_score":     trust.get("overall_trust_score", fa.get("trust_score", 0.85)),
    }


def _extract_claim_fields(state: ClaimState) -> dict:
    """Robustly extracts claim identity from various mock data formats."""
    cd = state.get("claim_details") or {}
    ext = state.get("extracted_data") or {}
    
    # Check multiple possible paths for the mock data
    claimant = state.get("claimant_name") or _safe(cd, "claimant", "name") or _safe(ext, "claimant_identity", "name", default="John Doe")
    policy = state.get("policy_id") or cd.get("policy_number") or ext.get("policy_reference", "POL-998877")
    
    return {
        "claim_id":          state.get("claim_id", "CLM-001"),
        "policy_id":         policy,
        "claimant_name":     claimant,
        "life_assured_name": state.get("life_assured_name", claimant),
        "incident_type":     state.get("incident_type", "NATURAL"),
        "incident_date":     state.get("incident_date", state.get("death_date", "2026-05-10")),
        "cause_of_death":    state.get("cause_of_death", "Cardiac Arrest"),
        "death_type":        state.get("death_type", "NATURAL"),
        "months_since_inception": state.get("months_since_inception", 48),
        "is_policy_in_force":state.get("is_policy_in_force", True),
        "premium_payment_type": state.get("premium_payment_type", "REGULAR"),
        "age_at_entry":      state.get("life_assured_age_at_entry", 40),
        "sum_assured":       state.get("sum_assured", 100000),
        "annualised_premium":state.get("annualised_premium", 105000),
        "payout_option":     state.get("payout_option_chosen", "LUMP_SUM"),
        "documents_verified": state.get("documents_verified", []),
        "document_flags":     state.get("document_flags", []),
    }


# =============================================================================
# SECTION 5 — AGENT NODES
# =============================================================================

# ── Node 1: Pre-Flight Fraud Gate ─────────────────────────────────────────────
def fraud_gate_node(state: ClaimState) -> dict:
    """
    Reads Stage 3 fraud output. Flags suicide window, non-disclosure, and
    early-claim risk. These flags are injected into the policy agent's prompt
    so the policy agent can apply the correct clause routing.
    """
    print("-> [Gate] Fraud signal pre-flight check...")

    fraud = _extract_fraud_fields(state)
    claim = _extract_claim_fields(state)

    flags = []
    suicide_window = claim["months_since_inception"] <= 12
    if suicide_window:
        flags.append(f"SUICIDE_WINDOW: Death within 12 months of inception ({claim['months_since_inception']} months)")
    if fraud["non_disclosure_detected"]:
        flags.append(f"NON_DISCLOSURE: {fraud['non_disclosure_detail']}")
    if not fraud["hospital_verified"]:
        flags.append("HOSPITAL_UNVERIFIED: Treating hospital not in registry")
    if fraud["early_claim_flag"]:
        flags.append("EARLY_CLAIM: Policy age < 180 days at time of claim")
    if fraud["fraud_risk_score"] >= 0.85:
        flags.append(f"CRITICAL_FRAUD_SCORE: {fraud['fraud_risk_score']} >= 0.85")

    return {
        "suicide_window_flag": suicide_window,
        "fraud_gate_flags": flags,
        "audit_log": [audit_entry(
            "Fraud Gate",
            f"Pre-flight check complete. {len(flags)} flag(s) raised: {flags}",
            confidence=fraud["stage3_confidence"],
            flags=flags
        )]
    }


# ── Node 2: Policy Interpretation Agent ──────────────────────────────────────
def policy_interpretation_node(state: ClaimState) -> dict:
    """
    Core RAG + LLM policy interpretation.
    Receives fraud signals from the Fraud Gate and applies
    iProtect Smart clause routing (death type, suicide window, WoP-PD, etc.)
    """
    print("-> [Agent 1] Policy Interpretation Agent reading policy...")

    fraud  = _extract_fraud_fields(state)
    claim  = _extract_claim_fields(state)
    suicide_window = state.get("suicide_window_flag", False)
    calc_multiplier = 10 if claim["age_at_entry"] < 45 else 7
    calc_gross = max(claim["sum_assured"], claim["annualised_premium"] * calc_multiplier)
    claim["system_calculated_gross"] = calc_gross
    # Build a rich, context-aware retrieval query
    query_parts = [state.get("conflict_reason", "")]
    query_parts.append(f"death benefit clause {claim['death_type']}")
    if claim["death_type"] == "SUICIDE":
        query_parts.append("suicide clause 12 months premium refund")
    if fraud["non_disclosure_detected"]:
        query_parts.append("non-disclosure material misstatement Section 45 repudiation")
    if claim["incident_type"] == "ACCIDENTAL":
        query_parts.append("waiver of premium permanent disability accident exclusions")
    retrieval_query = " | ".join(q for q in query_parts if q)

    clauses = retrieve_policy_clauses(retrieval_query)
    combined_clauses = "\n\n---\n\n".join(clauses) if clauses else "No relevant clauses retrieved."

    prompt = PromptTemplate.from_template(POLICY_PROMPT)
    formatted_prompt = prompt.format(
        claim_id=claim["claim_id"],
        claimant_name=claim["claimant_name"],
        life_assured_name=claim["life_assured_name"],
        incident_type=claim["incident_type"],
        incident_date=claim["incident_date"],
        cause_of_death=claim["cause_of_death"],
        death_type=claim["death_type"],
        months_since_inception=claim["months_since_inception"],
        is_policy_in_force=claim["is_policy_in_force"],
        premium_payment_type=claim["premium_payment_type"],
        age_at_entry=claim["age_at_entry"],
        sum_assured=claim["sum_assured"],
        annualised_premium=claim["annualised_premium"],
        system_calculated_gross=claim["system_calculated_gross"],
        payout_option=claim["payout_option"],
        fraud_risk_level=fraud["fraud_risk_level"],
        fraud_risk_score=fraud["fraud_risk_score"],
        non_disclosure_detected=fraud["non_disclosure_detected"],
        non_disclosure_detail=fraud["non_disclosure_detail"],
        hospital_verified=fraud["hospital_verified"],
        suicide_window_flag=suicide_window,
        early_claim_flag=fraud["early_claim_flag"],
        conflict_reason=state.get("conflict_reason", "No specific conflict reported."),
        retrieved_clauses=combined_clauses,
    )

    structured_llm = llm.with_structured_output(PolicyOutput)
    response: PolicyOutput = structured_llm.invoke(formatted_prompt)

    # Persist structured policy output into state
    policy_result = {
        "is_covered":            response.is_covered,
        "coverage_verdict":      response.coverage_verdict,
        "gross_covered_amount":  response.gross_covered_amount,
        "deductible_applied":    response.deductible_applied,
        "net_coverage_amount":   response.net_coverage_amount,
        "cited_policy_section":  response.cited_policy_section,
        "reasoning":             response.reasoning,
        "exclusions_triggered":  response.exclusions_triggered,
        "uncertainty_flags":     response.uncertainty_flags,
        "confidence_score":      response.confidence_score,
    }

    agent_reasoning = (
        f"[POLICY AGENT]\n"
        f"  Coverage Verdict   : {response.coverage_verdict}\n"
        f"  Gross Amount       : ₹{response.gross_covered_amount:,.2f}\n"
        f"  Deductible         : ₹{response.deductible_applied:,.2f}\n"
        f"  Net Amount         : ₹{response.net_coverage_amount:,.2f}\n"
        f"  Clause Cited       : {response.cited_policy_section}\n"
        f"  Exclusions         : {response.exclusions_triggered}\n"
        f"  Uncertainty Flags  : {response.uncertainty_flags}\n"
        f"  Agent Confidence   : {response.confidence_score}\n"
        f"  Reasoning          :\n    {response.reasoning}\n"
    )

    return {
        "retrieved_policy_clauses": clauses,
        "policy_result":            policy_result,
        "agent_reasoning":          agent_reasoning,
        "audit_log": [audit_entry(
            "Policy Interpretation",
            f"RAG retrieved {len(clauses)} clause(s). "
            f"Verdict: {response.coverage_verdict}. "
            f"Net amount: ₹{response.net_coverage_amount:,.2f}. "
            f"Confidence: {response.confidence_score}.",
            confidence=response.confidence_score,
            flags=response.uncertainty_flags
        )]
    }


# ── Node 3: Claim Summary & Review Agent ─────────────────────────────────────
def claim_summary_and_review_node(state: ClaimState) -> dict:
    """
    Synthesises all upstream signals into the final ClaimDecisionReport.
    Applies mandatory decision rules before calling the LLM so that
    rule-based decisions (suicide clause, critical fraud) are never
    overridden by LLM creativity.
    """
    print("-> [Agent 2] Review Agent generating ClaimDecisionReport...")

    fraud   = _extract_fraud_fields(state)
    claim   = _extract_claim_fields(state)
    pr      = state.get("policy_result", {})
    suicide_window = state.get("suicide_window_flag", False)

    # ── Pre-LLM deterministic overrides ──────────────────────────────────────
    # These rules mirror the policy's non-negotiable clauses.
    forced_decision = None
    forced_reason   = ""

    # Rule: Suicide within 12 months → NOT full benefit, only 80% premiums
    if suicide_window and claim["death_type"] == "SUICIDE":
        forced_decision = "REJECTED"
        forced_reason   = (
            "Per Part F Clause 11 (ICICI Pru iProtect Smart): death by suicide within "
            "12 months of policy inception. Full death benefit is NOT payable. "
            "Only 80% of premiums paid will be refunded. Automated approval is blocked."
        )

    # Rule: Non-disclosure within contestability window → must escalate
    if fraud["non_disclosure_detected"] and claim["months_since_inception"] <= 36:
        forced_decision = "ESCALATE_HUMAN"
        forced_reason   = (
            "Material non-disclosure detected (Annexure IV, Section 45). "
            "Policy is within the 3-year contestability window. "
            f"Detail: {fraud['non_disclosure_detail']}. "
            "Senior investigator review required before any decision."
        )

    # Rule: Critical fraud score
    if fraud["fraud_risk_score"] >= 0.85:
        forced_decision = "ESCALATE_HUMAN"
        forced_reason   = (
            f"Critical fraud risk score ({fraud['fraud_risk_score']}) detected by Stage 3. "
            "Automated decision is blocked. Senior fraud investigator must review."
        )

    # ── Format inputs for LLM ─────────────────────────────────────────────────
    docs_reviewed_str = "; ".join(
        [f"{d}" for d in claim.get("documents_verified", [])]
    ) or "Not specified"
    doc_flags_str     = "; ".join(claim.get("document_flags", [])) or "None"
    fraud_reasons_str = (
        "; ".join(fraud["fraud_reasoning_summary"])
        if isinstance(fraud["fraud_reasoning_summary"], list)
        else str(fraud["fraud_reasoning_summary"])
    )
    policy_flags_str  = "; ".join(pr.get("uncertainty_flags", [])) or "None"
    excl_str          = "; ".join(pr.get("exclusions_triggered", [])) or "None"
    clauses_str       = pr.get("cited_policy_section", "None cited")

    prompt = PromptTemplate.from_template(REVIEW_PROMPT)
    formatted_prompt = prompt.format(
        claim_id=claim["claim_id"],
        policy_id=claim["policy_id"],
        claimant_name=claim["claimant_name"],
        life_assured_name=claim["life_assured_name"],
        incident_type=claim["incident_type"],
        incident_date=claim["incident_date"],
        documents_reviewed=docs_reviewed_str,
        document_flags=doc_flags_str,
        fraud_risk_level=fraud["fraud_risk_level"],
        fraud_risk_score=fraud["fraud_risk_score"],
        fraud_reasoning_summary=fraud_reasons_str,
        non_disclosure_detected=fraud["non_disclosure_detected"],
        non_disclosure_detail=fraud["non_disclosure_detail"],
        hospital_verified=fraud["hospital_verified"],
        early_claim_flag=fraud["early_claim_flag"],
        suicide_window_flag=suicide_window,
        stage3_confidence=fraud["stage3_confidence"],
        policy_coverage_verdict=pr.get("coverage_verdict", "UNKNOWN"),
        gross_covered_amount=pr.get("gross_covered_amount", 0.0),
        deductible_applied=pr.get("deductible_applied", 0.0),
        net_coverage_amount=pr.get("net_coverage_amount", 0.0),
        policy_clauses_cited=clauses_str,
        exclusions_applied=excl_str,
        policy_uncertainty_flags=policy_flags_str,
        policy_confidence=pr.get("confidence_score", 0.0),
    )

    structured_llm = llm.with_structured_output(ClaimDecisionReport)
    report: ClaimDecisionReport = structured_llm.invoke(formatted_prompt)
    
    report.claim_id             = claim["claim_id"]
    report.policy_id            = claim["policy_id"]
    report.claimant_name        = claim["claimant_name"]
    report.life_assured_name    = claim["life_assured_name"]
    report.incident_type        = claim["incident_type"]
    report.incident_date        = claim["incident_date"]
    report.documents_reviewed   = claim["documents_verified"]
    report.document_flags       = claim["document_flags"]
    report.fraud_risk_level     = fraud["fraud_risk_level"]
    report.fraud_risk_score     = fraud["fraud_risk_score"]
    report.fraud_reasoning_summary = fraud["fraud_reasoning_summary"] if isinstance(fraud["fraud_reasoning_summary"], str) else "; ".join(fraud["fraud_reasoning_summary"])
    report.non_disclosure_detected = fraud["non_disclosure_detected"]
    report.non_disclosure_detail   = fraud["non_disclosure_detail"]
    report.hospital_verified    = fraud["hospital_verified"]
    report.early_claim_flag     = fraud["early_claim_flag"]
    report.suicide_window_flag  = suicide_window

    report.gross_covered_amount = pr.get("gross_covered_amount", 0.0)
    report.deductible_applied   = pr.get("deductible_applied", 0.0)
    report.net_coverage_amount  = pr.get("net_coverage_amount", 0.0)
    report.policy_coverage_verdict = pr.get("coverage_verdict", report.policy_coverage_verdict)
    report.policy_clauses_cited = [pr.get("cited_policy_section", "")] if pr.get("cited_policy_section") else []
    report.exclusions_applied   = pr.get("exclusions_triggered", [])

    # ── Apply deterministic overrides AFTER LLM (safety net) ─────────────────
    if forced_decision:
        report.final_decision    = forced_decision
        report.escalation_reason = forced_reason
        report.decision_reasoning = forced_reason + "\n\n" + report.decision_reasoning

    # ── Persist decision and full report into state ───────────────────────────
    return {
        "claim_decision_report": report.model_dump(),
        "final_status":          report.final_decision,
        "confidence_score":      report.confidence_score,
        "detailed_summary":      report.decision_reasoning,
        "agent_reasoning":       state.get("agent_reasoning", "") + f"\n\n[REVIEW AGENT]\n  Final Decision: {report.final_decision}\n  Confidence: {report.confidence_score}\n  Reasoning: {report.decision_reasoning[:300]}...",
        "audit_log": [audit_entry(
            "Claim Review",
            f"ClaimDecisionReport generated. "
            f"Decision: {report.final_decision}. "
            f"Confidence: {report.confidence_score:.2f}. "
            f"Net coverage: ₹{report.net_coverage_amount:,.2f}.",
            confidence=report.confidence_score,
            flags=report.uncertainty_flags
        )]
    }


# ── Node 4a: Approved ─────────────────────────────────────────────────────────
def approved_node(state: ClaimState) -> dict:
    print("-> [Decision] APPROVED — full coverage confirmed.")
    return {
        "final_status": "APPROVED",
        "audit_log": [audit_entry(
            "Decision",
            "Automated approval criteria fully met. All fraud, policy, and confidence "
            "thresholds passed. Claim decision report finalised.",
            confidence=state.get("confidence_score", 1.0)
        )]
    }


# ── Node 4b: Partially Approved ───────────────────────────────────────────────
def partial_approved_node(state: ClaimState) -> dict:
    print("-> [Decision] PARTIALLY APPROVED — partial coverage confirmed.")
    rpt = state.get("claim_decision_report", {})
    net = rpt.get("net_coverage_amount", 0.0)
    return {
        "final_status": "PARTIALLY_APPROVED",
        "audit_log": [audit_entry(
            "Decision",
            f"Partial approval granted. Net covered amount: ₹{net:,.2f}. "
            "Remaining portion requires further review.",
            confidence=state.get("confidence_score")
        )]
    }


# ── Node 4c: Escalate to Human ────────────────────────────────────────────────
def escalate_human_node(state: ClaimState) -> dict:
    print("-> [Decision] ESCALATE_HUMAN — context bundle packaged for reviewer.")
    rpt = state.get("claim_decision_report", {})
    reason = rpt.get("escalation_reason", "See uncertainty flags and fraud findings.")

    # Build a structured handoff bundle
    context_bundle = {
        "escalation_reason":    reason,
        "claim_id":             state.get("claim_id"),
        "fraud_risk_score":     _extract_fraud_fields(state)["fraud_risk_score"],
        "non_disclosure":       _extract_fraud_fields(state)["non_disclosure_detected"],
        "policy_verdict":       state.get("policy_result", {}).get("coverage_verdict"),
        "uncertainty_flags":    rpt.get("uncertainty_flags", []),
        "agent_reasoning":      state.get("agent_reasoning", ""),
        "full_report_snapshot": rpt,
    }

    return {
        "final_status":   "ESCALATE_HUMAN",
        "context_bundle": context_bundle,
        "audit_log": [audit_entry(
            "Decision",
            f"Escalated to human reviewer. Reason: {reason[:200]}",
            confidence=state.get("confidence_score"),
            flags=rpt.get("uncertainty_flags", [])
        )]
    }


# ── Node 4d: Rejected ─────────────────────────────────────────────────────────
def rejected_node(state: ClaimState) -> dict:
    print("-> [Decision] REJECTED — policy exclusion or fraud confirmed.")
    rpt = state.get("claim_decision_report", {})
    return {
        "final_status": "REJECTED",
        "audit_log": [audit_entry(
            "Decision",
            f"Claim rejected. Reason: {rpt.get('decision_reasoning', '')[:300]}",
            confidence=state.get("confidence_score", 0.0)
        )]
    }


# =============================================================================
# SECTION 6 — CONDITIONAL ROUTER
# =============================================================================

def review_router(state: ClaimState) -> Literal["approved", "partial_approved", "escalate_human", "rejected"]:
    """
    Routes based on final_status set by the Review Agent.
    Deterministic — the LLM's decision is already baked into final_status.
    """
    status     = state.get("final_status", "ESCALATE_HUMAN").upper()
    confidence = state.get("confidence_score", 0.0)

    print(f"   [Router] Status: {status}  |  Confidence: {confidence:.2f}")

    if status == "APPROVED" and confidence >= 0.80:
        return "approved"
    elif status == "PARTIALLY_APPROVED" and confidence >= 0.50:
        return "partial_approved"
    elif status == "REJECTED":
        return "rejected"
    else:
        # Default: any ambiguity → human escalation
        return "escalate_human"


# =============================================================================
# SECTION 7 — GRAPH COMPILATION
# =============================================================================

workflow = StateGraph(ClaimState)

workflow.add_node("fraud_gate",               fraud_gate_node)
workflow.add_node("policy_interpretation",    policy_interpretation_node)
workflow.add_node("claim_summary_and_review", claim_summary_and_review_node)
workflow.add_node("approved",                 approved_node)
workflow.add_node("partial_approved",         partial_approved_node)
workflow.add_node("escalate_human",           escalate_human_node)
workflow.add_node("rejected",                 rejected_node)

workflow.set_entry_point("fraud_gate")
workflow.add_edge("fraud_gate",            "policy_interpretation")
workflow.add_edge("policy_interpretation", "claim_summary_and_review")

workflow.add_conditional_edges("claim_summary_and_review", review_router)

workflow.add_edge("approved",       END)
workflow.add_edge("partial_approved", END)
workflow.add_edge("escalate_human", END)
workflow.add_edge("rejected",       END)

app = workflow.compile()


# =============================================================================
# SECTION 8 — REPORT RENDERER
# =============================================================================

def _fmt_currency(val) -> str:
    try:
        return f"₹{float(val):>14,.2f}"
    except (TypeError, ValueError):
        return str(val)

def _fmt_list(items, indent=4) -> str:
    pad = " " * indent
    if not items:
        return f"{pad}(none)"
    return "\n".join(f"{pad}• {i}" for i in items)

def render_final_report(final_state: dict):
    rpt = final_state.get("claim_decision_report", {})
    sep = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print("  ICICI PRU iPROTECT SMART — CLAIM DECISION REPORT")
    print(f"  Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(sep)

    # ── 1. Identity ──────────────────────────────────────────────────────────
    print("\n  1. CLAIM IDENTITY")
    print(thin)
    print(f"  Claim ID          : {rpt.get('claim_id', final_state.get('claim_id', 'N/A'))}")
    print(f"  Policy ID         : {rpt.get('policy_id', final_state.get('policy_id', 'N/A'))}")
    print(f"  Claimant          : {rpt.get('claimant_name', 'N/A')}")
    print(f"  Life Assured      : {rpt.get('life_assured_name', 'N/A')}")
    print(f"  Incident Type     : {rpt.get('incident_type', 'N/A')}")
    print(f"  Incident Date     : {rpt.get('incident_date', 'N/A')}")

    # ── 2. Documents Reviewed ────────────────────────────────────────────────
    print("\n  2. DOCUMENTS REVIEWED")
    print(thin)
    docs = rpt.get("documents_reviewed", [])
    print(_fmt_list(docs if isinstance(docs, list) else [str(docs)]))
    flags = rpt.get("document_flags", [])
    if flags:
        print("  Document Flags:")
        print(_fmt_list(flags))

    # ── 3. Fraud & Risk Assessment ───────────────────────────────────────────
    print("\n  3. FRAUD & RISK ASSESSMENT")
    print(thin)
    print(f"  Fraud Risk Level  : {rpt.get('fraud_risk_level', 'N/A')}")
    print(f"  Fraud Risk Score  : {rpt.get('fraud_risk_score', 'N/A')}")
    print(f"  Hospital Verified : {rpt.get('hospital_verified', 'N/A')}")
    print(f"  Non-Disclosure    : {rpt.get('non_disclosure_detected', 'N/A')}")
    if rpt.get("non_disclosure_detail"):
        print(f"  ND Detail         : {rpt['non_disclosure_detail']}")
    print(f"  Early Claim Flag  : {rpt.get('early_claim_flag', 'N/A')}")
    print(f"  Suicide Window    : {rpt.get('suicide_window_flag', 'N/A')}")
    print(f"  Fraud Reasoning   :\n    {rpt.get('fraud_reasoning_summary', 'N/A')}")

    # ── 4. Policy Coverage Analysis ──────────────────────────────────────────
    print("\n  4. POLICY COVERAGE ANALYSIS (iProtect Smart)")
    print(thin)
    print(f"  Coverage Verdict  : {rpt.get('policy_coverage_verdict', 'N/A')}")
    print(f"  Gross Amount      : {_fmt_currency(rpt.get('gross_covered_amount', 0))}")
    print(f"  Deductible        : {_fmt_currency(rpt.get('deductible_applied', 0))}")
    print(f"  Net Payable       : {_fmt_currency(rpt.get('net_coverage_amount', 0))}")
    print("  Policy Clauses Cited:")
    clauses = rpt.get("policy_clauses_cited", [])
    print(_fmt_list(clauses if isinstance(clauses, list) else [str(clauses)]))
    excl = rpt.get("exclusions_applied", [])
    if excl:
        print("  Exclusions Applied:")
        print(_fmt_list(excl))

    # ── 5. Areas of Uncertainty ──────────────────────────────────────────────
    unc = rpt.get("uncertainty_flags", [])
    if unc:
        print("\n  5. AREAS OF UNCERTAINTY")
        print(thin)
        print(_fmt_list(unc))

    # ── 6. Final Decision ────────────────────────────────────────────────────
    decision = rpt.get("final_decision", final_state.get("final_status", "UNKNOWN"))
    confidence = rpt.get("confidence_score", final_state.get("confidence_score", 0.0))

    VERDICT_LABELS = {
        "APPROVED":          "✔  APPROVED",
        "PARTIALLY_APPROVED":"◑  PARTIALLY APPROVED",
        "ESCALATE_HUMAN":    "⚠  ESCALATE TO HUMAN REVIEWER",
        "REJECTED":          "✘  REJECTED",
    }
    verdict_label = VERDICT_LABELS.get(decision.upper(), decision)

    print(f"\n{'=' * 70}")
    print(f"  FINAL DECISION   : {verdict_label}")
    print(f"  CONFIDENCE SCORE : {confidence:.2f} / 1.00")
    print(f"{'=' * 70}")

    if rpt.get("escalation_reason"):
        print(f"\n  ESCALATION REASON:\n    {rpt['escalation_reason']}")

    print("\n  DECISION REASONING:")
    print(thin)
    reasoning = rpt.get("decision_reasoning", final_state.get("detailed_summary", "No reasoning generated."))
    for line in reasoning.split("\n"):
        print(f"    {line}")

    # ── 7. Agent Reasoning Chain ─────────────────────────────────────────────
    print("\n  7. AGENT REASONING CHAIN")
    print(thin)
    for line in final_state.get("agent_reasoning", "").split("\n"):
        print(f"    {line}")

    # ── 8. Compliance Audit Log ───────────────────────────────────────────────
    print("\n  8. COMPLIANCE AUDIT TRAIL")
    print(thin)
    for entry in final_state.get("audit_log", []):
        ts   = entry.get("timestamp", "")
        step = entry.get("step", "")
        det  = entry.get("details", "")
        conf = f"  [conf: {entry['confidence']:.2f}]" if "confidence" in entry else ""
        print(f"  [{ts}] [{step}]{conf}")
        print(f"    {det}")
        if entry.get("flags"):
            print(f"    Flags: {entry['flags']}")

    # ── Platform boundary notice ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("  PLATFORM BOUNDARY — No payment action is triggered by this report.")
    print("  This document is the terminal output of the AI Claims Decision Engine.")
    print("  All downstream financial actions require authorised human sign-off.")
    print(f"{sep}\n")


# =============================================================================
# SECTION 9 — ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PHASE 3 — POLICY REASONING & AUTONOMOUS DECISION LAYER")
    print("  ICICI Pru iProtect Smart | Claim Processing Pipeline")
    print("=" * 70)

    initial_state = get_initial_state()
    final_state   = app.invoke(initial_state)

    render_final_report(final_state)
