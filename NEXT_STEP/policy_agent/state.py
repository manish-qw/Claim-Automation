from typing import TypedDict, List, Dict, Any, Optional


class ClaimState(TypedDict, total=False):
    # ── Identifiers ───────────────────────────────────────────────────────────
    claim_id: str
    policy_id: str

    # ── Raw data blobs ────────────────────────────────────────────────────────
    claim_details: dict
    extracted_data: dict          # full Phase 2 result dict stored here

    # ── Fraud sub-agent outputs (top-level for easy access) ───────────────────
    fraud_analysis: dict          # Agent 2 output (flat: fraud_risk_score etc.)
    non_disclosure_analysis: dict # Agent 4 output
    external_verification: dict   # Agent 1 output
    early_claim_analysis: dict    # Agent 3 output
    conflict_resolution: dict     # Agent 5 output  ← NEW: was missing

    # ── Claim identity fields (top-level for orchestrator helpers) ────────────
    claimant_name: str
    life_assured_name: str
    life_assured_age_at_entry: int
    incident_type: str
    incident_date: str
    cause_of_death: str
    death_type: str
    months_since_inception: int
    is_policy_in_force: bool
    premium_payment_type: str
    sum_assured: float
    annualised_premium: float
    payout_option_chosen: str
    documents_verified: list
    document_flags: list

    # ── Pipeline working variables ────────────────────────────────────────────
    conflict_reason: str
    retrieved_policy_clauses: list
    agent_reasoning: str
    confidence_score: float
    final_status: str
    detailed_summary: str
    suicide_window_flag: bool
    fraud_gate_flags: list

    # ── Agent outputs ─────────────────────────────────────────────────────────
    policy_result: dict
    claim_decision_report: dict

    # ── Escalation handoff bundle (used by escalate_human_node) ──────────────
    context_bundle: dict          # ← was missing, caused KeyError in escalate node

    # ── Audit trail ───────────────────────────────────────────────────────────
    audit_log: list