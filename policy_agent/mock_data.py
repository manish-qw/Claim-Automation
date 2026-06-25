"""
mock_data.py
------------
Full Phase 2 payload + get_initial_state() for the policy orchestrator.

BUG FIXED: Original get_initial_state() referenced phase2_data["claim_profile"]
which didn't exist in FULL_MOCK_PAYLOAD — crashed immediately on first run.
claim_profile block is now included in the payload and read correctly.
"""
import json

FULL_MOCK_PAYLOAD = {
    "claim_case_id": "CLM-001",
    "status": "success",
    "result": {
        "claim_case_id": "CLM-001",

        # ── NEW: claim_profile block — all identity fields live here ──────────
        # Previously missing; get_initial_state() crashed trying to read this.
        "claim_profile": {
            "policy_id":                 "POL-ICICI-998877",
            "claimant_name":             "Priya Sharma",
            "life_assured_name":         "Rajesh Sharma",
            "life_assured_age_at_entry": 42,
            "incident_type":             "NATURAL",
            "incident_date":             "2026-04-15",
            "cause_of_death":            "Cardiac Arrest",
            "death_type":                "NATURAL",
            "months_since_inception":    48,
            "is_policy_in_force":        True,
            "premium_payment_type":      "REGULAR",
            "sum_assured":               5000000.0,
            "annualised_premium":        105000.0,
            "payout_option_chosen":      "LUMP_SUM",
            "documents_verified": [
                "Death Certificate (OCR conf: 0.95)",
                "Medical Report (OCR conf: 0.88)",
                "Claimant ID Proof (OCR conf: 0.97)",
                "Policy Document (OCR conf: 0.99)"
            ],
            "document_flags": [
                "Cause-of-death mismatch: Death Certificate says 'Cardiac Arrest', "
                "Medical Report says 'Heart Disease'"
            ],
        },

        # ── Fraud pipeline agent outputs (unchanged from your Phase 2) ────────
        "fraud_analysis": {
            "confidence_score": 0.9744,
            "trust_score": 0.7308,
            "validation_flags": [],
            "audit_reference": "AUD-3B83DC880A5F",
            "created_at": "2026-05-25T17:35:24.152977+00:00",
            "fraud_risk_score": 0.2692,
            "fraud_risk_level": "LOW",
            "fraud_reasons": [],
            "anomaly_score": 0.8975,
            "suspicious_nominee": False,
            "recommended_action": "APPROVE"
        },
        "trust_analysis": {
            "confidence_score": 0.8258,
            "trust_score": 0.8258,
            "validation_flags": [],
            "audit_reference": "AUD-68D12208B315",
            "created_at": "2026-05-25T17:35:56.061839+00:00",
            "ocr_trust_score": 0.95,
            "external_trust_score": 1,
            "fraud_confidence_score": 0.9744,
            "overall_trust_score": 0.8258,
            "human_review_required": False,
            "trust_breakdown": {},
            "automation_decision": "HOLD",
            "automation_eligible": False,
            "trust_reduction_reasons": ["Unresolved document conflict — trust reduced"]
        },
        "external_verification": {
            "confidence_score": 1,
            "trust_score": 1,
            "validation_flags": [],
            "audit_reference": "AUD-5C07B978E2BD",
            "created_at": "2026-05-25T17:35:24.143033+00:00",
            "hospital_verified": True,
            "doctor_verified": True,
            "fir_verified": True,
            "metadata_tampered": False,
            "image_manipulation_detected": False,
            "geo_location_match": True,
            "verification_confidence": 1,
            "hospital_verification_detail": None,
            "fir_verification_detail": None,
            "metadata_issues": []
        },
        "early_claim_analysis": {
            "confidence_score": 0.4,
            "trust_score": 0.85,
            "validation_flags": [],
            "audit_reference": "AUD-9FAA6C2459B1",
            "created_at": "2026-05-25T17:35:24.152977+00:00",
            "policy_age_days": 1800,
            "policy_revival_detected": False,
            "days_since_revival": None,
            "premium_irregularities": False,
            "lapsed_periods": [],
            "early_claim_risk": "LOW",
            "risk_factors": []
        },
        "non_disclosure_analysis": {
            "confidence_score": 0.3,
            "trust_score": 0.9,
            "validation_flags": [],
            "audit_reference": "AUD-0D750C2420D5",
            "created_at": "2026-05-25T17:35:56.060806+00:00",
            "contradiction_detected": False,
            "non_disclosure_findings": [],
            "non_disclosure_score": 0,
            "proposal_disclosures": {},
            "medical_findings": {},
            "proposal_claims": [],
            "medical_facts": ["heart disease"]
        },
        "conflict_resolution": {
            "confidence_score": 0.6,
            "trust_score": 0.45,
            "validation_flags": ["CONFLICT_DETECTED"],
            "audit_reference": "AUD-E742A85321DF",
            "created_at": "2026-05-25T17:35:56.061839+00:00",
            "conflict_detected": True,
            "conflicts": [],
            "conflicts_found": [
                "cause_of_death: death_certificate says 'Cardiac Arrest' vs "
                "medical_report says 'Heart Disease'"
            ],
            "resolved_action": "ESCALATE",
            "winning_source": "ESCALATED",
            "resolution_rationale": None,
            "resolution_reasoning": "Tie detected — human review required"
        },
        "graph_analysis": {
            "confidence_score": 0.6,
            "trust_score": 1,
            "validation_flags": [],
            "audit_reference": "AUD-3155A5C308A6",
            "created_at": "2026-05-25T17:35:56.065357+00:00",
            "connected_claims": 0,
            "shared_nominees": 0,
            "shared_bank_accounts": 0,
            "shared_hospitals": 0,
            "high_risk_relationships": [],
            "fraud_ring_detected": False,
            "network_risk_score": 0,
            "graph_risk_score": 0
        },
        "final_recommendation": "ESCALATE",
        "escalation_required": True,
        "pipeline_duration_ms": None,
        "created_at": "2026-05-25T17:35:56.065357+00:00"
    }
}


def get_initial_state(fraud_payload: dict = None) -> dict:
    """
    Build the LangGraph initial state from a fraud pipeline payload.

    Args:
        fraud_payload: The JSON dict from the fraud pipeline (Phase 2 output).
                       Defaults to FULL_MOCK_PAYLOAD if not provided.

    Returns:
        A ClaimState-compatible dict ready to invoke the LangGraph app.
    """
    payload = fraud_payload or FULL_MOCK_PAYLOAD

    # Unwrap "result" wrapper if present
    phase2_data = payload.get("result", payload)

    # claim_profile holds all identity fields
    profile = phase2_data.get("claim_profile", {})

    # Extract conflict reason safely
    conflicts = phase2_data.get("conflict_resolution", {}).get("conflicts_found", [])
    conflict_reason = conflicts[0] if conflicts else "No conflict detected"

    return {
        # ── Identifiers ───────────────────────────────────────────────────────
        "claim_id":                  phase2_data.get("claim_case_id", "CLM-UNKNOWN"),
        "extracted_data":            phase2_data,     # full blob available for any node

        # ── Fraud sub-agent outputs (flat — orchestrator reads these directly) ─
        "fraud_analysis":            phase2_data.get("fraud_analysis", {}),
        "non_disclosure_analysis":   phase2_data.get("non_disclosure_analysis", {}),
        "external_verification":     phase2_data.get("external_verification", {}),
        "early_claim_analysis":      phase2_data.get("early_claim_analysis", {}),
        "conflict_resolution":       phase2_data.get("conflict_resolution", {}),

        # ── Conflict reason string for policy agent prompt ─────────────────────
        "conflict_reason":           conflict_reason,

        # ── Identity fields (top-level so _extract_claim_fields() finds them) ──
        "policy_id":                 profile.get("policy_id", ""),
        "claimant_name":             profile.get("claimant_name", ""),
        "life_assured_name":         profile.get("life_assured_name", ""),
        "life_assured_age_at_entry": profile.get("life_assured_age_at_entry", 0),
        "incident_type":             profile.get("incident_type", "NATURAL"),
        "incident_date":             profile.get("incident_date", ""),
        "cause_of_death":            profile.get("cause_of_death", ""),
        "death_type":                profile.get("death_type", "NATURAL"),
        "months_since_inception":    profile.get("months_since_inception", 0),
        "is_policy_in_force":        profile.get("is_policy_in_force", True),
        "premium_payment_type":      profile.get("premium_payment_type", "REGULAR"),
        "sum_assured":               profile.get("sum_assured", 0.0),
        "annualised_premium":        profile.get("annualised_premium", 0.0),
        "payout_option_chosen":      profile.get("payout_option_chosen", "LUMP_SUM"),
        "documents_verified":        profile.get("documents_verified", []),
        "document_flags":            profile.get("document_flags", []),

        # ── Phase 3 working variables — always start fresh ─────────────────────
        "retrieved_policy_clauses":  [],
        "agent_reasoning":           "",
        "confidence_score":          0.0,
        "final_status":              "PENDING",
        "detailed_summary":          "",
        "suicide_window_flag":       False,
        "fraud_gate_flags":          [],
        "policy_result":             {},
        "claim_decision_report":     {},
        "context_bundle":            {},
        "audit_log": [{
            "step":    "Initialization",
            "details": f"Phase 2 payload loaded. Claim: {phase2_data.get('claim_case_id')}. "
                       f"Pipeline verdict: {phase2_data.get('final_recommendation', 'UNKNOWN')}."
        }],
    }


# ── Save test case for inspection ──────────────────────────────────────────────
if __name__ == "__main__":
    with open("test_case_1_full.json", "w") as f:
        json.dump(FULL_MOCK_PAYLOAD, f, indent=4)
    print("Mock payload saved to test_case_1_full.json")

    state = get_initial_state()
    print(f"Initial state built. Claim ID: {state['claim_id']}")
    print(f"Claimant: {state['claimant_name']}  |  Policy: {state['policy_id']}")
    print(f"Fraud risk: {state['fraud_analysis'].get('fraud_risk_level')}  "
          f"|  Conflict: {state['conflict_reason'][:60]}...")
