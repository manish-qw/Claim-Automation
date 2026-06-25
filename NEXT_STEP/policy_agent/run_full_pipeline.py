"""
Run fraud pipeline + policy orchestrator end-to-end.

Usage:
    python run_full_pipeline.py
    python run_full_pipeline.py --live test_case_1_full.json
"""

import json
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAUD_PROJECT_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "claimos-fraud-project"))

for _path in (FRAUD_PROJECT_DIR, THIS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Windows console safety (rupee symbol, unicode output).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def run_with_mock():
    from mock_data import get_initial_state
    from orchestrator import app, render_final_report

    initial_state = get_initial_state()
    final_state = app.invoke(initial_state)
    render_final_report(final_state)
    return final_state


def _build_policy_payload(raw_claim_data, state, package):
    documents_verified = [
        f"{d.doc_type} (doc_ocr_conf: {d.ocr_confidence})"
        for d in state.submitted_documents
    ]

    return {
        "claim_case_id": package.claim_case_id,
        "status": "success",
        "result": {
            "claim_case_id": package.claim_case_id,
            "claim_profile": {
                "policy_id": state.policy_number,
                "claimant_name": state.claimant.name,
                "life_assured_name": state.life_assured.name,
                "life_assured_age_at_entry": state.life_assured.age,
                "incident_type": raw_claim_data.get("incident_type", "NATURAL"),
                "incident_date": state.death_information.date_of_death,
                "cause_of_death": state.death_information.cause_of_death,
                "death_type": raw_claim_data.get("death_type", "NATURAL"),
                "months_since_inception": raw_claim_data.get("months_since_inception", 0),
                "is_policy_in_force": raw_claim_data.get("is_policy_in_force", True),
                "premium_payment_type": raw_claim_data.get("premium_payment_type", "REGULAR"),
                "sum_assured": state.policy_sum_assured,
                "annualised_premium": state.policy_premium,
                "payout_option_chosen": raw_claim_data.get("payout_option_chosen", "LUMP_SUM"),
                "documents_verified": documents_verified,
                "document_flags": state.validation_flags,
            },
            "fraud_analysis": package.fraud_analysis or {},
            "trust_analysis": package.trust_analysis or {},
            "external_verification": package.external_verification or {},
            "early_claim_analysis": package.early_claim_analysis or {},
            "non_disclosure_analysis": package.non_disclosure_analysis or {},
            "conflict_resolution": package.conflict_resolution or {},
            "graph_analysis": package.graph_analysis or {},
            "final_recommendation": package.final_recommendation,
            "escalation_required": package.escalation_required,
        },
    }


def run_live(claim_input_path: str):
    from fraud_pipeline.pipeline import FraudPipeline
    from fraud_pipeline.schemas.base import ClaimState as FraudClaimState
    from mock_data import get_initial_state
    from orchestrator import app, render_final_report

    with open(claim_input_path, encoding="utf-8") as f:
        raw_claim_data = json.load(f)

    fraud_state = FraudClaimState(**raw_claim_data)
    package = FraudPipeline().run(fraud_state)

    fraud_payload = _build_policy_payload(raw_claim_data, fraud_state, package)
    initial_state = get_initial_state(fraud_payload=fraud_payload)
    final_state = app.invoke(initial_state)
    render_final_report(final_state)
    return final_state


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        if len(sys.argv) < 3:
            print("Usage: python run_full_pipeline.py --live <claim_input.json>")
            sys.exit(1)
        run_live(sys.argv[2])
    else:
        print("Running with mock payload (use --live <claim.json> for real claims)\n")
        run_with_mock()
