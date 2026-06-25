from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.main import app


def build_payload(claim_case_id: str) -> dict:
    return {
        "claim_case": {
            "claim_case_id": claim_case_id,
            "policy_number": f"P-{claim_case_id}",
            "claimant": {"name": "Test Claimant"},
            "life_assured": {"name": "Test Life Assured"},
            "death_information": {
                "date_of_death": "2025-01-01",
                "cause_of_death": "natural",
            },
            "submitted_documents": [],
        }
    }


def main() -> None:
    client = TestClient(app)

    fraud_payload = build_payload("CLM-E2E-FRAUD")
    fraud_resp = client.post("/v1/claims/analyze/fraud", json=fraud_payload)
    print("fraud_status:", fraud_resp.status_code)
    if fraud_resp.status_code != 200:
        print("fraud_error:", fraud_resp.text[:1000])
        return

    full_payload = build_payload("CLM-E2E-FULL")
    full_resp = client.post("/v1/claims/analyze/full", json=full_payload)
    print("full_status:", full_resp.status_code)
    if full_resp.status_code != 200:
        print("full_error:", full_resp.text[:1500])
        return

    body = full_resp.json()
    print("full_ok:", True)
    print("policy_final_status:", body.get("policy_final_status"))
    report = body.get("policy_decision_report") or {}
    print("final_decision:", report.get("final_decision"))


if __name__ == "__main__":
    main()
