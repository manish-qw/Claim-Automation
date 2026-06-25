"""
fraud_bridge.py
---------------
Connects the OCR extraction pipeline (api/) to:
  1. The 7-agent Fraud Intelligence Pipeline  (fraud_pipeline/)
  2. The LangGraph Policy Orchestrator        (policy_agent/)

Endpoints
---------
POST /v1/claims/analyze/fraud   — fraud analysis only
POST /v1/claims/analyze/full    — fraud + policy decision report
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

# ---------------------------------------------------------------------------
# fraud_pipeline is a top-level package at the project root — import directly.
# ---------------------------------------------------------------------------
_BRIDGE_IMPORT_ERROR: Exception | None = None
normalize_claim_case_payload = None
ClaimState = None
FraudPipeline = None

try:
    from fraud_pipeline.utils.ocr_payload_normalizer import (
        normalize_claim_case_payload as _norm,
    )
    from fraud_pipeline.schemas import ClaimState as _ClaimState
    from fraud_pipeline.pipeline import FraudPipeline as _FraudPipeline

    normalize_claim_case_payload = _norm
    ClaimState = _ClaimState
    FraudPipeline = _FraudPipeline
except Exception as exc:
    _BRIDGE_IMPORT_ERROR = exc

# ---------------------------------------------------------------------------
# policy_agent/orchestrator.py uses bare imports (from state import …).
# Adding the policy_agent directory to sys.path lets those bare imports
# resolve without modifying orchestrator.py at all.
# ---------------------------------------------------------------------------
_POLICY_AGENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "policy_agent")
)
if _POLICY_AGENT_DIR not in sys.path:
    sys.path.insert(0, _POLICY_AGENT_DIR)


# ---------------------------------------------------------------------------
router = APIRouter(prefix="/v1/claims/analyze", tags=["Fraud Analysis"])
_PIPELINE: Any = None


def _ensure_bridge_ready() -> None:
    if _BRIDGE_IMPORT_ERROR is not None:
        raise HTTPException(
            status_code=500,
            detail=f"Fraud bridge unavailable: {_BRIDGE_IMPORT_ERROR}",
        )


def _get_pipeline() -> Any:
    global _PIPELINE
    _ensure_bridge_ready()
    if _PIPELINE is None:
        _PIPELINE = FraudPipeline()
    return _PIPELINE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_claim_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap optional { claim_case: {...} } envelope and carry top-level policy fields."""
    if not isinstance(body, dict):
        return {}

    payload: Dict[str, Any] = dict(body.get("claim_case", body))

    # Documents key — carry if sent at wrapper level
    if "documents" in body and "documents" not in payload:
        payload["documents"] = body["documents"]

    # Policy/routing fields sent outside the claim_case envelope
    _POLICY_FIELDS = (
        "claim_case_id", "policy_number", "policy_issue_date", "policy_age_days",
        "policy_revival_detected", "policy_sum_assured", "policy_premium",
        "premium_payment_history", "incident_type", "death_type",
        "months_since_inception", "is_policy_in_force", "premium_payment_type",
        "payout_option_chosen", "proposal_smoking", "proposal_alcohol_use",
        "proposal_pre_existing_conditions",
    )
    for key in _POLICY_FIELDS:
        if key in body and key not in payload:
            payload[key] = body[key]

    return payload


def _build_policy_payload(
    state: Any, package: Any, raw_claim_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the payload expected by policy_agent/mock_data.get_initial_state()."""
    documents_verified: List[str] = []
    for d in state.submitted_documents:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        low_conf = len(meta.get("low_confidence_fields") or [])
        documents_verified.append(
            f"{d.doc_type} (ocr_conf: {d.ocr_confidence}, low_conf_fields: {low_conf})"
        )

    return {
        "claim_case_id": package.claim_case_id,
        "status": "success",
        "result": {
            "claim_case_id": package.claim_case_id,
            "claim_profile": {
                "policy_id":                  state.policy_number,
                "claimant_name":              state.claimant.name,
                "life_assured_name":          state.life_assured.name,
                "life_assured_age_at_entry":  state.life_assured.age,
                "incident_type":              raw_claim_data.get("incident_type", "NATURAL"),
                "incident_date":              state.death_information.date_of_death,
                "cause_of_death":             state.death_information.cause_of_death,
                "death_type":                 raw_claim_data.get("death_type", "NATURAL"),
                "months_since_inception":     raw_claim_data.get("months_since_inception", 0),
                "is_policy_in_force":         raw_claim_data.get("is_policy_in_force", True),
                "premium_payment_type":       raw_claim_data.get("premium_payment_type", "REGULAR"),
                "sum_assured":                state.policy_sum_assured,
                "annualised_premium":         state.policy_premium,
                "payout_option_chosen":       raw_claim_data.get("payout_option_chosen", "LUMP_SUM"),
                "documents_verified":         documents_verified,
                "document_flags":             state.validation_flags,
            },
            "fraud_analysis":        package.fraud_analysis or {},
            "trust_analysis":        package.trust_analysis or {},
            "external_verification": package.external_verification or {},
            "early_claim_analysis":  package.early_claim_analysis or {},
            "non_disclosure_analysis": package.non_disclosure_analysis or {},
            "conflict_resolution":   package.conflict_resolution or {},
            "graph_analysis":        package.graph_analysis or {},
            "final_recommendation":  package.final_recommendation,
            "escalation_required":   package.escalation_required,
        },
    }


def run_full_analysis_sync(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous integration entrypoint used by both FastAPI routes and the
    upload-batch auto trigger.
    """
    _ensure_bridge_ready()
    payload = _extract_claim_payload(request)
    if not payload:
        raise ValueError("Invalid payload: expected a JSON object.")

    try:
        normalized = normalize_claim_case_payload(payload)
        state = ClaimState(**normalized)
    except Exception as exc:
        raise ValueError(f"Payload normalisation failed: {exc}") from exc

    try:
        package = FraudPipeline().run(state)
        fraud_result = package.model_dump()
    except Exception as exc:
        raise RuntimeError(f"Fraud pipeline failed: {exc}") from exc

    try:
        from mock_data import get_initial_state      # type: ignore[import]
        from orchestrator import app as policy_app   # type: ignore[import]

        fraud_payload = _build_policy_payload(state, package, normalized)
        initial_state = get_initial_state(fraud_payload=fraud_payload)
        final_state = policy_app.invoke(initial_state)
    except Exception as exc:
        raise RuntimeError(f"Policy orchestrator failed: {exc}") from exc

    return {
        "fraud_result": fraud_result,
        "policy_decision_report": final_state.get("claim_decision_report"),
        "policy_final_status": final_state.get("final_status"),
        "detailed_summary": final_state.get("detailed_summary"),
        "audit_log": final_state.get("audit_log", []),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/fraud", summary="Run 7-agent fraud analysis on a claim")
async def run_fraud_analysis(request: Dict[str, Any]):
    """
    Accepts OCR-extracted document JSON (or a ClaimState-compatible dict).
    Returns the full FraudIntelligencePackage.
    """
    _ensure_bridge_ready()
    payload = _extract_claim_payload(request)
    if not payload:
        raise HTTPException(422, "Invalid payload — expected a JSON object.")

    try:
        normalized = normalize_claim_case_payload(payload)
        state = ClaimState(**normalized)
    except Exception as exc:
        raise HTTPException(422, f"Payload normalisation failed: {exc}")

    try:
        pipeline = _get_pipeline()
        package = await run_in_threadpool(pipeline.run, state)
        return package.model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Fraud pipeline failed: {exc}")


@router.post("/full", summary="Run fraud analysis + policy decision report")
async def run_full_pipeline(request: Dict[str, Any]):
    """
    Chains: OCR JSON → Fraud Pipeline (7 agents) → Policy Orchestrator (LangGraph).
    Returns fraud result + ClaimDecisionReport + final status.
    """
    _ensure_bridge_ready()
    payload = _extract_claim_payload(request)
    if not payload:
        raise HTTPException(422, "Invalid payload — expected a JSON object.")

    try:
        normalized = normalize_claim_case_payload(payload)
        state = ClaimState(**normalized)
    except Exception as exc:
        raise HTTPException(422, f"Payload normalisation failed: {exc}")

    # Stage 1: Fraud pipeline
    try:
        pipeline = _get_pipeline()
        package = await run_in_threadpool(pipeline.run, state)
        fraud_result = package.model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Fraud pipeline failed: {exc}")

    # Stage 2: Policy orchestrator
    # orchestrator.py uses bare imports (from state import …) — resolved via
    # _POLICY_AGENT_DIR on sys.path (set at module load above).
    try:
        from mock_data import get_initial_state      # type: ignore[import]
        from orchestrator import app as policy_app   # type: ignore[import]

        fraud_payload = _build_policy_payload(state, package, normalized)
        initial_state = get_initial_state(fraud_payload=fraud_payload)
        final_state = await run_in_threadpool(policy_app.invoke, initial_state)
    except Exception as exc:
        raise HTTPException(500, f"Policy orchestrator failed: {exc}")

    return {
        "fraud_result":           fraud_result,
        "policy_decision_report": final_state.get("claim_decision_report"),
        "policy_final_status":    final_state.get("final_status"),
        "detailed_summary":       final_state.get("detailed_summary"),
        "audit_log":              final_state.get("audit_log", []),
    }
