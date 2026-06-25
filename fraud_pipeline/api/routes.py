"""
FastAPI routes for the CLAIMOS Fraud Intelligence Pipeline.
Endpoints:
  GET  /fraud/health          — liveness check
  GET  /fraud/results         — list all stored results from PostgreSQL
  GET  /fraud/results/{id}    — fetch one result by claim_case_id
  POST /fraud/analyze         — run full 7-agent pipeline + save to PostgreSQL
  POST /fraud/analyze/batch   — batch processing
"""
from __future__ import annotations
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List

from fraud_pipeline.schemas import ClaimState
from fraud_pipeline.pipeline import FraudPipeline
from fraud_pipeline.services.postgres_service import PostgresService
from fraud_pipeline.utils.ocr_payload_normalizer import normalize_claim_case_payload
from fraud_pipeline.utils.logger import get_logger
import sys
import os

POLICY_AGENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../policy_agent"))
if POLICY_AGENT_PATH not in sys.path:
    sys.path.insert(0, POLICY_AGENT_PATH)

logger = get_logger("API")

# Singletons — created once at startup
_pipeline = FraudPipeline()
_postgres  = PostgresService()

router = APIRouter(prefix="/fraud", tags=["Fraud Intelligence"])


# ── Request / Response Models ──────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    claim_case: dict


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    service: str = "claimos-fraud-pipeline"
    postgres: str = "unknown"


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        postgres="connected" if _postgres.is_available else "unavailable",
    )


# ── Results (read from PostgreSQL) ─────────────────────────────────────────────

@router.get("/results", summary="List all stored fraud results")
async def list_results():
    """Return summary of all claims stored in PostgreSQL (no full JSON blob)."""
    rows = _postgres.get_all_results_summary()
    return {"total": len(rows), "results": rows}


@router.get("/results/{claim_case_id}", summary="Get result for a specific claim")
async def get_result(claim_case_id: str):
    """Fetch the full stored result for a given claim_case_id."""
    row = _postgres.get_fraud_result(claim_case_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No result found for claim_case_id: {claim_case_id}",
        )
    import json
    # Parse the full_package JSON string back to dict
    if row.get("full_package"):
        try:
            row["full_package"] = json.loads(row["full_package"])
        except Exception:
            pass
    return row


# ── Analyze (run pipeline + save) ──────────────────────────────────────────────

@router.post("/analyze", summary="Run fraud analysis on a claim")
async def analyze_claim(request: AnalyzeRequest):
    """
    Run all 7 fraud intelligence agents on the submitted claim.
    Result is automatically saved to PostgreSQL.

    Body: { "claim_case": { ...ClaimState fields... } }
    """
    # 1. Validate input
    try:
        claim_case = normalize_claim_case_payload(request.claim_case)
        state = ClaimState(**claim_case)
    except Exception as e:
        logger.error("invalid_input", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid claim_case format: {e}",
        )

    # 2. Run pipeline
    try:
        package = _pipeline.run(state)
        result  = package.model_dump()
    except Exception as e:
        logger.error("pipeline_error", error=str(e), claim_id=state.claim_case_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {e}",
        )

    # 3. Persist to PostgreSQL
    saved = _postgres.save_fraud_result(result)
    result["_saved_to_db"] = saved   # tell the caller whether it was persisted

    return result


# ── Batch Analyze ──────────────────────────────────────────────────────────────

@router.post("/analyze/batch", summary="Run fraud analysis on multiple claims")
async def analyze_batch(requests: List[AnalyzeRequest]):
    """Batch endpoint — processes multiple claims in sequence."""
    results = []
    for req in requests:
        try:
            claim_case = normalize_claim_case_payload(req.claim_case)
            state   = ClaimState(**claim_case)
            package = _pipeline.run(state)
            result  = package.model_dump()
            _postgres.save_fraud_result(result)
            results.append({
                "claim_case_id": state.claim_case_id,
                "status":        "success",
                "result":        result,
            })
        except Exception as e:
            results.append({"status": "error", "error": str(e)})
    return {"total": len(results), "results": results}


# ── Analyze Full (Fraud + Policy Interpretation) ────────────────────────────────

@router.post("/analyze-full", summary="Run fraud analysis AND policy interpretation")
async def analyze_full(request: AnalyzeRequest):
    """
    Run all 7 fraud intelligence agents AND the policy orchestrator.
    """
    # 1. Validate input
    try:
        claim_case = normalize_claim_case_payload(request.claim_case)
        state = ClaimState(**claim_case)
    except Exception as e:
        logger.error("invalid_input", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid claim_case format: {e}",
        )

    # 2. Run fraud pipeline
    try:
        package = _pipeline.run(state)
        result = package.model_dump()
    except Exception as e:
        logger.error("pipeline_error", error=str(e), claim_id=state.claim_case_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {e}",
        )

    # 3. Persist to PostgreSQL
    saved = _postgres.save_fraud_result(result)
    result["_saved_to_db"] = saved

    # 4. Construct fraud_payload for policy orchestrator
    raw_claim_data = claim_case
    fraud_payload = {
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
                "documents_verified": [
                    f"{d.doc_type} (OCR conf: {d.ocr_confidence})"
                    for d in state.submitted_documents
                ],
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
        }
    }

    # 5. Run policy orchestrator
    try:
        from mock_data import get_initial_state  # type: ignore
        from orchestrator import app as policy_app  # type: ignore
        
        initial_state = get_initial_state(fraud_payload=fraud_payload)
        final_state = policy_app.invoke(initial_state)
        
        return {
            "fraud_result": result,
            "policy_decision_report": final_state.get("claim_decision_report"),
            "policy_final_status": final_state.get("final_status"),
            "detailed_summary": final_state.get("detailed_summary"),
        }
    except Exception as e:
        logger.error(f"policy_orchestrator_error: {str(e)} | claim_id: {state.claim_case_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Policy Orchestrator execution failed: {e}",
        )

