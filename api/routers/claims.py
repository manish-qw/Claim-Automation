"""
api/routers/claims.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    External (claimant-facing) API endpoints. No auth required — claimants
    access these directly. Rate limited and input-validated.

ENDPOINTS TO IMPLEMENT:

    POST /v1/claims
        Submit a new death claim.
        Request: multipart/form-data with:
            policy_id (str), claimant_name (str), date_of_death (date),
            cause_of_death (str), contact_email (str), contact_phone (str),
            documents (list of UploadFile)
        Response: {claim_id, status: "RECEIVED", acknowledgement_sla: datetime}
        Action: triggers intake_agent → starts the pipeline asynchronously.
        Validation: policy_id format, date format, file size/type limits.
        Rate limit: 5 submissions per IP per hour.

    GET /v1/claims/{claim_id}
        Get claim status for claimant self-service tracking.
        Response: {claim_id, current_stage, submitted_at, sla_deadline,
                   missing_documents: list, last_updated}
        Note: returns ONLY claimant-safe fields — no internal agent scores,
              no fraud flags, no confidence scores.

    POST /v1/claims/{claim_id}/documents
        Upload additional documents after initial submission.
        Request: multipart with document_type (str) + file (UploadFile).
        Action: triggers missing_doc_tracker to clear pending status,
                re-queues document through OCR pipeline.

    GET /v1/claims/{claim_id}/decision
        Get final decision when processing is complete.
        Response: only available when current_stage = SETTLED or REJECTED.
        Returns: {decision, decision_date, net_payout_amount (if approved),
                  plain_language_explanation, appeal_window_ends}
        Returns 404 if claim not found, 202 if still processing.

    POST /v1/claims/{claim_id}/appeal
        Submit an appeal against a denial or partial approval.
        Request: {appeal_reason (str, max 1000 words), additional_documents}
        Action: reactivates human review workflow with APPEAL flag.
        Constraint: only within 30-day appeal window from decision_date.

DEPENDENCIES:
    fastapi, shared.db.claim_repository,
    agents.extraction.intake_agent (async trigger),
    shared.schemas.claim_context
"""
