"""
agents/extraction/missing_doc_tracker.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Tracks mandatory document requirements for a claim, identifies gaps,
    and manages follow-up requests with SLA timers.

WHAT GOES HERE:

    MissingDocReport (dataclass):
        claim_id           — UUID
        mandatory_missing  — list[MissingDocument] — must have to proceed
        conditional_missing — list[MissingDocument] — required based on claim type
        all_mandatory_received — bool
        checked_at         — UTC timestamp

    MANDATORY MATRIX CHECK:
        Compare received document types against MANDATORY_DOCUMENTS_BY_CAUSE
        matrix (from shared/schemas/document_types.py) for this claim's
        cause_of_death_type.
        Produce two lists:
            mandatory_missing  — documents that MUST be received to proceed
            conditional_missing — documents required only for this specific
                                  claim scenario (e.g., FIR only for ACCIDENTAL)

    PER MISSING DOCUMENT — WHAT HAPPENS:

    1. Set document status to DOCUMENTS_PENDING in claim context.

    2. Construct a specific (not generic) follow-up request:
            "We require your [FIR copy / Postmortem Report / specific doc]
             to process your claim. Please upload via [portal link] or
             email to [address]. Reference: [claim_id]."
        DO NOT send generic "please submit documents" messages.
        Each message names the exact document and explains why it's needed.

    3. Publish follow-up request to Communications Agent queue.
       (Communications Agent sends the actual email/SMS — this module
       only constructs and queues the request.)

    4. Start a 15-day SLA timer for each mandatory missing document.
       If not received within 15 days: escalate and re-notify.

    PIPELINE BEHAVIOUR:
        Receipt of a missing doc → clear its pending status.
        Resume pipeline from document processing step ONLY for that document.
        Do NOT re-run agents that already completed successfully.

    ON DOCUMENT RECEIVED (after initial submission):
        Triggered when claimant uploads additional document via
        POST /v1/claims/{claim_id}/documents.
        Clears the pending flag for that document type.
        Re-queues just that document through the OCR → classify →
        extract → forensics sub-pipeline.
        If it was the last mandatory missing doc: resumes full pipeline.

DEPENDENCIES:
    shared.schemas.document_types (MANDATORY_DOCUMENTS_BY_CAUSE),
    shared.schemas.claim_context (MissingDocument),
    shared.events.kafka_client, shared.config.settings,
    shared.db.claim_repository
"""
