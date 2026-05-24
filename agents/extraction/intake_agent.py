"""
agents/extraction/intake_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Entry point for all claim submissions. The first agent to run on every
    new claim. Implements a LangGraph node using the ReAct strategy.

WHAT GOES HERE:

    CHANNEL ADAPTERS (three, one per intake method):

    web_form_adapter(raw_json: dict) → UnifiedPayload
        Parses a structured JSON form submission from the claimant web portal.
        Maps web form field names to the internal canonical field names.

    email_parser_adapter(email_body: str, attachments: list) → UnifiedPayload
        Extracts structured data from email body text and attachment list.
        Identifies which fields came from body text vs. form attachments.
        On ambiguous input (form data in body + attachments): reasons through
        which source is authoritative per field (ReAct reasoning step).

    document_upload_adapter(files: list[UploadFile]) → UnifiedPayload
        Handles multipart file upload. Saves documents to S3 staging bucket.
        Returns file references (S3 keys), not file content.

    DEDUP CHECKER:
        Fuzzy match on (policy_id + incident_date + claim_type) within 30-day window.
        SHA-256 content hash on each uploaded document file.
        If EXACT duplicate found → return DUPLICATE_REJECTED + original claim_id.
        If SOFT duplicate (same policy, similar incident) → create claim
        but set soft_duplicate_flag = True for Fraud Intelligence Agent.

    MAIN FLOW (LangGraph node function):
        1. Determine intake channel, select correct adapter
        2. Normalise to UnifiedPayload
        3. Run DedupChecker
        4. Create ClaimContextObject via claim_repository.create_claim()
        5. Trigger pre_intake_router to set cause_of_death_type +
           contestability_status routing flags
        6. Start IRDAI 24-hour acknowledgement SLA timer
        7. Publish CLAIM_REGISTERED to Kafka
        8. Queue acknowledgement message to Communications Agent
        9. Return AgentOutput with confidence score + claim_id

    REACT STRATEGY:
        Agent reasons step by step (Thought → Action → Observation) before
        choosing adapter. On ambiguous input, explicitly reasons through
        which source is authoritative for each field before resolving.

DEPENDENCIES:
    langgraph, shared.db.claim_repository, shared.events.kafka_client,
    shared.schemas.claim_context, shared.schemas.agent_output,
    shared.audit.audit_service, agents.extraction.pre_intake_router
"""
