"""
shared/audit/audit_service.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Sidecar audit logging service. Every agent calls audit_service.log(event).
    Never write to the audit store directly from an agent.

WHAT GOES HERE:

    FUNCTIONS:

    log(event: AuditEvent) → None
        Async, fire-and-forget audit logger.
        Steps performed internally:
            1. Compute SHA-256 hash of input payload → event.input_hash
            2. Compute SHA-256 hash of output payload → event.output_hash
            3. Fetch the previous AuditEvent hash for this claim_id
               → event.previous_entry_hash  (builds the hash chain)
            4. Write to AWS QLDB asynchronously (primary store)
            5. If QLDB write fails: fall back to DynamoDB Streams
            6. If both fail: push to a dead-letter SQS queue for retry
        Internal errors are swallowed — audit failures must never
        propagate back to the calling agent or block the pipeline.

    get_chain(claim_id: str) → list[AuditEvent]
        Fetches the full ordered audit trail for a claim from QLDB.
        Returns events in chronological order (oldest first).

    verify_chain_integrity(claim_id: str) → bool
        Validates the hash chain: for each AuditEvent, verifies that
        its previous_entry_hash matches the output_hash of the
        preceding entry. Returns True if chain is intact, False if
        any link is broken or missing.

    export_regulatory_report(claim_id: str) → bytes
        Generates an IRDAI-compliant PDF audit report for a claim.
        Includes: all agent actions, timestamps, decisions made,
        flags raised, human review records, and the final decision.
        Returns the PDF as bytes for download via the API.

    HASH CHAIN DESIGN:
        Each AuditEvent stores the hash of the event immediately before it
        for the same claim. This creates a tamper-evident chain — if any
        historical record is altered, all subsequent hashes break.
        The first event in a chain has previous_entry_hash = "GENESIS".

    STORAGE BACKENDS:
        Primary:  AWS QLDB (immutable ledger, cryptographically verifiable)
        Fallback: DynamoDB Streams (if QLDB unavailable)
        DLQ:      SQS dead-letter queue (if both above fail)

DEPENDENCIES:
    boto3, hashlib, reportlab, shared.schemas.agent_output (AuditEvent),
    shared.config.settings
"""
