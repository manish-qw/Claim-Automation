"""
tests/integration/test_intake_flow.py
────────────────────────────────────────────────────────────────
PURPOSE:
    Integration tests for the claim intake flow.
    Requires: docker-compose up (PostgreSQL + Kafka running).
    Uses real DB + Kafka but mock external APIs.

TESTS TO WRITE HERE:

    test_new_claim_submitted_via_web_form()
        POST to /v1/claims with a valid multipart web form payload.
        Assert: HTTP 201 response with claim_id in body.
        Assert: claim record exists in PostgreSQL with status = INTAKE.
        Assert: CLAIM_REGISTERED event published to Kafka topic.
        Assert: acknowledgement queued in Communications queue.

    test_duplicate_claim_rejected()
        Submit the same claim twice (same policy_id + incident_date).
        Assert: second submission returns HTTP 409 with DUPLICATE_REJECTED.
        Assert: only one claim record in database.

    test_unknown_cause_blocks_pipeline()
        Submit claim with cause_of_death = UNKNOWN.
        Assert: claim created but current_stage = DOCUMENTS_PENDING.
        Assert: follow-up request for PMR queued in Communications queue.
        Assert: no downstream agents triggered.

    test_contestability_window_detected()
        Submit claim where policy inception = 100 days ago.
        Assert: pre_intake_router sets contestability_status = IN_WINDOW.
        Assert: routing_flags in claim context reflect IN_WINDOW.

    test_missing_mandatory_doc_triggers_followup()
        Submit NATURAL death claim without DEATH_CERTIFICATE.
        Assert: missing_doc_tracker identifies DEATH_CERTIFICATE as mandatory missing.
        Assert: follow-up request for DEATH_CERTIFICATE queued.
        Assert: claim continues processing available documents.
"""
