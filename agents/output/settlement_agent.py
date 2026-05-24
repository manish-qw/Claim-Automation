"""
agents/output/settlement_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Executes the financial payout for approved claims.
    Entirely deterministic — ZERO LLM involvement.
    Called after decision_agent returns APPROVE or PARTIAL_APPROVE.

WHAT GOES HERE:

    PRE-PAYMENT VALIDATIONS (run all before initiating payment):

    Validation 1 — Policy still active:
        Re-verify the policy has not been cancelled between the decision
        and settlement (edge case: policy cancellation can take time to
        propagate through systems).
        If cancelled: halt settlement, route to manual finance queue.

    Validation 2 — IFSC code valid:
        Call RBI IFSC validation API.
        Verify the IFSC code from bank proof document is active and valid.
        If invalid: halt, request updated bank details from claimant.

    Validation 3 — Account holder name match:
        Compare account_holder_name from bank proof against claimant's
        identity-verified name (Aadhaar/PAN name).
        Fuzzy match threshold: 85% (Levenshtein).
        If below threshold: halt, flag for manual review.

    Validation 4 — Amount does not exceed maximum:
        net_payout_amount ≤ policy_assessment.net_payout_amount (max payable).
        Prevents calculation errors from paying more than policy allows.

    PAYMENT API CALL:
        POST to banking_api.BASE_URL/v1/settlements with:
            claim_id, net_amount, tds_amount, account_details, reference_number
        Idempotency key: "{claim_id}_{settlement_attempt_number}"
            → Safe to retry without double payment.

    RETRY LOGIC:
        On banking API failure: retry up to 3 times.
        Backoff schedule: 2s → 4s → 8s.
        If all 3 attempts fail:
            → Publish SETTLEMENT_FAILED event to Kafka
            → Route to human finance queue with full payout details
            → Notify claimant of delay via communications queue
            → Continue monitoring IRDAI SLA during delay

    SettlementResult (dataclass):
        claim_id             — UUID
        transaction_id       — from banking API response
        amount_settled       — Decimal in paise
        tds_deducted         — Decimal in paise
        settlement_date      — UTC timestamp
        attempt_number       — int (1, 2, or 3)
        status               — "SUCCESS" | "FAILED"

    AUDIT LOGGING:
        Log every API call attempt with:
            request payload hash, response payload hash, latency_ms.
        Store in AuditEvent.tool_calls list.
        Full request/response hashes allow tamper detection of settlement records.

    KAFKA PUBLISH:
        On SUCCESS: publish SETTLEMENT_COMPLETE to Kafka.
        On FAILED: publish SETTLEMENT_FAILED to Kafka.

DEPENDENCIES:
    httpx (banking API, RBI IFSC API), Levenshtein, decimal,
    shared.events.kafka_client, shared.db.claim_repository,
    shared.config.settings, shared.audit.audit_service
"""
