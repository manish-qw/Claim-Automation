"""
agents/verification/external_verification_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Master orchestrator for all external API verification checks.
    Runs in PARALLEL with Document Intelligence Agent — no data dependency
    between the two. Both start immediately after intake completes.

WHAT GOES HERE:

    PARALLEL VERIFIERS (launched simultaneously via asyncio.gather):
        1. aadhaar_verifier              — always runs
        2. crs_registry_verifier         — always runs
        3. claim_history_checker         — always runs
        4. premium_status_checker        — always runs
        5. network_graph_agent           — always runs
        6. underwriting_mismatch_agent   — ONLY if contestability_status = IN_WINDOW

    VerificationReport (dataclass — written to claim context):
        aadhaar_result         — AadhaarVerificationResult
        crs_result             — CRSVerificationResult
        claim_history_result   — ClaimHistoryResult
        premium_result         — PremiumStatusResult
        network_flags          — list[NetworkFlag]
        underwriting_result    — UnderwritingResult | None
        overall_verified       — bool (True only if all mandatory checks passed)
        unverified_items       — list of items that could not be verified
                                 with reason code per item

    API FAILURE HANDLING:
        For each failed API call: mark the specific item as UNVERIFIED
        with a reason code:
            API_TIMEOUT, API_UNAVAILABLE, RATE_LIMITED, AUTH_ERROR
        Never block the pipeline on an external API failure.
        Log the failure and continue with the remaining verifiers.

    CIRCUIT BREAKER (per external API endpoint):
        After 5 consecutive failures within 2 minutes:
            → Open the circuit (stop calling that API immediately)
            → Return UNAVAILABLE for all calls to that endpoint
        After 60 seconds: half-open (allow one test call)
        If test call succeeds: close circuit, resume normal calls
        If test call fails: keep circuit open, reset 60s timer

    KAFKA PUBLISH:
        On completion (all 6 verifiers done, regardless of individual failures):
        Publish EXTERNAL_VERIFY_COMPLETE to Kafka.
        The fraud pipeline waits for BOTH DOC_INTEL_COMPLETE AND
        EXTERNAL_VERIFY_COMPLETE before starting.

DEPENDENCIES:
    asyncio, agents.verification.aadhaar_verifier,
    agents.verification.crs_registry_verifier,
    agents.verification.claim_history_checker,
    agents.verification.premium_status_checker,
    agents.verification.network_graph_agent,
    agents.verification.underwriting_mismatch_agent,
    shared.events.kafka_client, shared.db.claim_repository,
    shared.schemas.agent_output
"""
