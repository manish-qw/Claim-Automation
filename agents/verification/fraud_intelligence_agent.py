"""
agents/verification/fraud_intelligence_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Master fraud orchestrator. Runs AFTER both DOC_INTEL_COMPLETE AND
    EXTERNAL_VERIFY_COMPLETE Kafka events have been received.
    Coordinates three fraud detection layers and a parallel network graph
    check, then aggregates results into a final FraudReport.

WHAT GOES HERE:

    TRIGGER CONDITION:
        Subscribes to both DOC_INTEL_COMPLETE and EXTERNAL_VERIFY_COMPLETE.
        Does not start until BOTH have been received for the same claim_id.
        Uses a claim-level gate stored in Redis/PostgreSQL to track which
        events have arrived.

    THREE-LAYER ARCHITECTURE:

    Layer 1 — fraud_rules_engine (ALWAYS runs first, fast, no LLM):
        Deterministic Python rules. Returns: rules_score (float), matched_rules list.
        Always completes in < 500ms.

    Layer 2 — anomaly_detector (runs in parallel with Layer 1):
        Isolation Forest ML model. Returns: anomaly_score (float), top_3_features.
        Always completes in < 2s.

    Layer 3 — fraud_debate_agents (CONDITIONAL):
        Three Claude Opus LLM agents (FraudAdvocate → Defense → Synthesis).
        ONLY runs if: Layer 1 returns any HIGH+ severity rule match
                   OR Layer 2 returns anomaly_score > 0.3
        If neither condition: skip debate, proceed with rules + anomaly only.
        This prevents LLM costs on clean, simple, low-risk claims.

    PARALLEL: network_graph_agent runs alongside Layers 1 and 2.

    SCORE AGGREGATION FORMULA:
        final_score = (rules_score × 0.40) + (anomaly_score × 0.30)
                    + (llm_debate_score × 0.30)
        CRITICAL OVERRIDE: if ANY single layer returns CRITICAL,
        the final overall_risk_level = CRITICAL regardless of other scores.

    EVIDENCE CITATION VALIDATION:
        Before writing the FraudReport: validate that every signal in
        driving_signals cites a specific evidence_field from a specific
        document. Signals without evidence citations are REJECTED and the
        agent retries that signal description. No vague signals permitted.

    OUTPUT:
        Writes FraudReport to ClaimContextObject via claim_repository.
        Publishes FRAUD_COMPLETE to Kafka.

DEPENDENCIES:
    agents.verification.fraud_rules_engine,
    agents.verification.anomaly_detector,
    agents.verification.fraud_debate_agents,
    agents.verification.network_graph_agent,
    shared.events.kafka_client, shared.db.claim_repository,
    shared.schemas.agent_output (FraudReport)
"""
