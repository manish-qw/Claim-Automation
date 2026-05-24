"""
shared/events/kafka_client.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Centralised Kafka producer and consumer wrappers for all inter-agent
    event messaging. No agent creates its own Kafka connection — all import
    from this module.

WHAT GOES HERE:

    TOPIC CONSTANTS (string constants, one per event type):
        CLAIM_REGISTERED        — published by Intake Agent at submission
        INTAKE_COMPLETE         — published by Intake Agent after ack sent
        DOC_INTEL_COMPLETE      — published by Document Intelligence Agent
        EXTERNAL_VERIFY_COMPLETE — published by External Verification Agent
        FRAUD_COMPLETE          — published by Fraud Intelligence Agent
        POLICY_COMPLETE         — published by Policy RAG Agent
        SYNTHESIS_COMPLETE      — published by Synthesis Agent
        DECISION_COMPLETE       — published by Decision Agent
        ESCALATION_TRIGGERED    — published by Escalation Evaluator
        SETTLEMENT_COMPLETE     — published by Settlement Agent
        AUDIT_EVENT             — published by every agent for every action

    FUNCTIONS:

    publish(topic: str, claim_id: str, payload: dict) → None
        Produces a single event to the specified topic.
        Event envelope wraps the payload with:
            topic, claim_id, producer_agent_id, timestamp,
            schema_version, payload
        Fire-and-forget with internal error handling and retry.

    subscribe(topic: str, group_id: str, handler: Callable) → None
        Registers a consumer for a topic with a specific consumer group.
        Consumer groups are defined per agent to ensure exactly-once delivery.
        handler is called with the full event envelope on each message.

    publish_batch(events: list[tuple[str, str, dict]]) → None
        Bulk-publishes multiple events atomically (e.g. when an agent
        produces multiple downstream events simultaneously).

    CONSUMER GROUP DEFINITIONS:
        Each agent has a fixed group_id — defined as constants here.
        This ensures Kafka delivers each event to exactly one instance
        of each agent type (no duplicate processing on scale-out).

    CONNECTION MANAGEMENT:
        aiokafka AIOKafkaProducer and AIOKafkaConsumer.
        Connection bootstraps from KAFKA_BOOTSTRAP_SERVERS in settings.
        Graceful shutdown hook — flushes pending messages before exit.

DEPENDENCIES:
    aiokafka, shared.config.settings
"""
