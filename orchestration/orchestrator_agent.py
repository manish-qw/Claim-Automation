"""
orchestration/orchestrator_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    The meta-agent. Monitors LangGraph execution, reacts to all agent
    completion events, makes dynamic routing decisions, and manages
    confidence thresholds and trust-score injection.

WHAT GOES HERE:

    KAFKA SUBSCRIPTIONS:
        Subscribes to ALL *_COMPLETE topics simultaneously.
        On each event: reads the AgentOutput envelope from the claim context
        and applies the logic below.

    ON EACH AGENT COMPLETION EVENT:

    1. Confidence Threshold Check:
        If agent_output.confidence < AGENT_CONFIDENCE_MINIMUM (0.50):
            → Trigger ONE retry of that agent with a rephrased prompt
              (add context: "Your previous attempt had confidence [X].
               Please re-examine the evidence more carefully.")
            If retry ALSO < 0.50:
                → Mark that agent's fields as UNVERIFIED
                → Continue pipeline — do NOT block indefinitely
                → Add to synthesis_agent's UNVERIFIED fields list

    2. CRITICAL Flag Check:
        If any FlagObject with severity = CRITICAL is present:
            → Trigger IMMEDIATE ESCALATION regardless of pipeline stage
            → Skip remaining agents (they still run for audit completeness)
            → Log as EARLY_ESCALATION in audit trail

    3. Trust Score Warning Injection:
        If trust_score_engine result shows aggregate_trust < 0.65:
            → Inject the warning string into the context of all
              not-yet-started downstream agents' prompts
            → Log injection as PROMPT_AUGMENTATION in audit trail

    4. Post-Synthesis Routing:
        After SYNTHESIS_COMPLETE: run escalation_evaluator.
        Based on result: route to decision_node or human_escalation_node.

    PLAN-AND-EXECUTE STRATEGY:
        At claim intake: generate an execution plan documenting:
            - which agents will run
            - in which order and which run in parallel
            - timeout for each agent (from settings)
            - retry rules per agent
        Log the plan as an AuditEvent of type EXECUTION_PLAN_CREATED.
        Monitor actual execution against the plan.
        If agents return unexpected states: generate a REPLAN and log it
        as EXECUTION_REPLAN in audit trail.

    AGENT TIMEOUT HANDLING:
        Each agent has a configurable max_runtime_seconds (from settings).
        If agent exceeds timeout: mark as DEGRADED, continue pipeline.
        Log timeout as AGENT_TIMEOUT in audit trail.

DEPENDENCIES:
    shared.events.kafka_client, shared.db.claim_repository,
    shared.config.settings, shared.audit.audit_service,
    agents.decision.escalation_evaluator,
    orchestration.langgraph_workflow
"""
