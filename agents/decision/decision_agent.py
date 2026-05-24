"""
agents/decision/decision_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    The final autonomous decision maker. Only reached if escalation_evaluator
    returns PROCEED (none of the 9 escalation criteria were triggered).
    Uses Constitutional AI + Chain-of-Thought strategy.

WHAT GOES HERE:

    INPUTS:
        synthesis_summary   — master summary from synthesis_agent
        fraud_report        — FraudReport from fraud_intelligence_agent
        policy_assessment   — PolicyAssessment from policy_rag_agent
        conflict_report     — ConflictReport from conflict_resolution
        benefit_calculation — BenefitCalculationResult from benefit_calculator

    STEP 1 — Re-Check 9 Constitutional Rules (defence in depth):
        Re-run all 9 escalation_evaluator criteria independently.
        If ANY fires at this stage:
            → Force ESCALATED output
            → Log as LATE_ESCALATION in audit trail (this means
              escalation_evaluator had a bug — critical to detect)
        This double-check catches any race conditions or state errors.

    STEP 2 — Claude Opus Reasoning (Constitutional AI + CoT):
        Structured prompt instructs the model to:

        a) Address EVERY CONFLICTED field by name:
               "For the conflict on [field], I resolve it in favour of
                [source] because [specific reason based on evidence]."

        b) Evaluate EVERY fraud signal:
               "The [signal_name] flag is / is not determinative because
                [specific evidence-based reasoning]."

        c) Cite policy clauses for EVERY coverage item:
               "Coverage for [item] is determined by clause [ID]:
                [exact clause text excerpt]."

        d) Produce TWO outputs:
               technical_rationale — full detailed reasoning for audit
               plain_language_explanation — max 150 words, for claimant letter

    STEP 3 — Final Decision:
        APPROVE         — all coverage confirmed, no unresolved issues
        PARTIAL_APPROVE — coverage confirmed for some items, not all
        DENY            — coverage excluded by policy or fraud confirmed

    STEP 4 — Trigger Downstream Agents:
        APPROVE or PARTIAL_APPROVE:
            → Trigger settlement_agent (pay out)
            → Trigger communications_agent (approval letter)
        DENY:
            → Trigger communications_agent only (denial letter with
              Ombudsman details and appeal window)

    OUTPUT:
        Write final_decision + net_payout_amount to ClaimContextObject.
        Write full AgentOutput with reasoning_trace (never truncated).
        Publish DECISION_COMPLETE to Kafka.
        Log complete reasoning trace in AuditEvent.

DEPENDENCIES:
    shared.llm.llm_client (LLMModel.OPUS),
    agents.decision.escalation_evaluator,
    agents.policy.benefit_calculator,
    shared.events.kafka_client, shared.db.claim_repository,
    shared.schemas.agent_output, shared.audit.audit_service
"""
