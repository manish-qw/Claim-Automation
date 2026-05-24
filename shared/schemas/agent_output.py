"""
shared/schemas/agent_output.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Standard output envelope that EVERY agent in the system wraps its results
    in before writing to the claim context. The orchestrator reads these
    envelopes to make routing decisions.

WHAT GOES HERE:
    Pydantic v2 models for:

    AgentOutput  (the universal wrapper — every agent returns this):
        agent_id          — string identifier e.g. "fraud_intelligence_agent"
        claim_id          — UUID
        status            — enum: COMPLETE | DEGRADED | FAILED
        confidence        — float 0.0–1.0, agent's self-assessed reliability
        completeness      — float 0.0–1.0, % of expected fields populated
        output            — dict, agent-specific payload
        flags             — list[FlagObject]
        unverified_fields — list of field names that could not be verified
        produced_at       — UTC timestamp
        model_used        — e.g. "claude-opus-4", "claude-sonnet-4"
        prompt_version    — e.g. "fraud-advocate-v2.3"
        duration_ms       — integer

    FlagObject  (raised by any agent for any anomaly):
        flag_type         — string e.g. "AADHAAR_ACTIVE_ON_DECEASED"
        severity          — enum: CRITICAL | HIGH | MEDIUM | LOW
        evidence_doc      — document_id that triggered the flag
        evidence_field    — specific field name within that document
        explanation       — plain English, max 100 words

    FraudReport  (specific output schema of Fraud Intelligence Agent):
        overall_risk_level    — enum: LOW | MEDIUM | HIGH | CRITICAL
        risk_score            — float 0.0–1.0
        driving_signals       — list of top 3 signals (signal_type, weight,
                                 evidence_field, explanation)
        legitimate_explanations — list of strings from Defense sub-agent
        recommendation        — enum: AUTO_PROCESS | FLAG_FOR_REVIEW |
                                       ESCALATE | INVESTIGATE
        aadhaar_status        — enum: INACTIVE | ACTIVE | UNVERIFIED
        crs_verified          — bool
        network_flags         — list of connections found in fraud graph

    PolicyAssessment  (specific output schema of Policy RAG Agent):
        coverage_determination — enum: FULL | PARTIAL | NONE
        covered_items          — list[{item_description, clause_id,
                                        clause_text_excerpt, amount}]
        excluded_items         — list[{item_description, clause_id, reason}]
        base_sum_assured       — integer in paise
        bonus_amount           — integer in paise
        rider_payouts          — list[{rider_name, amount,
                                        trigger_condition_met: bool}]
        outstanding_loan_deduction — integer in paise
        tds_applicable         — bool
        tds_amount             — integer in paise
        net_payout_amount      — integer in paise
        ambiguous_clauses      — list[{clause_id, ambiguity_description}]
        policy_version_used    — string
        all_citations_verified — bool (set by grounding validator)

    EscalationPackage  (delivered to human reviewer):
        claim_snapshot         — dict of key claim facts
        escalation_reason      — single plain English sentence
        completed_verifications — list of what AI confirmed
        open_questions         — list of specific questions for human
        ai_recommendation      — {suggested_action, reasoning} — non-binding
        document_links         — list[{doc_type, url, highlighted_fields}]
        audit_trail_link       — URL to audit viewer
        sla_deadline           — UTC timestamp
        priority_level         — enum: HIGH | MEDIUM | LOW
        decision_form_options  — structured options for human input

    AuditEvent  (written by every agent for every action):
        audit_id              — UUID
        claim_id              — UUID
        agent_id              — string
        action_type           — e.g. "ASSESSMENT_COMPLETE", "FLAG_RAISED"
        input_hash            — SHA-256 of agent input payload
        output_hash           — SHA-256 of agent output payload
        previous_entry_hash   — SHA-256 of the preceding audit entry (chain)
        model_version         — exact model string
        prompt_version        — version tag
        tool_calls            — list[{tool_name, params_hash, response_hash}]
        confidence            — float
        reasoning_trace       — full CoT text (never truncated)
        timestamp             — UTC, millisecond precision
        duration_ms           — integer

    ENUMS ALSO DEFINED HERE:
        AgentStatus, FlagSeverity, RiskLevel, FraudRecommendation,
        CoverageDetermination, EscalationPriority, AadhaarStatus

DEPENDENCIES:
    pydantic v2, uuid, datetime
"""
