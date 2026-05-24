"""
agents/verification/fraud_debate_agents.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Layer 3 of fraud detection. Three Claude Opus agents run sequentially,
    each building on the previous agent's output. Produces a balanced,
    regulatory-grade fraud assessment with both prosecution and defense
    perspectives.

    ⚠️ ONLY RUNS IF: Layer 1 has any HIGH+ severity rule match
                  OR Layer 2 anomaly_score > 0.3
    Otherwise: this entire module is skipped to avoid unnecessary LLM costs.

WHAT GOES HERE:

    AGENT 1 — FraudAdvocate:
        System prompt instructs: build the strongest possible case FOR fraud.
        Input: all extracted fields, Layer 1 rule matches, Layer 2 anomalies,
               verification results, network graph connections.
        Must: cite a specific document field as evidence for EVERY signal raised.
        Cannot: say "appears suspicious" without explaining the exact causal mechanism.
        Cannot: make a final judgment — only presents evidence.
        Cannot: speculate about evidence not present in the provided data.
        Output: AdvocateReport — list of signals with evidence citations.

    AGENT 2 — Defense:
        Receives: AdvocateReport from Agent 1.
        System prompt instructs: find the most plausible innocent explanation
        for each signal the Advocate raised.
        Must: consider India-specific legitimate context:
            - Administrative delays in government record-keeping
            - Regional variations in documentation practices
            - Family financial circumstances explaining payment patterns
            - Rural/remote area delays in filing documents
        Cannot: dismiss signals without providing a specific alternative explanation.
        Output: DefenseReport — for each advocate signal, an innocent explanation.

    AGENT 3 — Synthesis:
        Receives: AdvocateReport + DefenseReport.
        System prompt instructs: weigh each signal independently.
        For each signal: "Is the advocate's case or the defense's explanation
        more probative given ALL available evidence?"
        Must produce:
            overall_risk_level        — LOW | MEDIUM | HIGH | CRITICAL
            driving_signals           — top 3 signals with weight and explanation
            legitimate_explanations   — top 3 defense explanations that hold up
            recommendation            — AUTO_PROCESS | FLAG_FOR_REVIEW | ESCALATE | INVESTIGATE
            regulatory_narrative      — max 200 words, plain English, could be
                                        read to a regulator or in a legal proceeding

    DebateResult (dataclass):
        advocate_signals     — list from Agent 1
        defense_explanations — list from Agent 2
        synthesis_output     — SynthesisOutput from Agent 3
        llm_debate_score     — float 0–1 derived from synthesis risk_level

    PROMPT VERSIONS:
        All three agents use versioned prompts tracked in prompt_version field.
        e.g. "fraud-advocate-v2.3", "fraud-defense-v1.8", "fraud-synthesis-v3.1"
        Prompt versions are logged in every AuditEvent.

DEPENDENCIES:
    shared.llm.llm_client (LLMModel.OPUS),
    agents.verification.fraud_rules_engine (RulesEngineOutput),
    agents.verification.anomaly_detector (AnomalyResult),
    shared.schemas.agent_output (FraudReport),
    shared.audit.audit_service
"""
