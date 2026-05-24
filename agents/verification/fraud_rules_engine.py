"""
agents/verification/fraud_rules_engine.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Layer 1 of fraud detection. Pure deterministic Python rules — NO LLM,
    NO ML. Fast, always runs first. Rules are loaded from the fraud_rules
    PostgreSQL table so thresholds can be updated without code deployment.

WHAT GOES HERE:

    FRAUD RULES (implement each as a function returning RuleResult):
    Rules are specific to death claims. Each rule has a configurable severity.

    CRITICAL severity rules:
        aadhaar_active_on_deceased
            Condition: aadhaar_verifier.status == ACTIVE
            Explanation: "Deceased's Aadhaar remains active — inconsistent
                          with a verified death."

        crs_certificate_not_found
            Condition: crs_verifier.status == CRS_NOT_FOUND
            Explanation: "Death certificate registration number not found
                          in government CRS database."

        pmr_cause_differs_from_death_cert
            Condition: cross_document_checker flagged cause-of-death INCONSISTENT
            Explanation: "Cause of death in Postmortem Report contradicts
                          the Death Certificate."

    HIGH severity rules:
        hospital_not_in_nha_registry
            Condition: hospital verification returns NOT_FOUND in NHA database
        claim_within_90_days_of_inception
            Condition: days_from_inception < 90
        fir_filed_48hrs_after_incident
            Condition: fir_filed_date > incident_date + 48 hours
        same_doctor_in_3plus_claims
            Condition: network_graph returns doctor with degree (connections) > 3
        third_party_premium_payer
            Condition: premium_checker.third_party_premium_flag == True
                       AND contestability_status == IN_WINDOW

    MEDIUM severity rules:
        claim_intimated_within_48hrs_of_death
            Condition: intimation_date < death_date + 48 hours
        lapse_revival_within_6_months_before_death
            Condition: last_revival_date > death_date - 180 days
        nominee_multi_policy_claims
            Condition: claim_history.same_nominee_multi_policy_flag == True

    RuleResult (dataclass):
        rule_name    — string
        matched      — bool
        severity     — FlagSeverity | None
        evidence     — dict of {field_name: value} that triggered the rule
        explanation  — plain English, max 50 words

    RulesEngineOutput (dataclass):
        matched_rules   — list[RuleResult]
        rules_score     — float 0–1 (weighted by severity of matched rules)
        highest_severity — FlagSeverity | None

    SCORE CALCULATION:
        CRITICAL match → rules_score = 1.0 (max)
        HIGH match     → contributes 0.70
        MEDIUM match   → contributes 0.40
        Multiple matches: take the MAX, don't sum (avoid double-counting)

DEPENDENCIES:
    shared.schemas.agent_output (FlagSeverity),
    shared.db.claim_repository (for rule config loading),
    shared.config.settings
"""
