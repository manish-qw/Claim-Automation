"""
agents/decision/escalation_evaluator.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Evaluates nine hard escalation criteria. Returns ESCALATE or PROCEED.
    These are the INVIOLABLE constitutional rules of the system.
    No LLM can override them. No confidence score bypasses them.

WHAT GOES HERE:

    RETURN TYPE:
        EscalationDecision (dataclass):
            decision          — "ESCALATE" | "PROCEED"
            triggered_criteria — list[str] — specific criteria that triggered
                                  (only populated if ESCALATE)
            first_trigger      — str — the first criterion that matched
                                  (evaluation stops at first match)

    THE NINE CRITERIA (checked IN ORDER — first match returns ESCALATE):

    Criterion 1: Contestability Window
        Condition: claim_context.contestability_status == IN_WINDOW
        Trigger message: "Criterion 1: Policy is within the 730-day
                          contestability window from inception/revival."

    Criterion 2: Fraud Risk MEDIUM or Above
        Condition: fraud_report.overall_risk_level in [MEDIUM, HIGH, CRITICAL]
        Trigger message: "Criterion 2: Fraud Intelligence Agent returned
                          risk level [X]."

    Criterion 3: CRS Not Found
        Condition: verification_report.crs_result.status == CRS_NOT_FOUND
        Trigger message: "Criterion 3: Death certificate registration number
                          not verifiable in CRS government database."

    Criterion 4: Aadhaar Active on Deceased
        Condition: verification_report.aadhaar_result.status == ACTIVE
        Trigger message: "Criterion 4: Deceased's Aadhaar remains active."

    Criterion 5: Cause-of-Death Inconsistency
        Condition: any CRITICAL ContradictionObject where
                   field_name contains "cause_of_death"
        Trigger message: "Criterion 5: Cause of death inconsistency detected
                          across documents."

    Criterion 6: Legal Heir Track
        Condition: legal_heir_track_activated == True (nominee is deceased,
                   alternative heir documents were submitted)
        Trigger message: "Criterion 6: Named nominee is deceased —
                          legal heir track activated."

    Criterion 7: Payout Exceeds Auto-Approve Threshold
        Condition: benefit_calculator.net_payout_amount >
                   settings.AUTO_APPROVE_MAX_SUM_ASSURED
        Trigger message: "Criterion 7: Net payout [₹X] exceeds
                          auto-approve threshold of ₹25 lakhs."

    Criterion 8: Suicide Claim
        Condition: cause_of_death_type in [SUICIDE_WITHIN_12M, SUICIDE_AFTER_12M]
        Trigger message: "Criterion 8: Claim involves suicide —
                          mandatory human review."

    Criterion 9: Uncertainty Score Exceeds Threshold
        Condition: claim_context.uncertainty_score >
                   settings.UNCERTAINTY_SCORE_ESCALATION_THRESHOLD
        Trigger message: "Criterion 9: Composite uncertainty score [X]
                          exceeds threshold of 0.65."

    EVALUATION LOGIC:
        Check criteria 1 through 9 in exact order.
        Return ESCALATE at the first match (short-circuit evaluation).
        If no criteria match: return PROCEED.
        Never aggregate or weight criteria — any single match = ESCALATE.

DEPENDENCIES:
    shared.schemas.agent_output (FraudReport, PolicyAssessment),
    shared.schemas.claim_context (ClaimContextObject),
    shared.config.settings — no LLM, no external APIs
"""
