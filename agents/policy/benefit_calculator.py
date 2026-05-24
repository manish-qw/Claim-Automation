"""
agents/policy/benefit_calculator.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Computes the exact net payout amount for an approved claim.
    Entirely deterministic Python — ZERO LLM involvement at any point.
    Uses Python Decimal throughout — NEVER float for monetary values.

WHAT GOES HERE:

    INPUT:
        PolicyAssessment    — from policy_rag_agent output
        premium_result      — from premium_status_checker (loan balance, ledger)
        claim_context       — for cause_of_death_type, date_of_death

    CALCULATION PATH 1 — Standard Approval:
        net_payout = (
            base_sum_assured
            + total_bonus_accrued
            + sum(active_rider_payouts)
            - outstanding_loan_balance
            - outstanding_loan_interest_accrued
            - tds_amount
        )
        All values in paise (integer). Never in rupees.

    CALCULATION PATH 2 — Suicide Within 12 Months:
        net_payout = sum(all_premiums_paid_since_inception_or_revival) × 0.80
        Source: premium payment ledger from premium_status_checker.
        Rider payouts: ZERO — no riders activated for suicide within window.

    CALCULATION PATH 3 — Accidental Death Rider:
        Add rider_amount ONLY IF BOTH conditions are met:
            Condition A: FIR confirms nature_of_incident = ACCIDENT
            Condition B: PMR confirms cause_of_death consistent with accident
        If either condition not met: rider_amount = 0 regardless of policy terms.
        Rider amount is specified in PolicyAssessment.rider_payouts.

    TDS CALCULATION (as per IT Rules 2026):
        Determine applicable TDS rate based on:
            gross payout amount
            claimant PAN availability (higher rate if PAN not provided)
        Compute:
            tds_amount = gross_payout × applicable_tds_rate
        Record tds_amount separately for Form 16A generation.
        tds_amount must be an integer (round up to nearest paise).

    BenefitCalculationResult (dataclass):
        base_sum_assured              — Decimal in paise
        bonus_accrued                 — Decimal in paise
        rider_payouts_applied         — list[{rider_name, amount}]
        outstanding_loan_deduction    — Decimal in paise
        loan_interest_deduction       — Decimal in paise
        gross_payout                  — Decimal in paise
        tds_rate                      — Decimal (e.g. Decimal("0.02") for 2%)
        tds_amount                    — Decimal in paise
        net_payout_amount             — Decimal in paise (final amount)
        calculation_path              — "STANDARD" | "SUICIDE_WITHIN_12M" |
                                        "ACCIDENTAL_DEATH"
        itemised_breakdown            — ordered list of line items (name + amount)
                                        used verbatim in the decision letter

    ITEMISED BREAKDOWN FORMAT (displayed to claimant):
        The breakdown is used exactly as-is in the claimant decision letter.
        Every component must be named clearly:
            "Base Sum Assured: ₹[X]"
            "Reversionary Bonus (21 years): ₹[X]"
            "Accidental Death Rider: ₹[X]"
            "Less: Outstanding Loan: -₹[X]"
            "Less: TDS @ 2%: -₹[X]"
            "Net Payout: ₹[X]"

DEPENDENCIES:
    decimal, shared.schemas.agent_output (PolicyAssessment),
    shared.config.settings — no LLM, no external APIs
"""
