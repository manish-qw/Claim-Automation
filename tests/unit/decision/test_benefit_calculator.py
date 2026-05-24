"""
tests/unit/decision/test_benefit_calculator.py
────────────────────────────────────────────────────────────────
TESTS TO WRITE HERE:

    test_standard_approval_calculation()
        Input: base_sum = 5_000_000, bonus = 500_000, no riders,
               loan = 200_000, tds_rate = 2%.
        Assert: gross = 5_500_000, tds = 110_000, net = 5_190_000
        Assert: all values are Decimal, not float.

    test_suicide_within_12m_returns_80_percent_premiums()
        Input: cause = SUICIDE_WITHIN_12M, total_premiums_paid = 120_000
        Assert: net_payout = 96_000 (120_000 × 0.80)
        Assert: rider_payouts = 0 (no riders for suicide within window)

    test_accidental_death_rider_added_when_both_conditions_met()
        Input: FIR confirms ACCIDENT, PMR confirms accidental cause,
               rider_amount = 1_000_000
        Assert: rider_amount added to payout calculation.

    test_accidental_death_rider_not_added_if_fir_missing()
        Input: FIR not available (UNVERIFIED), PMR confirms accidental cause.
        Assert: rider_amount = 0 (both conditions not met).

    test_payout_is_integer_paise()
        Assert: net_payout_amount is int, not float, not Decimal.
        (Final value stored in ClaimContextObject must be int paise.)

    test_no_negative_payout()
        Input: loan balance > sum assured (edge case)
        Assert: net_payout_amount >= 0 (floor at zero, not negative)

    test_itemised_breakdown_contains_all_components()
        Assert: itemised_breakdown list includes entries for
                base_sum_assured, bonus, riders, loan_deduction, tds, net_payout.
"""
