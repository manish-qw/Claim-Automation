"""
tests/unit/verification/test_fraud_rules_engine.py
────────────────────────────────────────────────────────────────
TESTS TO WRITE HERE:

    test_aadhaar_active_triggers_critical()
        Input: aadhaar_status = ACTIVE
        Assert: rule 'aadhaar_active_on_deceased' matched = True
        Assert: severity = CRITICAL
        Assert: rules_score >= 1.0

    test_aadhaar_inactive_no_flag()
        Input: aadhaar_status = INACTIVE
        Assert: 'aadhaar_active_on_deceased' matched = False

    test_crs_not_found_triggers_critical()
        Input: crs_status = CRS_NOT_FOUND
        Assert: rule 'crs_certificate_not_found' matched = True
        Assert: severity = CRITICAL

    test_claim_within_90_days_triggers_high()
        Input: days_from_inception = 45
        Assert: rule 'claim_within_90_days_of_inception' matched = True
        Assert: severity = HIGH

    test_claim_after_90_days_no_flag()
        Input: days_from_inception = 120
        Assert: 'claim_within_90_days_of_inception' matched = False

    test_third_party_premium_in_contestability_triggers_high()
        Input: third_party_premium_flag = True, contestability_status = IN_WINDOW
        Assert: rule 'third_party_premium_payer' matched = True
        Assert: severity = HIGH

    test_third_party_premium_out_of_window_no_flag()
        Input: third_party_premium_flag = True, contestability_status = OUT_OF_WINDOW
        Assert: 'third_party_premium_payer' matched = False

    test_multiple_rules_score_takes_max()
        Input: two HIGH-severity rules triggered
        Assert: rules_score = max of their individual contributions (not sum)

    test_critical_override_on_multiple_rules()
        Input: one CRITICAL rule + one HIGH rule
        Assert: highest_severity = CRITICAL
        Assert: rules_score = 1.0
"""
