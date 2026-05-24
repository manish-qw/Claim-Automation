"""
tests/unit/decision/test_escalation_evaluator.py
────────────────────────────────────────────────────────────────
TESTS TO WRITE HERE:

    test_proceed_when_all_criteria_clear()
        Input: clean claim — OUT_OF_WINDOW, fraud LOW, CRS verified,
               Aadhaar INACTIVE, no contradictions, payout < ₹25L.
        Assert: decision = PROCEED
        Assert: triggered_criteria = []

    test_escalate_on_contestability_window()
        Input: contestability_status = IN_WINDOW (all else clean)
        Assert: decision = ESCALATE
        Assert: first_trigger = "Criterion 1: ..."
        Assert: evaluation stops at criterion 1 (short-circuit)

    test_escalate_on_fraud_medium()
        Input: fraud_report.overall_risk_level = MEDIUM
        Assert: decision = ESCALATE, first_trigger contains "Criterion 2"

    test_escalate_on_crs_not_found()
        Input: crs_result.status = CRS_NOT_FOUND
        Assert: decision = ESCALATE, first_trigger contains "Criterion 3"

    test_escalate_on_aadhaar_active()
        Input: aadhaar_result.status = ACTIVE
        Assert: decision = ESCALATE, first_trigger contains "Criterion 4"

    test_escalate_on_payout_exceeds_threshold()
        Input: net_payout_amount = 3_000_000 (above ₹25L threshold)
        Assert: decision = ESCALATE, first_trigger contains "Criterion 7"

    test_escalate_on_suicide_claim()
        Input: cause_of_death_type = SUICIDE_WITHIN_12M
        Assert: decision = ESCALATE, first_trigger contains "Criterion 8"

    test_escalate_on_high_uncertainty()
        Input: uncertainty_score = 0.80 (above 0.65 threshold)
        Assert: decision = ESCALATE, first_trigger contains "Criterion 9"

    test_short_circuit_stops_at_first_match()
        Input: contestability IN_WINDOW + fraud CRITICAL (both trigger)
        Assert: only Criterion 1 is in triggered_criteria (evaluation stopped)
"""
