"""
tests/unit/shared/test_claim_context.py
────────────────────────────────────────────────────────────────
TESTS TO WRITE HERE:

    test_claim_context_creation_with_valid_data()
        Create a ClaimContextObject with all required fields.
        Assert all fields are set correctly.

    test_claim_context_defaults()
        Create a ClaimContextObject with only required fields.
        Assert optional fields default correctly (empty lists, None values).

    test_claim_stage_enum_values()
        Assert all 9 ClaimStage enum values exist and have correct string values.

    test_final_decision_enum_values()
        Assert APPROVE, PARTIAL_APPROVE, DENY, ESCALATED all exist.

    test_net_payout_must_be_integer()
        Assert that setting net_payout_amount to a float raises a validation error.
        Must be integer (paise representation).

    test_uuid_auto_generated()
        Create a ClaimContextObject without providing claim_id.
        Assert that claim_id is auto-generated as a valid UUID.
"""
