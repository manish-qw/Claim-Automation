"""
agents/verification/premium_status_checker.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Confirms the exact policy status on the date of death — not today.
    Also detects third-party premium payment (a Sambhal mafia fraud pattern).

WHAT GOES HERE:

    INPUT:
        policy_id      — from claim context
        date_of_death  — from extracted DEATH_CERTIFICATE fields

    DATA FETCH (from internal policy system API):
        full_premium_payment_ledger — all premium payments with dates and payer
        lapse_periods               — list of lapse start/end dates
        revival_dates               — list of revival dates and amounts
        revival_amounts             — premiums paid at each revival
        outstanding_loan_balance    — current loan against policy (if any)
        accrued_loan_interest       — interest accrued on outstanding loan

    POLICY STATUS CALCULATION (on date_of_death specifically — NOT today):
        Was the policy in lapse on the exact date_of_death?
        Was it in the grace period (typically 30 days after premium due)?
        Was it in revival pending status?
        Handle edge case: policy that lapsed and was revived recently —
            status on date_of_death depends on exact revival date vs. death date.

        policy_status_on_dod (enum):
            ACTIVE          — fully active on the date of death
            LAPSED          — lapsed with no revival by date of death
            GRACE_PERIOD    — lapsed but within the 30-day grace period
            REVIVAL_PENDING — revival initiated but not completed

    THIRD-PARTY PREMIUM PAYMENT DETECTION:
        Review the full premium ledger for each payment:
            Was the premium paid from an account not belonging to the policyholder?
        If YES and the policy is in the contestability window:
            → Set third_party_premium_flag = True
            → This is a HIGH severity fraud signal (Sambhal mafia pattern:
              fraudsters pay premiums on policies of their victims to keep
              them active)

    PremiumStatusResult (dataclass):
        policy_id                  — UUID
        policy_status_on_dod       — enum (see above)
        premium_ledger_summary     — {total_paid, last_paid_date, payment_count}
        third_party_premium_flag   — bool
        third_party_payments       — list[{date, amount, payer_account_last4}]
        outstanding_loan_balance   — Decimal (in paise)
        accrued_loan_interest      — Decimal (in paise)
        checked_at                 — UTC timestamp

DEPENDENCIES:
    httpx (policy system internal API), decimal,
    shared.config.settings, shared.schemas.claim_context,
    shared.audit.audit_service
"""
