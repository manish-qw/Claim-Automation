"""
agents/verification/claim_history_checker.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Checks internal PostgreSQL claims database for prior claim history
    related to this policy and claimant. Provides signals to the
    Fraud Intelligence Agent.

WHAT GOES HERE:

    INPUT:
        policy_id    — from claim context
        claimant_id  — from claim context
        nominee_id   — extracted from claim documents

    DATABASE QUERIES (via claim_repository — no raw SQL):

    Query 1: Has this policy_id ever had a prior death claim?
        Returns: prior_death_claim_count (int), last_death_claim_date

    Query 2: Has this nominee filed claims on multiple policies
             in the last 12 months?
        Returns: same_nominee_multi_policy_flag (bool),
                 policies_claimed_count (int)

    Query 3: Does this claimant have any fraud flags from prior claims?
        Returns: prior_fraud_flags (list of FlagType strings),
                 highest_prior_flag_severity

    Query 4: Are there soft-duplicate signals across this claimant's history?
        Returns: soft_duplicate_claims (list of claim_ids with similarity scores)

    ClaimHistoryResult (dataclass):
        policy_id                    — UUID
        prior_claims_count           — int
        prior_fraud_flags            — list[str]
        last_claim_date              — date | None
        same_nominee_multi_policy_flag — bool
        policies_claimed_count       — int
        soft_duplicate_history       — list[{claim_id, similarity_score}]
        checked_at                   — UTC timestamp

    FRAUD SIGNAL GENERATION:
        All findings are returned as structured data — this module does NOT
        raise FlagObjects directly. The Fraud Intelligence Agent reads
        ClaimHistoryResult and decides which signals to include in the
        FraudReport based on their combined weight with other signals.

    PERFORMANCE:
        All queries run against indexed columns (policy_id, claimant_id,
        nominee_id). Should complete in < 100ms for typical database sizes.
        No full-table scans permitted.

DEPENDENCIES:
    shared.db.claim_repository, shared.schemas.claim_context,
    shared.audit.audit_service
"""
