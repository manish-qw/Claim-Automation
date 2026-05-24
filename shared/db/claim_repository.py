"""
shared/db/claim_repository.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    The sole PostgreSQL data access layer for claim data. Every agent that
    reads or writes claim information must go through this module.
    No raw SQL is permitted anywhere else in the codebase.

WHAT GOES HERE:
    An async repository class (using asyncpg) with these functions:

    create_claim(claim_context: ClaimContextObject) → str
        Inserts a new claim record into PostgreSQL. Returns the claim_id.
        Carries an idempotency key — safe to call multiple times without
        creating duplicate records.

    get_claim(claim_id: str) → ClaimContextObject
        Fetches the full claim object (all nested fields) for a given claim_id.
        Deserialises JSON columns back into Pydantic models.

    update_agent_output(claim_id: str, agent_id: str, output: AgentOutput) → None
        Writes a single agent's output to the claim record.
        Uses OPTIMISTIC LOCKING with a version counter — if two parallel
        agents (e.g. Doc Intel + Ext Verify) complete simultaneously,
        this prevents a write conflict / lost update.

    update_claim_status(claim_id: str, status: ClaimStage) → None
        Updates the current_stage field of the claim.

    append_flag(claim_id: str, flag: FlagObject) → None
        Atomically appends a new flag to the claim's flags array.
        Never overwrites existing flags.

    get_claims_by_status(status: ClaimStage, limit: int) → list[ClaimContextObject]
        Returns claims currently at a given stage, ordered by sla_deadline ASC.
        Used by the dashboard queue to show the most urgent claims first.

    get_escalated_claims() → list[ClaimContextObject]
        Returns all claims in ESCALATED stage, ordered by sla_deadline ASC.
        Used by the human reviewer queue.

    CONNECTION POOL MANAGEMENT:
        Implements an asyncpg connection pool (initialised at app startup).
        Pool size and overflow from settings.py.
        All functions are async — called with await.

    IDEMPOTENCY:
        All write operations carry an idempotency key derived from:
        claim_id + operation_type + agent_id.
        Safe to retry on transient failures without duplicating data.

DEPENDENCIES:
    asyncpg, shared.schemas.claim_context, shared.schemas.agent_output,
    shared.config.settings
"""
