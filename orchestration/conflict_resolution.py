"""
orchestration/conflict_resolution.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Deterministic conflict resolution engine. Called by the orchestrator
    after all parallel agents complete, before synthesis. Not an LLM —
    a pure rule engine with a defined priority hierarchy.

WHAT GOES HERE:

    ConflictReport (dataclass):
        resolved_fields   — dict[field_name → ResolvedField]
        conflicted_fields — list[ConflictedField]  (could not resolve)
        resolved_at       — UTC timestamp

    ResolvedField (dataclass):
        field_name        — string
        winning_value     — the resolved value
        winning_source    — which agent/document provided it
        losing_sources    — list of sources that were overridden
        resolution_rule   — which priority rule applied

    ConflictedField (dataclass):
        field_name        — string
        sources           — list[{source, value, confidence}]
        reason            — why conflict could not be resolved
        requires_human    — bool (True if this conflict is material to decision)

    PRIORITY HIERARCHY (applied in this exact order):

    Rule 1 — Policy questions:
        Policy RAG Agent output wins over any other agent on coverage,
        exclusions, clause interpretation, and payout amounts.

    Rule 2 — External registry facts:
        External Verification Agent (government APIs) wins over
        Document Intelligence Agent on:
            CRS registration validity, Aadhaar status, hospital NHA ID.

    Rule 3 — Document-internal facts:
        Document Intelligence Agent wins on what a specific document says
        (field values extracted from a particular document).

    Rule 4 — Timeline facts:
        When two sources contradict on a date/timeline:
        The source with higher extraction confidence wins.

    Rule 5 — Fraud risk:
        ESCALATES, never averages down.
        If any agent raises CRITICAL fraud risk, final risk = CRITICAL.
        Highest severity from any source wins. Risk is never averaged.

    CONFLICT ESCALATION:
        When two sources have equal confidence AND contradictory values:
            → Mark field as CONFLICTED
            → Add to conflicted_fields list
            → Set requires_human = True IF the field is material to:
                (a) coverage determination, OR
                (b) fraud assessment
        If any CONFLICTED field has requires_human = True:
            → Mandatory human escalation regardless of all other scores
              (this is escalation_evaluator criterion — enforced here as well)

DEPENDENCIES:
    shared.schemas.agent_output, shared.schemas.claim_context,
    shared.audit.audit_service
"""
