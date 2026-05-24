"""
agents/extraction/pre_intake_router.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Runs immediately after intake, before any other downstream agent.
    Makes two hard, immutable routing determinations that govern the
    entire rest of the pipeline.

WHAT GOES HERE:

    DETERMINATION 1 — CONTESTABILITY CHECK:
        Fetches policy inception date and last revival date from
        the internal policy system API.
        Calculates: days_elapsed = date_of_death - max(inception_date, revival_date)
        If days_elapsed ≤ CONTESTABILITY_WINDOW_DAYS (730):
            → sets contestability_status = IN_WINDOW
        Else:
            → sets contestability_status = OUT_OF_WINDOW

        ⚠️ HARD GATE: This status CANNOT be overridden by any downstream
        agent, confidence score, or LLM output. IN_WINDOW automatically
        triggers escalation via escalation_evaluator criterion #1.

    DETERMINATION 2 — CAUSE-OF-DEATH ROUTER:
        Reads cause_of_death from claimant statement form (extracted by field
        extractor if document already uploaded, or from form submission).
        Classifies into one of:
            NATURAL
            ACCIDENTAL
            MURDER
            SUICIDE_WITHIN_12M   (if date_of_death - policy_date ≤ 365 days)
            SUICIDE_AFTER_12M    (if > 365 days — coverage may apply)
            UNKNOWN              → BLOCKS pipeline, requests PMR before proceeding

        UNKNOWN handling: publishes a targeted document request to Communications
        queue asking specifically for the Postmortem Report. Pipeline halts
        at this point — no downstream agents run until cause is determined.

    OUTPUT:
        routing_flags dict containing:
            contestability_status, cause_of_death_type,
            mandatory_document_set (from MANDATORY_DOCUMENTS_BY_CAUSE),
            fraud_patterns_to_activate (list of rule IDs relevant to this cause)

        routing_flags is injected into every subsequent agent's LangGraph state.

DEPENDENCIES:
    shared.schemas.claim_context, shared.schemas.document_types,
    shared.config.settings, shared.events.kafka_client
"""
