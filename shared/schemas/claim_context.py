"""
shared/schemas/claim_context.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    The canonical, single source of truth for every claim in the system.
    Every agent reads from and writes back to this object — agents never
    pass data directly to each other. Lives in PostgreSQL.

WHAT GOES HERE:
    Pydantic v2 models for the full ClaimContextObject with these fields:

    IDENTIFIERS & ROUTING:
        claim_id              — UUID, system-generated at intake
        policy_id             — from claimant submission
        claimant_id           — linked to identity verification output
        intake_channel        — enum: WEB_FORM | EMAIL | DOCUMENT_UPLOAD | API
        cause_of_death_type   — enum: NATURAL | ACCIDENTAL | MURDER | SUICIDE | UNKNOWN
        contestability_status — enum: IN_WINDOW | OUT_OF_WINDOW | UNKNOWN
        policy_snapshot_version — exact policy version frozen at intake (date of death)
        current_stage         — enum: INTAKE | DOC_INTEL | VERIFICATION | FRAUD |
                                       POLICY | SYNTHESIS | DECISION | ESCALATED |
                                       SETTLED | REJECTED

    SLA:
        sla_deadline          — IRDAI 30-day clock, starts when all mandatory docs received
        created_at            — claim intake timestamp

    DOCUMENTS:
        documents             — dict[document_id → DocumentRecord]
        missing_documents     — list[MissingDocument]

    AGENT OUTPUTS:
        agent_outputs         — dict[agent_id → AgentOutput envelope]
        flags                 — list of all FlagObjects raised by any agent

    SCORES:
        uncertainty_score     — composite float, computed by Synthesis Agent
        escalation_reasons    — list of specific criteria that triggered escalation

    DECISION:
        final_decision        — APPROVE | PARTIAL_APPROVE | DENY | ESCALATED
        net_payout_amount     — integer in paise (never float), from Benefit Calculator
        audit_trail_id        — reference to AWS QLDB ledger entry chain

    NESTED MODELS ALSO DEFINED HERE:
        DocumentRecord        — doc_type, extracted_fields, trust_score, tampering_flags
        ExtractedField        — value, confidence (float 0–1)
        MissingDocument       — doc_type, is_mandatory, follow_up_sent_at
        ClaimStage (enum)
        IntakeChannel (enum)
        CauseOfDeathType (enum)
        ContestabilityStatus (enum)
        FinalDecision (enum)

DEPENDENCIES:
    pydantic v2, uuid, datetime
SHARED WITH:
    Every agent in the system — this is the most-imported file in the project.
"""
