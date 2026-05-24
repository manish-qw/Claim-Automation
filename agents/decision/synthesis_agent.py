"""
agents/decision/synthesis_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Aggregates all upstream agent outputs into a single master summary.
    Does NOT make decisions — only aggregates, quantifies uncertainty,
    and surfaces conflicts/gaps for the decision layer.

WHAT GOES HERE:

    INPUTS COLLECTED (from claim context):
        All AgentOutput envelopes (from agent_outputs dict)
        FraudReport (from fraud_intelligence_agent)
        PolicyAssessment (from policy_rag_agent)
        ContradictionReport (from cross_document_checker)
        VerificationReport (from external_verification_agent)

    COMPOSITE UNCERTAINTY SCORE FORMULA:
        uncertainty = (
            (1 - avg_document_confidence)        × 0.20  +
            fraud_risk_score                      × 0.30  +
            (unverified_field_count / total_fields) × 0.20  +
            (ambiguous_clause_count / total_clauses) × 0.15 +
            (contradiction_count / total_checks)   × 0.15
        )
        float 0–1. Written to ClaimContextObject.uncertainty_score.

    MASTER SUMMARY (Claude Sonnet — summarisation, not reasoning):
        Plain English summary that both Decision Agent and human reviewer
        consume. Must explicitly surface:

        Section 1 — CONFLICTED FIELDS:
            List every field where conflict_resolution.py marked CONFLICTED.
            Name the conflict: "deceased_name differs between DEATH_CERTIFICATE
            ('Ramesh Kumar') and FIR ('Suresh Kumar') — CRITICAL"

        Section 2 — UNVERIFIED FIELDS:
            List every field marked UNVERIFIED with its reason code.
            "CRS registration: UNVERIFIED (DigiLocker API timeout)"

        Section 3 — FLAGS:
            All CRITICAL and HIGH FlagObjects with their evidence.

        Section 4 — DEGRADED AGENTS:
            Any agent that ran with status = DEGRADED (some tools failed).
            "OCR Engine: DEGRADED (Azure failed, fell back to PaddleOCR)"

        Section 5 — UNCERTAINTY SCORE:
            The composite score with breakdown by component.

    OUTPUT:
        Write uncertainty_score to ClaimContextObject.
        Write master_summary to ClaimContextObject.
        Publish SYNTHESIS_COMPLETE to Kafka.

DEPENDENCIES:
    shared.llm.llm_client (LLMModel.SONNET), shared.events.kafka_client,
    shared.db.claim_repository, shared.schemas.agent_output,
    orchestration.conflict_resolution (ConflictReport)
"""
