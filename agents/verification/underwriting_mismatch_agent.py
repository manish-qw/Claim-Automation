"""
agents/verification/underwriting_mismatch_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Checks for material misrepresentation in the original insurance application.
    ONLY activates for claims where contestability_status = IN_WINDOW
    (death within 730 days of policy inception or revival).

WHAT GOES HERE:

    ACTIVATION CONDITION:
        Only runs if routing_flags.contestability_status == IN_WINDOW.
        If OUT_OF_WINDOW: this module returns None immediately and exits.
        This is enforced by external_verification_agent.py before calling.

    INPUT FETCH — Underwriting Application (from internal policy system API):
        All health conditions declared at underwriting:
            declared_diseases, declared_surgeries, declared_hospitalisations
        Lifestyle factors declared:
            smoking (bool + amount), alcohol (bool + amount)
        Occupation and hazard level declared
        Family medical history declared

    ACTUAL MEDICAL EVIDENCE (from claim documents):
        cause_of_death from DEATH_CERTIFICATE
        primary_diagnosis from DISCHARGE_SUMMARY
        findings from POSTMORTEM_REPORT
        chronic conditions mentioned in INDOOR_CASE_PAPERS

    LLM CALL — Claude Opus with STRICT binary output:
        System prompt:
            "You are a medical advisor reviewing an insurance claim.
             Based ONLY on the medical information provided:
             (1) the health declarations at underwriting, and
             (2) the actual medical findings at death —
             determine: could the cause of death plausibly arise from a
             pre-existing condition that a reasonable person would have been
             expected to declare at underwriting?
             Answer ONLY one of:
                POSSIBLE_MATERIAL_MISREPRESENTATION
                CONSISTENT
             Then provide your reasoning citing ONLY the specific medical
             information provided. Do not use external medical knowledge."

    UnderwritingResult (dataclass):
        assessment          — "POSSIBLE_MATERIAL_MISREPRESENTATION" | "CONSISTENT"
        declared_conditions — list of what was declared at underwriting
        actual_conditions   — list of conditions found in death documents
        llm_reasoning       — the model's cited reasoning
        confidence          — float 0–1
        produced_at         — UTC timestamp

    ⚠️ IMPORTANT — THIS MODULE NEVER AUTO-DENIES:
        A POSSIBLE_MATERIAL_MISREPRESENTATION result feeds ONLY into the
        EscalationPackage for human review.
        A human medical advisor must evaluate all contested cases.
        This agent produces a signal, not a decision.

DEPENDENCIES:
    httpx (policy system API), shared.llm.llm_client (LLMModel.OPUS),
    shared.schemas.claim_context, shared.audit.audit_service,
    shared.config.settings
"""
