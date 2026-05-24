"""
agents/extraction/cross_document_checker.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Runs after ALL documents in a claim have been processed. Compares fields
    across documents to detect contradictions. Produces a ContradictionReport.
    Entirely deterministic — no LLM (except one specific call noted below).

WHAT GOES HERE:

    ContradictionObject (dataclass):
        field_name    — the field being compared (e.g. "deceased_name")
        doc_a         — document_id of first source
        value_a       — value from doc_a
        doc_b         — document_id of second source
        value_b       — value from doc_b
        severity      — FlagSeverity: CRITICAL | HIGH | MEDIUM | LOW | INFO
        description   — plain English explanation of the discrepancy

    ContradictionReport (dataclass):
        claim_id         — UUID
        contradictions   — list[ContradictionObject]
        checked_at       — UTC timestamp

    COMPARISON METHODS (all deterministic):

    Name Matching — Levenshtein distance:
        Threshold: 85% similarity.
        "Ramesh Kumar" vs "Ramesh K." → soft match, flag severity INFO
        "Ramesh Kumar" vs "Suresh Kumar" → hard mismatch, flag CRITICAL
        Implementation: python-Levenshtein library, normalised ratio.

    Date Matching — Python datetime comparison:
        Parse all date strings to datetime.date objects before comparing.
        Tolerance window: 24 hours (for transcription errors in timestamps).
        Never use string comparison for dates. Never use LLM.

    Identifier Matching — exact string:
        CRS registration number, FIR number, vehicle registration, PAN number.
        Zero tolerance: any mismatch is CRITICAL.

    ⚠️ ONE LLM CALL — Cause-of-Death Semantic Consistency:
        Cause of death appears in: DEATH_CERTIFICATE, POSTMORTEM_REPORT,
        TREATING_DOCTOR_CERTIFICATE.
        The exact wording differs but meaning may be consistent
        (e.g. "cardiac arrest" vs "heart failure due to IHD").
        A single Claude Sonnet call with STRICT binary output prompt:
            "Given these cause-of-death statements from separate documents,
             are they medically consistent? Answer only: CONSISTENT or
             INCONSISTENT. If INCONSISTENT, list which specific field differs
             and how."
        No reasoning beyond binary output is accepted.

    FIELDS CHECKED ACROSS DOCUMENTS:
        deceased_name       — death cert vs FIR vs discharge summary vs PMR
        date_of_death       — death cert vs FIR incident date vs PMR date
        place_of_death      — death cert vs FIR vs hospital records
        cause_of_death      — LLM semantic consistency check (see above)
        crs_registration_number — death cert vs CRS verification result
        doctor_name         — PMR vs treating doctor certificate
        hospital_name       — discharge summary vs PMR vs hospital certificate

DEPENDENCIES:
    Levenshtein, datetime, shared.llm.llm_client (for one semantic check),
    shared.schemas.document_types, shared.schemas.agent_output (FlagObject)
"""
