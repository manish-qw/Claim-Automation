"""
agents/extraction/field_extractor.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Takes the classified document type + OCR output and extracts a typed,
    structured set of fields specific to that document type.
    Uses Claude Sonnet with strict grounding — never infers or guesses.

WHAT GOES HERE:

    LLM STRATEGY:
        Model: Claude Sonnet (via shared.llm.llm_client)
        System prompt (invariant for all document types):
            "Extract only what is explicitly written in this document.
             If a field is blank, illegible, or absent, output FIELD_MISSING.
             Do NOT infer. Do NOT guess. Do NOT use surrounding context to
             fill in a missing value. Every extracted value must be verbatim
             from the document text."
        A document-type-specific user prompt is appended with the full
        field schema for that document type.

    FIELD SCHEMAS (implement extraction for each document type):

    DEATH_CERTIFICATE:
        deceased_name, dob, date_of_death, time_of_death, place_of_death,
        cause_of_death_primary, cause_of_death_secondary, manner_of_death,
        crs_registration_number, registration_date, issuing_authority,
        registrar_name, registrar_signature_present

    FIR:
        fir_number, police_station_name, district, state, date_filed,
        time_filed, date_of_incident, nature_of_incident,
        vehicle_registration_if_accident, parties_involved,
        investigating_officer_name, officer_badge_number

    POSTMORTEM_REPORT:
        pmr_reference_number, hospital_name, doctor_name,
        doctor_registration_number, date_of_pmr,
        cause_of_death_immediate, cause_of_death_antecedent,
        manner_of_death, injuries_described, toxicology_pending

    DISCHARGE_SUMMARY:
        hospital_name, admission_date, discharge_date, patient_name,
        treating_doctor_name, primary_diagnosis, secondary_diagnosis,
        procedures_performed, outcome

    BANK_PROOF:
        account_holder_name, account_number, ifsc_code, bank_name,
        branch_name, account_type

    (Implement all 22 document types per FIELD_SCHEMA_BY_DOCUMENT_TYPE)

    ExtractedField (dataclass — one per field):
        value      — the extracted value, or "FIELD_MISSING"
        confidence — float 0–1 (from LLM's self-assessed certainty)

    DATA TYPE RULES:
        Date fields  → parse to Python datetime.date objects
        Amount fields → use Python Decimal — NEVER float
        Boolean fields (e.g. registrar_signature_present) → parse to bool
        Identifier fields (CRS number, FIR number) → keep as string exactly

DEPENDENCIES:
    shared.llm.llm_client, shared.schemas.document_types,
    agents.extraction.ocr_engine (OcrResult),
    agents.extraction.document_classifier (ClassificationResult)
"""
