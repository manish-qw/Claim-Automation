"""
shared/schemas/document_types.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    All document-type definitions, mandatory document matrices, field schemas,
    and legal heir document options. Imported by all agents — especially the
    Document Intelligence pipeline and Missing Doc Tracker.

WHAT GOES HERE:

    DocumentType (enum — 22 values):
        DEATH_CERTIFICATE, FIR, POSTMORTEM_REPORT, VISCERA_REPORT,
        DISCHARGE_SUMMARY, INDOOR_CASE_PAPERS, DIAGNOSIS_REPORT,
        TREATING_DOCTOR_CERTIFICATE, HOSPITAL_ATTENDANT_CERTIFICATE,
        EMPLOYER_CERTIFICATE, PAN_CARD, AADHAAR_CARD, BANK_PROOF,
        CLAIMANT_STATEMENT_FORM, SUCCESSION_CERTIFICATE, WILL,
        COURT_ORDER, INDEMNITY_BOND, FAMILY_TREE, NOC,
        NEWSPAPER_CUTTING, UNKNOWN

    MANDATORY_DOCUMENTS_BY_CAUSE (dict: CauseOfDeathType → document lists):
        Maps each cause of death to its mandatory and conditional document set.
        NATURAL death mandatory:
            DEATH_CERTIFICATE, MEDICO_LEGAL_CERT, PAST_MEDICAL_RECORDS,
            DISCHARGE_SUMMARY, TREATING_DOCTOR_CERT, CLAIMANT_STATEMENT_FORM,
            PAN_OR_FORM97, BANK_PROOF, PAYOUT_MANDATE
        Unnatural death additionally requires:
            FIR, INQUEST_PANCHNAMA, FINAL_POLICE_REPORT,
            POSTMORTEM_REPORT, VISCERA_REPORT

    FIELD_SCHEMA_BY_DOCUMENT_TYPE (dict: DocumentType → field definitions):
        Maps every document type to its expected extracted fields.
        Each field definition includes:
            field_name, python_type, is_mandatory (bool), description
        Used by FieldExtractor to know what to look for and by
        MissingDocTracker to identify incomplete extractions.

    LEGAL_HEIR_DOCUMENT_OPTIONS (list of 4 option sets):
        The four alternative document packages acceptable when the
        named nominee is deceased. Options A through D, each with
        their specific required documents. Includes Maharashtra-specific
        stamp duty difference (₹100 vs ₹500 for other states).

    OMBUDSMAN_BY_STATE (dict: state_name → OmbudsmanOffice):
        Maps all Indian states to their IRDAI Insurance Ombudsman
        regional office details (address, phone, email, jurisdiction).
        Used by Communications Agent for mandatory denial letters.

DEPENDENCIES:
    enum, dataclasses — no external libraries
"""
