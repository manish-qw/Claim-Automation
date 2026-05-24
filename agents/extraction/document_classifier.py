"""
agents/extraction/document_classifier.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Takes the OCR output of a document and classifies it into one of the
    22 DocumentType enum values defined in shared/schemas/document_types.py.

WHAT GOES HERE:

    CLASSIFICATION MODEL:
        Uses LayoutLMv3 — a layout-aware multimodal transformer that considers
        BOTH the text content AND the spatial position of text on the page.
        Key advantage: a form with "Date of Death" at position (x, y) is
        classified differently from the same words in a narrative letter.
        Model loaded from Hugging Face on startup, cached in memory.

    ClassificationResult (dataclass):
        document_type   — DocumentType enum value
        confidence      — float 0–1
        alternative     — second-best classification + its confidence
                          (useful for borderline cases)

    MIXED-MODALITY HANDLING:
        Indian government documents often have both printed templates
        (fixed structure) and handwritten variable content.
        Classifier uses the PRINTED template structure for classification —
        not the handwritten variable fields.
        E.g., a death certificate is classified by its printed header and
        layout, not by the handwritten name of the deceased.

    UNKNOWN CLASSIFICATION:
        If confidence < classification threshold OR no class is clear:
        → Set document_type = DocumentType.UNKNOWN
        → DO NOT guess.
        → Raise a DOCUMENT_UNCLASSIFIED flag (MEDIUM severity).
        → Construct a targeted re-upload request for the claimant:
          "Please re-upload [document] and specify its type using the
           document type selector."
        → Mark document as PENDING_RECLASSIFICATION.
        → Continue pipeline with remaining documents.

    SUPPORTED DOCUMENT TYPES (all 22):
        DEATH_CERTIFICATE, FIR, POSTMORTEM_REPORT, VISCERA_REPORT,
        DISCHARGE_SUMMARY, INDOOR_CASE_PAPERS, DIAGNOSIS_REPORT,
        TREATING_DOCTOR_CERTIFICATE, HOSPITAL_ATTENDANT_CERTIFICATE,
        EMPLOYER_CERTIFICATE, PAN_CARD, AADHAAR_CARD, BANK_PROOF,
        CLAIMANT_STATEMENT_FORM, SUCCESSION_CERTIFICATE, WILL,
        COURT_ORDER, INDEMNITY_BOND, FAMILY_TREE, NOC,
        NEWSPAPER_CUTTING, UNKNOWN

DEPENDENCIES:
    transformers (LayoutLMv3), torch, shared.schemas.document_types,
    agents.extraction.ocr_engine (OcrResult)
"""
