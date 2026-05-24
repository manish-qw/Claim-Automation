"""
agents/extraction/document_forensics.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Tampering detection for every uploaded document. Runs four independent
    checks and produces a composite tamper_score with evidence indicators.

WHAT GOES HERE:

    ForensicsResult (dataclass):
        tamper_score       — float 0–1 (0 = clean, 1 = strong evidence of tampering)
        tamper_indicators  — list[TamperIndicator]
        signature_status   — enum: SIGNATURE_VALID | SIGNATURE_INVALID | SIGNATURE_ABSENT

    TamperIndicator (dataclass):
        indicator_type  — string: "PDF_METADATA" | "FONT_INCONSISTENCY" |
                          "ELA_ANOMALY" | "SIGNATURE_INVALID"
        location        — bounding box coordinates where anomaly was found
                          (None for metadata-level checks)
        description     — plain English explanation of what was detected
        severity        — float 0–1, contribution to overall tamper_score

    CHECK 1 — PDF Metadata Analysis:
        Extract: creation_date, modification_date, creator_application.
        Flag if: creation_date > stated document date (e.g., PDF created in
        2025 for a death certificate dated 2023).
        Flag if: creator_application is a consumer photo editor (Photoshop,
        GIMP, Canva) rather than a government document system.
        Uses: PyMuPDF (fitz) or pdfminer.

    CHECK 2 — Font Consistency Analysis:
        Authentic single-source documents have uniform font embedding.
        Extract all embedded fonts from the PDF.
        Flag if: multiple unrelated font families present.
        Flag if: encoding inconsistencies suggesting content pasted from
        different source documents.
        Uses: pdfminer.six font extraction.

    CHECK 3 — Error Level Analysis (ELA) for images (JPEG/PNG):
        Save image at known compression quality (e.g., JPEG 95%).
        Compute pixel-level difference between original and re-saved.
        Regions with different compression history (previously edited and
        re-saved) appear as bright anomalies in the ELA output.
        tamper_regions → list of bounding boxes with bright ELA anomalies.
        Implementation: Pillow + NumPy.
        Skip this check if document is a text-only PDF.

    CHECK 4 — Digital Signature Validation (government PDFs):
        DigiLocker-issued documents carry a verifiable digital signature.
        Attempt to validate signature against DigiLocker CA public key.
        Output states:
            SIGNATURE_VALID   — document authentic, no further checks needed
            SIGNATURE_INVALID — signature present but verification failed
            SIGNATURE_ABSENT  — no digital signature (expected for physical scans)

    TAMPER SCORE AGGREGATION:
        Weighted combination of indicator severities.
        SIGNATURE_INVALID alone sets tamper_score to minimum 0.80.
        ELA anomaly covering > 20% of document area sets minimum 0.70.

DEPENDENCIES:
    PyMuPDF (fitz), pdfminer.six, Pillow, numpy,
    httpx (for DigiLocker signature validation endpoint),
    shared.config.settings
"""
