"""
agents/extraction/ocr_engine.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    OCR abstraction layer with a three-tier fallback chain.
    Provides text extraction from uploaded claim documents regardless of
    which OCR provider is available.

WHAT GOES HERE:

    PRE-PROCESSING (runs before any OCR engine):
        deskew()       — correct rotated/skewed document scans
        denoise()      — remove scan noise using Pillow filters
        enhance_contrast() — improve legibility for faded documents
        detect_and_mask_stamps() — identify ink stamps, extract separately
                                   so stamp text does not corrupt form fields

    OcrResult (dataclass):
        raw_text           — full extracted text string
        per_word_confidence — list[{word, confidence, bounding_box}]
        layout             — spatial bounding box map of all text blocks
        overall_confidence — float 0–1 (weighted average of word confidences)
        fallback_level_used — 0 (Azure) | 1 (Textract) | 2 (PaddleOCR)
        low_quality_ocr_flag — bool, set True if PaddleOCR was used

    TIER 1 — Azure Document Intelligence (primary):
        Uses Indian prebuilt models trained on:
            Aadhaar cards, PAN cards, Indian government death certificates,
            FIRs, multi-column insurance forms, stamp-overlaid documents.
        Returns per-word confidence and spatial bounding boxes.
        Activate Tier 2 if: overall_confidence < OCR_CONFIDENCE_MINIMUM
        or if Azure throws any exception.

    TIER 2 — AWS Textract (first fallback):
        General-purpose OCR, good on printed documents.
        Weaker on handwritten Devanagari and multi-language documents.
        Activate Tier 3 if: Textract returns confidence < threshold or errors.

    TIER 3 — PaddleOCR self-hosted (last resort):
        Lowest accuracy on handwritten Indian scripts and stamped documents.
        Sets low_quality_ocr_flag = True on result.
        If activated: downstream agents are notified to treat extraction
        results with lower confidence.

    TOTAL FAILURE HANDLING:
        If all three OCR engines fail on a document:
        Mark document status as OCR_FAILED.
        Raise a HIGH severity flag for manual review.
        Return a null OcrResult. Continue with other documents.
        Never raise an exception that blocks the pipeline.

DEPENDENCIES:
    azure-ai-formrecognizer, boto3 (Textract), paddleocr,
    Pillow, numpy, shared.config.settings, shared.audit.audit_service
"""
