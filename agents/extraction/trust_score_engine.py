"""
agents/extraction/trust_score_engine.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Computes a quantitative trust score for each document and an aggregate
    trust score for the overall claim. These scores propagate downstream —
    the orchestrator injects low-trust warnings into all downstream agent
    prompts when trust is below threshold.

WHAT GOES HERE:

    PER-DOCUMENT TRUST SCORE FORMULA:
        trust_score = (
            ocr_confidence              × 0.25 +
            classification_confidence   × 0.15 +
            avg_field_extraction_conf   × 0.30 +
            (1 - tamper_score)          × 0.20 +
            consistency_score           × 0.10
        )
        Where:
            ocr_confidence           — from OcrResult.overall_confidence
            classification_confidence — from ClassificationResult.confidence
            avg_field_extraction_conf — mean confidence across all extracted fields
            tamper_score             — from document_forensics output (0=clean, 1=tampered)
            consistency_score        — 1.0 if no contradictions, reduced per severity:
                                       CRITICAL: -0.40, HIGH: -0.20, MEDIUM: -0.10

    AGGREGATE CLAIM TRUST SCORE (weighted average across all documents):
        Document type weights:
            DEATH_CERTIFICATE           → 0.30
            POSTMORTEM_REPORT           → 0.25
            FIR                         → 0.20
            All other document types    → 0.25 (split equally among them)

        aggregate_trust = weighted_average(per_document_trust_scores)

    TrustScoreResult (dataclass):
        document_scores    — dict[document_id → float]
        aggregate_score    — float 0–1
        low_trust_fields   — list of specific field names with confidence < 0.65
        warning_injections — list of warning strings to inject into downstream
                             agent prompts (auto-generated when aggregate < 0.65)

    LOW TRUST HANDLING:
        If aggregate_score < OCR_CONFIDENCE_MINIMUM (0.65):
        Generate warning message:
            "NOTE: Document trust score for this claim is [score].
             The following fields have low confidence: [list].
             Treat these as uncertain rather than ground truth."
        This warning string is stored in TrustScoreResult.warning_injections
        and the orchestrator injects it into every downstream agent's prompt.

DEPENDENCIES:
    agents.extraction.ocr_engine (OcrResult),
    agents.extraction.document_classifier (ClassificationResult),
    agents.extraction.document_forensics (ForensicsResult),
    agents.extraction.cross_document_checker (ContradictionReport),
    shared.config.settings
"""
