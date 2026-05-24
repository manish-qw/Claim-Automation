"""
tests/unit/extraction/test_trust_score_engine.py
────────────────────────────────────────────────────────────────
TESTS TO WRITE HERE:

    test_per_document_score_formula()
        Input: ocr_confidence=0.90, classification_confidence=0.85,
               avg_field_confidence=0.88, tamper_score=0.0,
               consistency_score=1.0.
        Assert: trust_score = (0.90×0.25) + (0.85×0.15) + (0.88×0.30)
                             + (1.0×0.20) + (1.0×0.10) ≈ 0.899

    test_tampered_document_lowers_score()
        Input: tamper_score=0.8 (high tampering evidence)
        Assert: (1 - 0.8) × 0.20 = 0.04 tamper component
        Assert: overall trust reduced significantly.

    test_aggregate_death_cert_weighted_highest()
        Input: two documents — DEATH_CERTIFICATE (trust=0.9) + FIR (trust=0.5)
        Assert: aggregate is weighted closer to DEATH_CERTIFICATE score.
        Expected: (0.9×0.30 + 0.5×0.20) / (0.30+0.20) = (0.27+0.10)/0.50 = 0.74

    test_low_trust_generates_warning_injection()
        Input: aggregate_trust = 0.55 (below 0.65 threshold)
        Assert: warning_injections list is non-empty.
        Assert: warning text contains "trust score" and lists low-confidence fields.

    test_high_trust_no_warning()
        Input: aggregate_trust = 0.90
        Assert: warning_injections = []
"""
