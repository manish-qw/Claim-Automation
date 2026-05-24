"""
shared/config/settings.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Central configuration store for every threshold, SLA, and external API
    parameter used across the entire CLAIMOS AI system.

WHAT GOES HERE:
    - A Pydantic `BaseSettings` class that reads all values from environment
      variables (with sane defaults).
    - Every numeric threshold used by agents: fraud risk thresholds, SLA
      windows, confidence minimums, payout caps — all defined here.
      NO magic numbers anywhere else in the codebase.
    - External API base URLs, timeouts, and retry counts for:
        UIDAI, DigiLocker, Surepass, NMC, NHA, VAHAN, Azure OCR,
        AWS Textract, Pinecone, Cohere, Banking settlement API.
    - A `get_settings()` function (cached with `lru_cache`) that returns the
      singleton settings instance — imported by all agents.
    - A configuration-change audit hook: any time a threshold is updated at
      runtime, an AuditEvent is automatically logged recording:
        changed_parameter, old_value, new_value, changed_by, timestamp.

KEY THRESHOLDS DEFINED HERE:
    FRAUD_AUTO_BLOCK_THRESHOLD       — default: "CRITICAL"
    AUTO_APPROVE_MAX_SUM_ASSURED     — default: 2_500_000  (₹25L in paise)
    CONTESTABILITY_WINDOW_DAYS       — default: 730
    SUICIDE_EXCLUSION_WINDOW_DAYS    — default: 365
    AGENT_CONFIDENCE_MINIMUM         — default: 0.50
    AGENT_CONFIDENCE_WARNING         — default: 0.65
    UNCERTAINTY_SCORE_ESCALATION_THRESHOLD — default: 0.65
    OCR_CONFIDENCE_MINIMUM           — default: 0.65
    MISSING_DOCUMENT_FOLLOWUP_SLA_DAYS     — default: 15
    IRDAI_ACKNOWLEDGEMENT_SLA_HOURS  — default: 24
    IRDAI_DECISION_SLA_DAYS          — default: 30
    HUMAN_REVIEW_RESPONSE_SLA_HOURS  — default: 4
    SECOND_REVIEWER_THRESHOLD        — default: 5_000_000  (₹50L in paise)

DEPENDENCIES:
    pydantic-settings, python-dotenv
"""
