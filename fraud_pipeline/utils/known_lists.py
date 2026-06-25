"""
fraud_pipeline/utils/known_lists.py
=====================================
DEPRECATED — This file has been replaced by:
  - fraud_pipeline/utils/format_constants.py  (static regulatory format rules)
  - fraud_pipeline/utils/runtime_queries.py   (live PostgreSQL frequency queries)

All hardcoded blacklists have been removed. See the Agent 1 rewrite specification
(Section 1) for the rationale.

This shim is kept to avoid ImportError in any code that was not yet updated.
All sets are now empty; frequency checks happen at runtime via PostgreSQL.
"""

# These are now empty — real checks use runtime_queries.py
KNOWN_SUSPICIOUS_HOSPITALS: set = set()
KNOWN_SUSPICIOUS_DOCTORS: set = set()
KNOWN_FAKE_FIR_STATIONS: set = set()
KNOWN_REPEATED_NOMINEES: set = set()
KNOWN_REPEATED_BANK_ACCOUNTS: set = set()
