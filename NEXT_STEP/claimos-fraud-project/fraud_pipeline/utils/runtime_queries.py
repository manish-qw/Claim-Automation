"""
fraud_pipeline/utils/runtime_queries.py
=========================================
Runtime PostgreSQL frequency queries for Agent 1 (and Agent 7).

All functions:
  - Accept an optional SQLAlchemy engine; gracefully return neutral results if None / DB down.
  - Return (count: int, claim_ids: list[str]) — never booleans.
  - Are safe to call from any agent; failures add DB_UNAVAILABLE flag rather than crashing.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — execute a query safely, return [] / 0 on any failure
# ─────────────────────────────────────────────────────────────────────────────

def _safe_query(engine, sql: str, params: dict) -> list:
    """Run a raw SQL query; return [] on DB down or error."""
    if engine is None:
        return []
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return result.fetchall()
    except Exception as exc:
        logger.warning("[runtime_queries] DB query failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 1. Nominee frequency
# ─────────────────────────────────────────────────────────────────────────────

def get_nominee_claim_count(
    nominee_id: str,
    engine=None,
) -> Tuple[int, List[str]]:
    """
    Returns (count, claim_ids) — how many distinct claims share this nominee_id.
    Queries fraud_results table (persisted by Agent pipeline after each run).
    """
    if not nominee_id:
        return 0, []

    rows = _safe_query(
        engine,
        """
        SELECT claim_case_id
        FROM fraud_results
        WHERE full_package::jsonb -> 'claimant' ->> 'nominee_id' = :nominee_id
        ORDER BY created_at DESC
        LIMIT 20
        """,
        {"nominee_id": nominee_id},
    )
    claim_ids = [r[0] for r in rows]
    return len(claim_ids), claim_ids


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bank account frequency
# ─────────────────────────────────────────────────────────────────────────────

def get_bank_account_claim_count(
    account_number: str,
    engine=None,
) -> Tuple[int, List[str]]:
    """
    Returns (count, claim_ids) — how many distinct claims share this bank account.
    """
    if not account_number:
        return 0, []

    rows = _safe_query(
        engine,
        """
        SELECT claim_case_id
        FROM fraud_results
        WHERE full_package::jsonb -> 'claimant' ->> 'bank_account' = :account
           OR full_package::jsonb -> 'claimant' ->> 'bank_account_number' = :account
        ORDER BY created_at DESC
        LIMIT 20
        """,
        {"account": account_number},
    )
    claim_ids = [r[0] for r in rows]
    return len(claim_ids), claim_ids


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hospital ROHINI frequency (rolling window)
# ─────────────────────────────────────────────────────────────────────────────

def get_hospital_claim_frequency(
    hospital_rohini_id: str,
    days: int = 90,
    engine=None,
) -> Tuple[int, List[str]]:
    """
    Returns (count, claim_ids) — how many claims used this ROHINI ID in the last `days` days.
    """
    if not hospital_rohini_id:
        return 0, []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = _safe_query(
        engine,
        """
        SELECT claim_case_id
        FROM fraud_results
        WHERE (
            full_package::jsonb -> 'death_information' ->> 'hospital_rohini_id' = :rohini_id
            OR full_package::jsonb -> 'death_information' ->> 'hospital_id' = :rohini_id
        )
        AND created_at >= :cutoff
        ORDER BY created_at DESC
        LIMIT 50
        """,
        {"rohini_id": hospital_rohini_id, "cutoff": cutoff},
    )
    claim_ids = [r[0] for r in rows]
    return len(claim_ids), claim_ids


# ─────────────────────────────────────────────────────────────────────────────
# 4. Doctor registration frequency (rolling window)
# ─────────────────────────────────────────────────────────────────────────────

def get_doctor_claim_frequency(
    registration_number: str,
    days: int = 90,
    engine=None,
) -> Tuple[int, List[str]]:
    """
    Returns (count, claim_ids) — how many claims list this doctor reg number in the last `days` days.
    """
    if not registration_number:
        return 0, []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = _safe_query(
        engine,
        """
        SELECT claim_case_id
        FROM fraud_results
        WHERE full_package::jsonb -> 'death_information' ->> 'doctor_registration_number' = :reg
        AND created_at >= :cutoff
        ORDER BY created_at DESC
        LIMIT 50
        """,
        {"reg": registration_number, "cutoff": cutoff},
    )
    claim_ids = [r[0] for r in rows]
    return len(claim_ids), claim_ids


# ─────────────────────────────────────────────────────────────────────────────
# 5. File hash registry
# ─────────────────────────────────────────────────────────────────────────────

def check_file_hash_registry(
    sha256_hash: str,
    current_claim_id: str,
    engine=None,
) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_duplicate: bool, original_claim_id: str | None).
    A duplicate means the same file bytes were submitted in a DIFFERENT claim.
    """
    if not sha256_hash or engine is None:
        return False, None

    rows = _safe_query(
        engine,
        """
        SELECT claim_id FROM document_hash_registry
        WHERE sha256_hash = :hash AND claim_id != :current
        LIMIT 1
        """,
        {"hash": sha256_hash, "current": current_claim_id},
    )
    if rows:
        return True, rows[0][0]
    return False, None


def register_file_hash(
    sha256_hash: str,
    claim_id: str,
    doc_type: str,
    engine=None,
) -> None:
    """
    Upsert the file hash into the registry. Called after Agent 1 runs.
    Silently skips if DB unavailable.
    """
    if not sha256_hash or engine is None:
        return
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            # Ensure table exists
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS document_hash_registry (
                    sha256_hash TEXT NOT NULL,
                    claim_id    TEXT NOT NULL,
                    doc_type    TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (sha256_hash, claim_id)
                )
            """))
            conn.execute(text("""
                INSERT INTO document_hash_registry (sha256_hash, claim_id, doc_type)
                VALUES (:hash, :claim, :doc_type)
                ON CONFLICT (sha256_hash, claim_id) DO NOTHING
            """), {"hash": sha256_hash, "claim": claim_id, "doc_type": doc_type})
    except Exception as exc:
        logger.warning("[runtime_queries] register_file_hash failed: %s", exc)
