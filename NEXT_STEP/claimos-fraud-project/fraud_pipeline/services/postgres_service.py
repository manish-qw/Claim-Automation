"""
PostgreSQL service — stores fraud scores, verification outputs, trust analysis.
Uses SQLAlchemy Core (sync) for simplicity.
Tables are auto-created on first connection.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    String, Float, Boolean, Text, DateTime, Integer,
    insert, select, text,
)

from fraud_pipeline.utils.config import settings
from fraud_pipeline.utils.logger import get_logger

logger = get_logger("PostgresService")

metadata = MetaData()

fraud_results_table = Table(
    "fraud_results",
    metadata,
    Column("id",                   Integer, primary_key=True, autoincrement=True),
    Column("claim_case_id",        String(64), index=True, nullable=False),
    Column("policy_number",        String(64), nullable=True),
    Column("fraud_risk_score",     Float,    nullable=True),
    Column("fraud_risk_level",     String(16), nullable=True),
    Column("final_recommendation", String(16), nullable=True),
    Column("escalation_required",  Boolean,  nullable=True),
    Column("overall_trust_score",  Float,    nullable=True),
    Column("full_package",         Text,     nullable=True),
    Column("created_at",           DateTime, default=lambda: datetime.now(timezone.utc)),
)

graph_entities_table = Table(
    "graph_entities",
    metadata,
    Column("claim_id",     String(100), nullable=False),
    Column("entity_type",  String(100), nullable=False),
    Column("entity_value", Text,        nullable=False),
    Column("created_at",   DateTime,    default=lambda: datetime.now(timezone.utc)),
)


class PostgresService:
    def __init__(self):
        self._available = False
        self._engine = None
        try:
            url = settings.POSTGRES_URL_SYNC
            logger.info(f"Connecting to PostgreSQL at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
            self._engine = create_engine(url, pool_pre_ping=True, echo=False)
            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            # Create tables if they don't exist
            metadata.create_all(self._engine)
            # Also ensure graph_entities has its PK constraint
            self._ensure_graph_entities_pk()
            self._available = True
            logger.info("PostgresService: connected and schema ready [OK]")
        except Exception as e:
            self._available = False
            logger.warning(f"PostgreSQL unavailable — results will NOT be persisted. Error: {e}")

    def _ensure_graph_entities_pk(self):
        """Ensure graph_entities primary key constraint exists (idempotent)."""
        try:
            with self._engine.begin() as conn:
                conn.execute(text("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint
                            WHERE conname = 'graph_entities_pkey'
                        ) THEN
                            ALTER TABLE graph_entities
                            ADD CONSTRAINT graph_entities_pkey
                            PRIMARY KEY (claim_id, entity_type, entity_value);
                        END IF;
                    EXCEPTION WHEN others THEN NULL;
                    END$$;
                """))
        except Exception:
            pass  # Non-critical

    @property
    def is_available(self) -> bool:
        return self._available

    def save_fraud_result(self, package: Dict[str, Any]) -> bool:
        """Persist the full analysis package to fraud_results table."""
        if not self._available:
            logger.warning("PostgreSQL not available — skipping save")
            return False
        try:
            fraud = package.get("fraud_analysis") or {}
            trust = package.get("trust_analysis") or {}
            with self._engine.begin() as conn:
                conn.execute(
                    insert(fraud_results_table).values(
                        claim_case_id        = str(package.get("claim_case_id", "")),
                        policy_number        = str(package.get("policy_number", "") or ""),
                        fraud_risk_score     = float(fraud.get("fraud_risk_score") or 0.0),
                        fraud_risk_level     = str(fraud.get("fraud_risk_level") or "UNKNOWN"),
                        final_recommendation = str(package.get("final_recommendation") or "REVIEW"),
                        escalation_required  = bool(package.get("escalation_required", False)),
                        overall_trust_score  = float(trust.get("overall_trust_score") or 0.0),
                        full_package         = json.dumps(package, default=str),
                        created_at           = datetime.now(timezone.utc),
                    )
                )
            logger.info(f"Saved fraud result for {package.get('claim_case_id')} -> PostgreSQL [OK]")
            return True
        except Exception as e:
            logger.error(f"Failed to save fraud result: {e}")
            return False

    def get_fraud_result(self, claim_case_id: str) -> Optional[Dict]:
        """Fetch the latest result for a given claim ID."""
        if not self._available:
            return None
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(fraud_results_table)
                    .where(fraud_results_table.c.claim_case_id == claim_case_id)
                    .order_by(fraud_results_table.c.id.desc())
                    .limit(1)
                ).fetchone()
                return dict(row._mapping) if row else None
        except Exception as e:
            logger.error(f"Failed to get fraud result: {e}")
            return None

    def get_all_results_summary(self) -> list:
        """Fetch all results (summary columns only — no full_package blob)."""
        if not self._available:
            return []
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(
                        fraud_results_table.c.id,
                        fraud_results_table.c.claim_case_id,
                        fraud_results_table.c.policy_number,
                        fraud_results_table.c.fraud_risk_score,
                        fraud_results_table.c.fraud_risk_level,
                        fraud_results_table.c.final_recommendation,
                        fraud_results_table.c.escalation_required,
                        fraud_results_table.c.overall_trust_score,
                        fraud_results_table.c.created_at,
                    ).order_by(fraud_results_table.c.id.desc())
                ).fetchall()
                return [dict(r._mapping) for r in rows]
        except Exception as e:
            logger.error(f"Failed to get all results: {e}")
            return []

    def save_graph_entities(self, claim_id: str, entities: list) -> None:
        """Persist graph entities for restart recovery."""
        if not self._available or not entities:
            return
        try:
            with self._engine.begin() as conn:
                for entity_type, entity_value in entities:
                    conn.execute(text("""
                        INSERT INTO graph_entities (claim_id, entity_type, entity_value)
                        VALUES (:claim_id, :entity_type, :entity_value)
                        ON CONFLICT DO NOTHING
                    """), {
                        "claim_id":     str(claim_id),
                        "entity_type":  str(entity_type),
                        "entity_value": str(entity_value),
                    })
        except Exception as e:
            logger.error(f"save_graph_entities failed: {e}")

    def get_all_graph_entities(self) -> list:
        """Fetch all persisted graph entities for graph warmup on startup."""
        if not self._available:
            return []
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT claim_id, entity_type, entity_value
                    FROM graph_entities
                    ORDER BY created_at ASC
                """)).fetchall()
                return [{"claim_id": r[0], "entity_type": r[1], "entity_value": r[2]} for r in rows]
        except Exception:
            return []
