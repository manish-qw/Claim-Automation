"""
Neo4j Graph Store — Phases 1 & 2
==================================
Phase 1: Pydantic graph schema (8 node types + ClaimGraph container)
Phase 2: Neo4jGraphStore — persist_claim_graph() writes all 8 nodes
         and 7 relationships atomically using MERGE (idempotent).

Node labels:
  Claim, Policy, Claimant, Nominee, BankAccount, Hospital, Doctor, DeathEvent

Relationships:
  (Claim)-[:FILED_BY]       ->(Claimant)
  (Claim)-[:UNDER_POLICY]   ->(Policy)
  (Claim)-[:NAMES_NOMINEE]  ->(Nominee)
  (Claim)-[:PAYS_TO]        ->(BankAccount)
  (Claim)-[:HAS_DEATH_EVENT]->(DeathEvent)
  (DeathEvent)-[:OCCURRED_AT]  ->(Hospital)
  (Hospital)  -[:ATTENDED_BY]  ->(Doctor)

Design rules:
  - MERGE everywhere — safe to call multiple times for same claim
  - Every external call in try/except — pipeline never blocked
  - Falls back silently if Neo4j is not connected
  - Extracts all data directly from ClaimState (no hardcoded field lists)
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — PYDANTIC GRAPH SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class ClaimNode(BaseModel):
    """(:Claim) — Central node. One per claim."""
    id: str                                  # claim_case_id — unique key
    policy_number: str = ""
    policy_age_days: int = 0
    policy_sum_assured: float = 0.0
    policy_revival_detected: bool = False
    final_recommendation: str = ""
    escalation_required: bool = False
    fraud_risk_score: float = 0.0
    fraud_risk_level: str = ""

    class Config:
        extra = "allow"


class PolicyNode(BaseModel):
    """(:Policy) — Insurance policy details."""
    id: str                                  # policy_number — unique key
    policy_number: str = ""
    issue_date: str = ""
    sum_assured: float = 0.0
    premium: float = 0.0
    revival_detected: bool = False
    revival_date: str = ""

    class Config:
        extra = "allow"


class ClaimantNode(BaseModel):
    """(:Claimant) — Person who filed the claim."""
    id: str                                  # claimant name (normalised)
    name: str = ""
    relationship: str = ""
    contact: str = ""
    address: str = ""

    class Config:
        extra = "allow"


class NomineeNode(BaseModel):
    """(:Nominee) — Named beneficiary. Key fraud entity."""
    id: str                                  # nominee_id — unique key
    name: str = ""
    claimant_name: str = ""

    class Config:
        extra = "allow"


class BankAccountNode(BaseModel):
    """(:BankAccount) — Payout destination. Key fraud entity."""
    id: str                                  # bank_account — unique key
    bank_name: str = ""
    bank_ifsc: str = ""
    account_holder: str = ""

    class Config:
        extra = "allow"


class HospitalNode(BaseModel):
    """(:Hospital) — Where death occurred or treatment was given."""
    id: str                                  # hospital_name normalised
    name: str = ""
    address: str = ""
    hospital_id: str = ""

    class Config:
        extra = "allow"


class DoctorNode(BaseModel):
    """(:Doctor) — Attending or treating physician."""
    id: str                                  # doctor_name normalised
    name: str = ""

    class Config:
        extra = "allow"


class DeathEventNode(BaseModel):
    """(:DeathEvent) — Structured death metadata."""
    id: str                                  # claim_case_id + ':death'
    date_of_death: str = ""
    cause_of_death: str = ""
    manner_of_death: str = ""
    place_of_death: str = ""

    class Config:
        extra = "allow"


class ClaimGraph(BaseModel):
    """
    Container holding all nodes for one claim.
    Passed to Neo4jGraphStore.persist_claim_graph().
    """
    claim:        ClaimNode
    policy:       Optional[PolicyNode]      = None
    claimant:     Optional[ClaimantNode]    = None
    nominee:      Optional[NomineeNode]     = None
    bank_account: Optional[BankAccountNode] = None
    hospital:     Optional[HospitalNode]    = None
    doctor:       Optional[DoctorNode]      = None
    death_event:  Optional[DeathEventNode]  = None


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — NEO4J GRAPH STORE
# ═══════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """Normalise a string for use as a node ID."""
    return s.strip().lower().replace(" ", "_") if s else ""


def _extract_claim_graph(state) -> ClaimGraph:
    """
    Build a ClaimGraph from any ClaimState object.
    Handles both Pydantic models and plain dicts gracefully.
    All fields default to empty — never raises.
    """
    # Support both Pydantic model and dict
    if hasattr(state, "model_dump"):
        d = state.model_dump()
    elif isinstance(state, dict):
        d = state
    else:
        d = vars(state)

    claim_id      = d.get("claim_case_id", "")
    claimant_d    = d.get("claimant") or {}
    life_assured_d = d.get("life_assured") or {}
    death_d       = d.get("death_information") or {}
    fraud_d       = d.get("fraud_analysis") or {}

    # ── ClaimNode ──────────────────────────────────────────────────────────────
    claim = ClaimNode(
        id                   = claim_id,
        policy_number        = d.get("policy_number", ""),
        policy_age_days      = d.get("policy_age_days", 0),
        policy_sum_assured   = d.get("policy_sum_assured", 0.0)
                               or life_assured_d.get("sum_assured", 0.0),
        policy_revival_detected = d.get("policy_revival_detected", False),
        final_recommendation = d.get("final_recommendation", ""),
        escalation_required  = d.get("escalation_required", False),
        fraud_risk_score     = fraud_d.get("fraud_risk_score", 0.0),
        fraud_risk_level     = fraud_d.get("fraud_risk_level", ""),
    )

    # ── PolicyNode ─────────────────────────────────────────────────────────────
    policy_number = d.get("policy_number", "")
    policy = PolicyNode(
        id               = _norm(policy_number) or f"policy:{claim_id}",
        policy_number    = policy_number,
        issue_date       = d.get("policy_issue_date", ""),
        sum_assured      = d.get("policy_sum_assured", 0.0)
                          or life_assured_d.get("sum_assured", 0.0),
        premium          = d.get("policy_premium", 0.0),
        revival_detected = d.get("policy_revival_detected", False),
        revival_date     = d.get("policy_revival_date", ""),
    ) if policy_number else None

    # ── ClaimantNode ───────────────────────────────────────────────────────────
    claimant_name = (
        claimant_d.get("name", "") if isinstance(claimant_d, dict)
        else getattr(claimant_d, "name", "")
    )
    claimant = ClaimantNode(
        id           = _norm(claimant_name) or f"claimant:{claim_id}",
        name         = claimant_name,
        relationship = (
            claimant_d.get("relationship_to_life_assured", "")
            or claimant_d.get("relationship", "")
        ) if isinstance(claimant_d, dict) else "",
        contact      = (
            claimant_d.get("contact_number", "")
            or claimant_d.get("contact", "")
        ) if isinstance(claimant_d, dict) else "",
        address      = claimant_d.get("address", "") if isinstance(claimant_d, dict) else "",
    ) if claimant_name else None

    # ── NomineeNode ────────────────────────────────────────────────────────────
    nominee_id = (
        claimant_d.get("nominee_id", "") if isinstance(claimant_d, dict)
        else getattr(claimant_d, "nominee_id", "")
    )
    nominee = NomineeNode(
        id           = nominee_id,
        name         = nominee_id,
        claimant_name= claimant_name,
    ) if nominee_id else None

    # ── BankAccountNode ────────────────────────────────────────────────────────
    bank_account = (
        claimant_d.get("bank_account", "")
        or claimant_d.get("bank_account_number", "")
    ) if isinstance(claimant_d, dict) else getattr(claimant_d, "bank_account", "")
    bank_node = BankAccountNode(
        id              = bank_account,
        bank_name       = claimant_d.get("bank_name", "") if isinstance(claimant_d, dict) else "",
        bank_ifsc       = claimant_d.get("bank_ifsc", "") if isinstance(claimant_d, dict) else "",
        account_holder  = claimant_name,
    ) if bank_account else None

    # ── HospitalNode ───────────────────────────────────────────────────────────
    hospital_name = (
        death_d.get("hospital_name", "") if isinstance(death_d, dict)
        else getattr(death_d, "hospital_name", "")
    )
    hospital = HospitalNode(
        id          = _norm(hospital_name) or f"hospital:{claim_id}",
        name        = hospital_name,
        address     = death_d.get("hospital_address", "") if isinstance(death_d, dict) else "",
        hospital_id = death_d.get("hospital_id", "") if isinstance(death_d, dict) else "",
    ) if hospital_name else None

    # ── DoctorNode ─────────────────────────────────────────────────────────────
    doctor_name = (
        death_d.get("doctor_name", "")
        or death_d.get("attending_doctor", "")
    ) if isinstance(death_d, dict) else getattr(death_d, "doctor_name", "")
    doctor = DoctorNode(
        id   = _norm(doctor_name) or f"doctor:{claim_id}",
        name = doctor_name,
    ) if doctor_name else None

    # ── DeathEventNode ─────────────────────────────────────────────────────────
    death_event = DeathEventNode(
        id             = f"{claim_id}:death",
        date_of_death  = death_d.get("date_of_death", "") if isinstance(death_d, dict) else "",
        cause_of_death = death_d.get("cause_of_death", "") if isinstance(death_d, dict) else "",
        manner_of_death= death_d.get("manner_of_death", "") if isinstance(death_d, dict) else "",
        place_of_death = death_d.get("place_of_death", "") if isinstance(death_d, dict) else "",
    )

    return ClaimGraph(
        claim        = claim,
        policy       = policy,
        claimant     = claimant,
        nominee      = nominee,
        bank_account = bank_node,
        hospital     = hospital,
        doctor       = doctor,
        death_event  = death_event,
    )


class Neo4jGraphStore:
    """
    Production-grade Neo4j persistence layer for the Claim Evidence Graph.

    Responsibilities:
      - Connect to Neo4j with retry-safe initialization
      - Build graph schema constraints on first run
      - persist_claim_graph() — writes all 8 nodes + 7 relationships atomically
      - Fail safe — pipeline is never blocked by Neo4j failures

    Usage:
      store = Neo4jGraphStore(uri, user, password)
      store.persist_claim_graph(state)
      store.close()
    """

    # Cypher: create uniqueness constraints for each node label
    _CONSTRAINTS = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Claim)        REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Policy)       REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Claimant)     REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Nominee)      REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:BankAccount)  REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Hospital)     REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Doctor)       REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DeathEvent)   REQUIRE n.id IS UNIQUE",
    ]

    def __init__(self, uri: str, user: str, password: str):
        self._driver = None
        self._available = False
        self._connect(uri, user, password)

    def _connect(self, uri: str, user: str, password: str) -> None:
        if not uri:
            logger.info("[Neo4jGraphStore] No URI provided — operating without Neo4j")
            return
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify connectivity
            self._driver.verify_connectivity()
            self._available = True
            logger.info("[Neo4jGraphStore] Connected to Neo4j: %s", uri)
            self._build_constraints()
        except Exception as exc:
            logger.warning(
                "[Neo4jGraphStore] Neo4j unavailable — graph stored in NetworkX only: %s", exc
            )

    def _build_constraints(self) -> None:
        """Create uniqueness constraints so MERGE works correctly."""
        try:
            with self._driver.session() as session:
                for cypher in self._CONSTRAINTS:
                    session.run(cypher)
            logger.info("[Neo4jGraphStore] Schema constraints ready")
        except Exception as exc:
            logger.warning("[Neo4jGraphStore] Constraint creation failed: %s", exc)

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Main public method ─────────────────────────────────────────────────────

    def persist_claim_graph(self, state) -> bool:
        """
        Extract all 8 nodes from ClaimState and persist to Neo4j atomically.
        Uses MERGE — safe to call multiple times for the same claim.
        Returns True on success, False on failure.
        """
        if not self._available:
            return False

        try:
            graph = _extract_claim_graph(state)
            with self._driver.session() as session:
                session.execute_write(self._write_graph, graph)
            logger.info(
                "[Neo4jGraphStore] Persisted graph for claim %s",
                graph.claim.id,
            )
            return True
        except Exception as exc:
            logger.error(
                "[Neo4jGraphStore] persist_claim_graph failed for %s: %s",
                getattr(state, "claim_case_id", "?"), exc,
            )
            return False

    # ── Neo4j write transaction ────────────────────────────────────────────────

    @staticmethod
    def _write_graph(tx, graph: ClaimGraph) -> None:
        """
        Write all nodes and relationships in a single transaction.
        Uses MERGE everywhere — fully idempotent.
        """
        c = graph.claim

        # 1. Claim node
        tx.run("""
            MERGE (n:Claim {id: $id})
            SET n.policy_number         = $policy_number,
                n.policy_age_days       = $policy_age_days,
                n.policy_sum_assured    = $policy_sum_assured,
                n.revival_detected      = $revival_detected,
                n.final_recommendation  = $final_recommendation,
                n.escalation_required   = $escalation_required,
                n.fraud_risk_score      = $fraud_risk_score,
                n.fraud_risk_level      = $fraud_risk_level
        """, **c.model_dump())

        # 2. Policy node + relationship
        if graph.policy:
            p = graph.policy
            tx.run("""
                MERGE (n:Policy {id: $id})
                SET n.policy_number    = $policy_number,
                    n.issue_date       = $issue_date,
                    n.sum_assured      = $sum_assured,
                    n.premium          = $premium,
                    n.revival_detected = $revival_detected,
                    n.revival_date     = $revival_date
                WITH n
                MATCH (c:Claim {id: $claim_id})
                MERGE (c)-[:UNDER_POLICY]->(n)
            """, **p.model_dump(), claim_id=c.id)

        # 3. Claimant node + relationship
        if graph.claimant:
            cl = graph.claimant
            tx.run("""
                MERGE (n:Claimant {id: $id})
                SET n.name         = $name,
                    n.relationship = $relationship,
                    n.contact      = $contact,
                    n.address      = $address
                WITH n
                MATCH (c:Claim {id: $claim_id})
                MERGE (c)-[:FILED_BY]->(n)
            """, **cl.model_dump(), claim_id=c.id)

        # 4. Nominee node + relationship (KEY FRAUD ENTITY)
        if graph.nominee:
            n = graph.nominee
            tx.run("""
                MERGE (n:Nominee {id: $id})
                SET n.name          = $name,
                    n.claimant_name = $claimant_name
                WITH n
                MATCH (c:Claim {id: $claim_id})
                MERGE (c)-[:NAMES_NOMINEE]->(n)
            """, **n.model_dump(), claim_id=c.id)

        # 5. BankAccount node + relationship (KEY FRAUD ENTITY)
        if graph.bank_account:
            ba = graph.bank_account
            tx.run("""
                MERGE (n:BankAccount {id: $id})
                SET n.bank_name      = $bank_name,
                    n.bank_ifsc      = $bank_ifsc,
                    n.account_holder = $account_holder
                WITH n
                MATCH (c:Claim {id: $claim_id})
                MERGE (c)-[:PAYS_TO]->(n)
            """, **ba.model_dump(), claim_id=c.id)

        # 6. DeathEvent node + relationship
        if graph.death_event:
            de = graph.death_event
            tx.run("""
                MERGE (n:DeathEvent {id: $id})
                SET n.date_of_death  = $date_of_death,
                    n.cause_of_death = $cause_of_death,
                    n.manner_of_death= $manner_of_death,
                    n.place_of_death = $place_of_death
                WITH n
                MATCH (c:Claim {id: $claim_id})
                MERGE (c)-[:HAS_DEATH_EVENT]->(n)
            """, **de.model_dump(), claim_id=c.id)

        # 7. Hospital node + relationship (via DeathEvent)
        if graph.hospital and graph.death_event:
            h = graph.hospital
            tx.run("""
                MERGE (n:Hospital {id: $id})
                SET n.name        = $name,
                    n.address     = $address,
                    n.hospital_id = $hospital_id
                WITH n
                MATCH (d:DeathEvent {id: $death_event_id})
                MERGE (d)-[:OCCURRED_AT]->(n)
            """, **h.model_dump(), death_event_id=graph.death_event.id)

        # 8. Doctor node + relationship (via Hospital)
        if graph.doctor and graph.hospital:
            doc = graph.doctor
            tx.run("""
                MERGE (n:Doctor {id: $id})
                SET n.name = $name
                WITH n
                MATCH (h:Hospital {id: $hospital_id})
                MERGE (h)-[:ATTENDED_BY]->(n)
            """, **doc.model_dump(), hospital_id=graph.hospital.id)

    # ── Helper: extract and return graph for inspection ────────────────────────

    def extract_graph(self, state) -> ClaimGraph:
        """Public method to extract graph schema from a ClaimState without writing."""
        return _extract_claim_graph(state)
