"""
Neo4j Graph Intelligence Queries — Phase 3
==========================================
Reusable Cypher query wrappers for fraud ring detection.

Five query methods:
  1. detect_shared_nominees()       — nominees across 2+ claims
  2. detect_shared_bank_accounts()  — accounts across 2+ claims
  3. detect_suspicious_hospitals()  — hospitals with unusually high claim volume
  4. detect_shared_doctors()        — doctors appearing repeatedly
  5. detect_fraud_rings()           — full connected cluster analysis for one claim

All methods:
  - Return structured dicts (never raise)
  - Log warnings on Neo4j failure
  - Return empty results when Neo4j unavailable
  - Are called by GraphIntelligenceAgent (Agent 7)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Neo4jGraphQueries:
    """
    Wraps all graph intelligence Cypher queries.
    Requires a connected Neo4jGraphStore instance.

    Usage:
        queries = Neo4jGraphQueries(neo4j_graph_store)
        results = queries.detect_shared_nominees(claim_id="CLM-001")
    """

    def __init__(self, store):
        """
        Args:
            store: Neo4jGraphStore instance (from neo4j_graph_store.py)
        """
        self._store = store

    @property
    def _driver(self):
        return self._store._driver if self._store.is_available else None

    def _run(self, cypher: str, **params) -> List[Dict[str, Any]]:
        """Execute a read query, return list of result dicts. Never raises."""
        if not self._driver:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(cypher, **params)
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("[Neo4jGraphQueries] Query failed: %s", exc)
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY 1 — Shared Nominees
    # ══════════════════════════════════════════════════════════════════════════

    def detect_shared_nominees(
        self,
        min_claims: int = 2,
        claim_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find nominees linked to multiple claims.

        Args:
            min_claims: Minimum number of claims to be flagged (default=2)
            claim_id:   If provided, only return results connected to this claim

        Returns:
            List of dicts:
              nominee_id, nominee_name, claim_ids, claim_count
        """
        if claim_id:
            cypher = """
                MATCH (c1:Claim {id: $claim_id})-[:NAMES_NOMINEE]->(n:Nominee)
                      <-[:NAMES_NOMINEE]-(c2:Claim)
                WHERE c2.id <> $claim_id
                WITH n,
                     [c1.id] + collect(DISTINCT c2.id) AS claim_ids,
                     count(DISTINCT c2) + 1             AS claim_count
                WHERE claim_count >= $min_claims
                RETURN n.id      AS nominee_id,
                       n.name    AS nominee_name,
                       claim_ids,
                       claim_count
                ORDER BY claim_count DESC
            """
            rows = self._run(cypher, claim_id=claim_id, min_claims=min_claims)
        else:
            cypher = """
                MATCH (n:Nominee)<-[:NAMES_NOMINEE]-(c:Claim)
                WITH n,
                     collect(DISTINCT c.id) AS claim_ids,
                     count(DISTINCT c)      AS claim_count
                WHERE claim_count >= $min_claims
                RETURN n.id      AS nominee_id,
                       n.name    AS nominee_name,
                       claim_ids,
                       claim_count
                ORDER BY claim_count DESC
            """
            rows = self._run(cypher, min_claims=min_claims)

        logger.debug(
            "[Neo4jQueries] detect_shared_nominees: %d results (min=%d)",
            len(rows), min_claims,
        )
        return rows

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY 2 — Shared Bank Accounts
    # ══════════════════════════════════════════════════════════════════════════

    def detect_shared_bank_accounts(
        self,
        min_claims: int = 2,
        claim_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find bank accounts linked to multiple claims.

        Returns:
            List of dicts:
              account_id, bank_name, claim_ids, claim_count
        """
        if claim_id:
            cypher = """
                MATCH (c1:Claim {id: $claim_id})-[:PAYS_TO]->(a:BankAccount)
                      <-[:PAYS_TO]-(c2:Claim)
                WHERE c2.id <> $claim_id
                WITH a,
                     [c1.id] + collect(DISTINCT c2.id) AS claim_ids,
                     count(DISTINCT c2) + 1             AS claim_count
                WHERE claim_count >= $min_claims
                RETURN a.id        AS account_id,
                       a.bank_name AS bank_name,
                       claim_ids,
                       claim_count
                ORDER BY claim_count DESC
            """
            rows = self._run(cypher, claim_id=claim_id, min_claims=min_claims)
        else:
            cypher = """
                MATCH (a:BankAccount)<-[:PAYS_TO]-(c:Claim)
                WITH a,
                     collect(DISTINCT c.id) AS claim_ids,
                     count(DISTINCT c)      AS claim_count
                WHERE claim_count >= $min_claims
                RETURN a.id        AS account_id,
                       a.bank_name AS bank_name,
                       claim_ids,
                       claim_count
                ORDER BY claim_count DESC
            """
            rows = self._run(cypher, min_claims=min_claims)

        logger.debug(
            "[Neo4jQueries] detect_shared_bank_accounts: %d results", len(rows)
        )
        return rows

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY 3 — Suspicious Hospitals
    # ══════════════════════════════════════════════════════════════════════════

    def detect_suspicious_hospitals(
        self,
        min_claims: int = 3,
        claim_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find hospitals with unusually high claim volume.
        A hospital appearing in 3+ death claims is suspicious.

        Returns:
            List of dicts:
              hospital_id, hospital_name, claim_ids, claim_count
        """
        if claim_id:
            cypher = """
                MATCH (c1:Claim {id: $claim_id})-[:HAS_DEATH_EVENT]->(d1:DeathEvent)
                      -[:OCCURRED_AT]->(h:Hospital)
                      <-[:OCCURRED_AT]-(d2:DeathEvent)<-[:HAS_DEATH_EVENT]-(c2:Claim)
                WHERE c2.id <> $claim_id
                WITH h,
                     [c1.id] + collect(DISTINCT c2.id) AS claim_ids,
                     count(DISTINCT c2) + 1             AS claim_count
                WHERE claim_count >= $min_claims
                RETURN h.id   AS hospital_id,
                       h.name AS hospital_name,
                       claim_ids,
                       claim_count
                ORDER BY claim_count DESC
            """
            rows = self._run(cypher, claim_id=claim_id, min_claims=min_claims)
        else:
            cypher = """
                MATCH (h:Hospital)<-[:OCCURRED_AT]-(d:DeathEvent)
                      <-[:HAS_DEATH_EVENT]-(c:Claim)
                WITH h,
                     collect(DISTINCT c.id) AS claim_ids,
                     count(DISTINCT c)      AS claim_count
                WHERE claim_count >= $min_claims
                RETURN h.id   AS hospital_id,
                       h.name AS hospital_name,
                       claim_ids,
                       claim_count
                ORDER BY claim_count DESC
            """
            rows = self._run(cypher, min_claims=min_claims)

        logger.debug(
            "[Neo4jQueries] detect_suspicious_hospitals: %d results", len(rows)
        )
        return rows

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY 4 — Shared Doctors
    # ══════════════════════════════════════════════════════════════════════════

    def detect_shared_doctors(
        self,
        min_claims: int = 2,
        claim_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find doctors appearing repeatedly across claims.
        A doctor appearing in 2+ death claims requires scrutiny.

        Returns:
            List of dicts:
              doctor_id, doctor_name, claim_ids, claim_count
        """
        if claim_id:
            cypher = """
                MATCH (c1:Claim {id: $claim_id})-[:HAS_DEATH_EVENT]->(d1:DeathEvent)
                      -[:OCCURRED_AT]->(h:Hospital)-[:ATTENDED_BY]->(doc:Doctor)
                      <-[:ATTENDED_BY]-(:Hospital)<-[:OCCURRED_AT]-(:DeathEvent)
                      <-[:HAS_DEATH_EVENT]-(c2:Claim)
                WHERE c2.id <> $claim_id
                WITH doc,
                     [c1.id] + collect(DISTINCT c2.id) AS claim_ids,
                     count(DISTINCT c2) + 1             AS claim_count
                WHERE claim_count >= $min_claims
                RETURN doc.id   AS doctor_id,
                       doc.name AS doctor_name,
                       claim_ids,
                       claim_count
                ORDER BY claim_count DESC
            """
            rows = self._run(cypher, claim_id=claim_id, min_claims=min_claims)
        else:
            cypher = """
                MATCH (doc:Doctor)<-[:ATTENDED_BY]-(:Hospital)<-[:OCCURRED_AT]-(d:DeathEvent)
                      <-[:HAS_DEATH_EVENT]-(c:Claim)
                WITH doc,
                     collect(DISTINCT c.id) AS claim_ids,
                     count(DISTINCT c)      AS claim_count
                WHERE claim_count >= $min_claims
                RETURN doc.id   AS doctor_id,
                       doc.name AS doctor_name,
                       claim_ids,
                       claim_count
                ORDER BY claim_count DESC
            """
            rows = self._run(cypher, min_claims=min_claims)

        logger.debug(
            "[Neo4jQueries] detect_shared_doctors: %d results", len(rows)
        )
        return rows

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY 5 — Fraud Ring Detection (for one claim)
    # ══════════════════════════════════════════════════════════════════════════

    def detect_fraud_rings(
        self,
        claim_id: str,
        max_hops: int = 3,
    ) -> Dict[str, Any]:
        """
        Identify the connected cluster of claims reachable from claim_id
        through any shared entity (Nominee, BankAccount, Hospital, Doctor).

        Args:
            claim_id: The claim to analyse
            max_hops: Maximum relationship hops (default=3)

        Returns:
            Dict with:
              fraud_ring_detected: bool
              connected_claims:    int
              ring_members:        List[str]  — claim IDs in ring
              ring_paths:          List[dict] — path details
              risk_score:          float
        """
        cypher = """
            MATCH (c:Claim {id: $claim_id})
            CALL apoc.path.subgraphNodes(c, {
                maxLevel: $max_hops,
                labelFilter: '+Claim|+Nominee|+BankAccount|+Hospital|+Doctor'
            })
            YIELD node
            WHERE node:Claim AND node.id <> $claim_id
            RETURN DISTINCT node.id AS connected_claim_id
        """

        # Try APOC first; fall back to manual path query if APOC unavailable
        rows = self._run(cypher, claim_id=claim_id, max_hops=max_hops)

        if not rows:
            # Fallback: manual variable-length path (no APOC required)
            cypher_fallback = """
                MATCH path = (c:Claim {id: $claim_id})-[*1..3]-(other:Claim)
                WHERE other.id <> $claim_id
                RETURN DISTINCT other.id                              AS connected_claim_id,
                       [r IN relationships(path) | type(r)]          AS path_types,
                       length(path)                                   AS hops
                ORDER BY hops ASC
            """
            rows = self._run(
                cypher_fallback, claim_id=claim_id, max_hops=max_hops
            )

        ring_members = list({r.get("connected_claim_id", "") for r in rows if r.get("connected_claim_id")})
        ring_paths   = [
            {
                "connected_claim": r.get("connected_claim_id"),
                "path_types":      r.get("path_types", []),
                "hops":            r.get("hops", 0),
            }
            for r in rows
            if r.get("connected_claim_id")
        ]

        fraud_ring_detected = len(ring_members) >= 1

        # Risk score: starts at 0, increases per connected claim (capped 0.99)
        risk_score = min(0.99, len(ring_members) * 0.20)

        logger.debug(
            "[Neo4jQueries] detect_fraud_rings(%s): %d connected claims",
            claim_id, len(ring_members),
        )

        return {
            "fraud_ring_detected": fraud_ring_detected,
            "connected_claims":    len(ring_members),
            "ring_members":        ring_members,
            "ring_paths":          ring_paths,
            "risk_score":          round(risk_score, 4),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # COMBINED ANALYSIS — run all queries for one claim
    # ══════════════════════════════════════════════════════════════════════════

    def full_claim_analysis(self, claim_id: str) -> Dict[str, Any]:
        """
        Run all 5 intelligence queries for a single claim.
        Returns consolidated results dict consumed by GraphIntelligenceAgent.

        Returns:
            Dict with:
              shared_nominees_list
              shared_bank_accounts_list
              suspicious_hospitals_list
              shared_doctors_list
              fraud_ring_result
              shared_nominees      (count)
              shared_bank_accounts (count)
              suspicious_hospitals (count)
              shared_doctors       (count)
        """
        nominees    = self.detect_shared_nominees(claim_id=claim_id)
        accounts    = self.detect_shared_bank_accounts(claim_id=claim_id)
        hospitals   = self.detect_suspicious_hospitals(claim_id=claim_id, min_claims=2)
        doctors     = self.detect_shared_doctors(claim_id=claim_id)
        fraud_ring  = self.detect_fraud_rings(claim_id=claim_id)

        return {
            # Detailed lists
            "shared_nominees_list":       nominees,
            "shared_bank_accounts_list":  accounts,
            "suspicious_hospitals_list":  hospitals,
            "shared_doctors_list":        doctors,
            "fraud_ring_result":          fraud_ring,

            # Counts (used by GraphIntelligenceAgent for scoring)
            "shared_nominees":       len(nominees),
            "shared_bank_accounts":  len(accounts),
            "suspicious_hospitals":  len(hospitals),
            "shared_doctors":        len(doctors),

            # Ring result passthrough
            "fraud_ring_detected":   fraud_ring["fraud_ring_detected"],
            "ring_members":          fraud_ring["ring_members"],
            "ring_risk_score":       fraud_ring["risk_score"],
        }
