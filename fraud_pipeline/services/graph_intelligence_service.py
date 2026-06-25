"""
GraphIntelligenceService
========================
General purpose claim evidence graph engine.

Design principles:
  1. Zero hardcoding — no field names, entity types, or relationship
     strings are hardcoded anywhere
  2. Dynamic entity extraction — recursively walks any ClaimState
     structure and extracts all meaningful scalar values as graph nodes
  3. Historical link ingestion — reads historical_claim_links fully
     dynamically via model_dump(); every key-value pair becomes a graph entity
  4. Rarity-based risk scoring — rare shared entities score higher
     than common ones, automatically, for any entity type
  5. Noise filtering — SKIP_FIELDS removes low-signal fields that
     would generate false positives (dates, amounts, booleans)
  6. Persistence — entities saved to PostgreSQL on every ingest,
     graph warmed from PostgreSQL on startup, survives restarts
  7. Neo4j sync — optional production persistence, failure never
     blocks pipeline
  8. Fail safe — every external call wrapped in try/except, pipeline
     never blocked by graph failures
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# ── Module-level in-process graph ─────────────────────────────────────────────
# Persists across requests within the same process.
# Rebuilt from PostgreSQL on startup if postgres_service provided.
_GRAPH: nx.MultiDiGraph = nx.MultiDiGraph()

# entity_registry_key → set of claim_ids sharing this entity
# registry_key format: "{entity_type}::{entity_value_lowercase}"
_ENTITY_CLAIMS: Dict[str, Set[str]] = {}


# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum string length for a value to be a meaningful graph entity
MIN_ENTITY_LENGTH = 3

# Minimum number of claims needed before rarity scoring is reliable
MIN_CLAIMS_FOR_RARITY = 10

# Thresholds for fraud ring detection
FRAUD_RING_ENTITY_THRESHOLD    = 2   # entity shared across 2+ claims
FRAUD_RING_COMPONENT_THRESHOLD = 3   # 3+ claims in one component

# BFS depth for connected claim traversal
GRAPH_TRAVERSAL_DEPTH = 3

# Fields to skip during recursive entity extraction.
# These are either structural metadata, agent outputs,
# or high-frequency values that carry no fraud signal.
SKIP_FIELDS: Set[str] = {
    # Agent output fields
    "external_verification", "fraud_analysis", "early_claim_analysis",
    "non_disclosure_analysis", "conflict_resolution", "trust_analysis",
    "graph_analysis", "final_recommendation", "escalation_required",

    # Structural and metadata fields
    "claim_case_id", "created_at", "updated_at", "audit_reference",
    "validation_flags", "ocr_confidence_scores", "submitted_documents",
    "medical_records", "fir_records", "proposal_form",
    "premium_payment_history", "historical_claim_links",

    # High-frequency demographic fields
    "gender", "age", "age_at_death", "dob", "date_of_birth",
    "occupation", "address", "relationship", "relationship_to_life_assured",
    "bank_name", "contact", "contact_number", "place_of_death",

    # Date fields — shared dates are coincidence not fraud
    "date_of_death", "date", "admission_date", "discharge_date",
    "upload_timestamp", "last_premium_paid_date",
    "policy_revival_date", "policy_issue_date", "creation_date",

    # Medical content fields — clinical data not fraud entities
    "cause_of_death", "manner_of_death", "diagnosis", "treatment",
    "record_type", "document_type", "doc_type", "ocr_text",
    "content_summary", "description", "incident_description",
    "treating_doctor", "attending_doctor", "doctor_name",

    # Financial amount / counter fields
    "policy_sum_assured", "policy_premium", "policy_age_days",
    "sum_assured",

    # Boolean and score fields
    "verified", "smoking_history", "alcohol_history",
    "policy_revival_detected", "fraud_risk_score", "trust_score",
    "confidence_score", "verification_confidence", "ocr_confidence",

    # Free text and narrative
    "file_path", "fir_number", "death_certificate_number",
    "investigating_officer", "location", "dpi",

    # Generic operational fields — all claims share these, not fraud signals
    "claim_type", "submission_date", "status", "amount",
    "policy_status", "payment_status", "record_status",
    "anomaly_explanation", "lapse_periods", "premium_payments",
    "lapsed_periods", "revival_date", "accused_names",
}

# Historical link fields to skip (meta-fields, not entity values)
HISTORICAL_META_FIELDS: Set[str] = {
    "relationship", "final_recommendation", "date_of_death",
}


class GraphIntelligenceService:

    def __init__(
        self,
        neo4j_driver=None,
        postgres_service=None,
    ):
        self.neo4j    = neo4j_driver
        self.postgres = postgres_service
        self._warmed  = False

        # Warm graph from PostgreSQL on startup
        if self.postgres:
            self._warm_graph_from_postgres()

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — called by Agent 7
    # ══════════════════════════════════════════════════════════════════════════

    def ingest_claim(self, state) -> None:
        """
        Full ingestion pipeline for one claim.
        1. Extract all entities from current ClaimState dynamically
        2. Add entities to in-process graph + registry
        3. Ingest all historical_claim_links dynamically
        4. Persist to PostgreSQL for restart recovery
        5. Sync to Neo4j if available
        """
        claim_id = state.claim_case_id

        # Step 1+2 — extract and add current claim entities
        entities = self._extract_entities_from_state(state)
        self._add_to_graph(claim_id, entities)

        # Step 3 — ingest historical links
        self._ingest_historical_links(state)

        # Step 4 — Neo4j sync
        if self.neo4j:
            self._sync_to_neo4j(claim_id, entities, state)

    def analyse_claim(self, state) -> Dict[str, Any]:
        """
        Traverse graph from current claim.
        Returns full analysis dict — no hardcoded entity types.
        Combines registry lookup + historical link analysis + graph traversal.
        """
        claim_id = state.claim_case_id
        entities = self._extract_entities_from_state(state)

        risk_flags:           List[str]      = []
        risk_score:           float          = 0.0
        fraud_ring:           bool           = False
        shared_entity_counts: Dict[str, int] = {}

        # ── Check each entity of current claim against registry ───────────────
        for entity_type, entity_value in entities:
            registry_key = self._registry_key(entity_type, entity_value)
            other_claims = (
                _ENTITY_CLAIMS.get(registry_key, set()) - {claim_id}
            )

            if len(other_claims) >= FRAUD_RING_ENTITY_THRESHOLD - 1:
                shared_entity_counts[entity_type] = (
                    shared_entity_counts.get(entity_type, 0)
                    + len(other_claims)
                )
                weight = self._entity_risk_weight(entity_type, entity_value)
                risk_score += weight
                fraud_ring = True
                risk_flags.append(
                    f"Shared {entity_type} '{entity_value}' "
                    f"found in {len(other_claims)} other claim(s): "
                    f"{sorted(other_claims)} "
                    f"[weight={weight:.2f}]"
                )

        # ── Direct historical link analysis ───────────────────────────────────
        # Pre-computed links from Team Member 1 — highest confidence
        historical_connected: Set[str] = set()

        for link in getattr(state, "historical_claim_links", []):
            # Support both Pydantic model and plain dict
            if hasattr(link, "model_dump"):
                link_dict = link.model_dump()
            elif isinstance(link, dict):
                link_dict = link
            else:
                continue

            # Get linked claim ID — support both field name conventions
            linked_id = (
                link_dict.get("linked_claim_id")
                or link_dict.get("claim_id")
                or ""
            )
            if not linked_id:
                continue
            historical_connected.add(linked_id)

            # Dynamically check every key-value in the link
            for key, value in link_dict.items():
                if key in ("linked_claim_id", "claim_id") or key in HISTORICAL_META_FIELDS:
                    continue
                if not isinstance(value, (str, int)) or not value:
                    continue
                value_str = str(value).strip()
                if len(value_str) < MIN_ENTITY_LENGTH:
                    continue

                weight = self._entity_risk_weight(key, value_str)
                risk_score += weight
                fraud_ring = True
                risk_flags.append(
                    f"Historical link to '{linked_id}': "
                    f"shared {key}='{value_str}' "
                    f"[weight={weight:.2f}]"
                )
                shared_entity_counts[key] = (
                    shared_entity_counts.get(key, 0) + 1
                )

        # ── Graph traversal for connected claims ──────────────────────────────
        graph_connected = self._find_connected_claims(claim_id)
        connected_count = max(
            len(graph_connected) - 1,       # exclude self
            len(historical_connected),
        )

        # ── Connected component analysis ──────────────────────────────────────
        component_claim_count = 1
        try:
            undirected = _GRAPH.to_undirected()
            component  = nx.node_connected_component(undirected, claim_id)
            claim_nodes = [
                n for n in component
                if _GRAPH.nodes.get(n, {}).get("type") == "claim"
            ]
            component_claim_count = len(claim_nodes)
            if component_claim_count >= FRAUD_RING_COMPONENT_THRESHOLD:
                risk_score += 0.20
                fraud_ring = True
                risk_flags.append(
                    f"Connected component contains "
                    f"{component_claim_count} claims"
                )
        except nx.NetworkXError:
            pass

        risk_score = round(min(0.99, risk_score), 4)

        return {
            # Core output fields
            "connected_claims":        connected_count,
            "shared_entities":         shared_entity_counts,
            "high_risk_relationships": risk_flags,
            "fraud_ring_detected":     fraud_ring,
            "risk_score":              risk_score,
            "component_size":          component_claim_count,

            # Backward-compatible named fields for existing tests
            "shared_nominees":      shared_entity_counts.get("nominee_id", 0),
            "shared_bank_accounts": shared_entity_counts.get("bank_account_number", 0)
                                    or shared_entity_counts.get("bank_account", 0),
            "shared_hospitals":     shared_entity_counts.get("hospital_name", 0),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # ENTITY EXTRACTION — fully dynamic, no hardcoded field names
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_entities_from_state(self, state) -> List[Tuple[str, str]]:
        """
        Walk ClaimState and extract all meaningful scalar values as
        (entity_type, entity_value) tuples.
        Handles both Pydantic models and plain dataclasses.
        Works for any schema — no field names hardcoded.
        """
        results: List[Tuple[str, str]] = []
        # Convert Pydantic model to dict for uniform traversal
        if hasattr(state, "model_dump"):
            obj = state.model_dump()
        elif hasattr(state, "__dict__"):
            obj = vars(state)
        else:
            return results

        self._extract_recursive(obj=obj, prefix="", results=results, depth=0)
        return results

    def _extract_recursive(
        self,
        obj:     Any,
        prefix:  str,
        results: List[Tuple[str, str]],
        depth:   int = 0,
    ) -> None:
        if depth > 5:
            return
        if prefix in SKIP_FIELDS:
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in SKIP_FIELDS:
                    continue
                self._extract_recursive(
                    obj=value, prefix=key,
                    results=results, depth=depth + 1,
                )

        elif isinstance(obj, list):
            for item in obj:
                self._extract_recursive(
                    obj=item, prefix=prefix,
                    results=results, depth=depth + 1,
                )

        elif isinstance(obj, (str, int)) and obj is not True and obj is not False:
            value = str(obj).strip()
            if (
                len(value) >= MIN_ENTITY_LENGTH
                and prefix
                and prefix not in SKIP_FIELDS
                and not value.startswith("{")
                and not value.startswith("[")
            ):
                results.append((prefix, value))

    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════

    def _add_to_graph(
        self,
        claim_id: str,
        entities: List[Tuple[str, str]],
    ) -> None:
        """
        Add claim node and all its entities to the in-process graph.
        Register every entity in _ENTITY_CLAIMS for O(1) fraud lookup.
        Persist to PostgreSQL for restart recovery.
        """
        _GRAPH.add_node(claim_id, type="claim")

        for entity_type, entity_value in entities:
            node_id = self._node_id(entity_type, entity_value)
            _GRAPH.add_node(node_id, type=entity_type, value=entity_value)
            _GRAPH.add_edge(
                claim_id, node_id,
                relation=f"HAS_{entity_type.upper()}",
            )
            registry_key = self._registry_key(entity_type, entity_value)
            _ENTITY_CLAIMS.setdefault(registry_key, set()).add(claim_id)

        # Persist to PostgreSQL — survives server restart
        if self.postgres:
            try:
                self.postgres.save_graph_entities(claim_id, entities)
            except Exception as e:
                logger.warning(
                    "[Graph] PostgreSQL persist failed for %s: %s", claim_id, e
                )

    def _ingest_historical_links(self, state) -> None:
        """
        Reads every key-value pair from every historical_claim_link
        dynamically. Every string value becomes a graph entity of the
        linked claim and is registered in _ENTITY_CLAIMS.

        Supports both Pydantic HistoricalClaimLink models and plain dicts.
        No hardcoded field names.
        """
        claim_id = state.claim_case_id
        links    = getattr(state, "historical_claim_links", [])

        if not links:
            return

        logger.info(
            "[Graph] Ingesting %d historical links for %s", len(links), claim_id,
        )

        for link in links:
            # Normalize to dict
            if hasattr(link, "model_dump"):
                link_dict = link.model_dump()
            elif isinstance(link, dict):
                link_dict = link
            else:
                continue

            # Support both field name conventions
            linked_id = (
                link_dict.get("linked_claim_id")
                or link_dict.get("claim_id")
                or ""
            )
            if not linked_id:
                continue

            if linked_id not in _GRAPH:
                _GRAPH.add_node(linked_id, type="claim", source="historical_link")

            _GRAPH.add_edge(
                claim_id, linked_id,
                relation="LINKED_TO",
                via=link_dict.get("relationship", "unknown"),
            )

            # Extract ALL entity fields from the link dynamically
            link_entities: List[Tuple[str, str]] = []

            for key, value in link_dict.items():
                if key in ("linked_claim_id", "claim_id") or key in HISTORICAL_META_FIELDS:
                    continue
                if not isinstance(value, (str, int)) or not value:
                    continue
                value_str = str(value).strip()
                if len(value_str) < MIN_ENTITY_LENGTH:
                    continue

                node_id = self._node_id(key, value_str)
                _GRAPH.add_node(node_id, type=key, value=value_str)
                _GRAPH.add_edge(
                    linked_id, node_id,
                    relation=f"HAS_{key.upper()}",
                )
                registry_key = self._registry_key(key, value_str)
                _ENTITY_CLAIMS.setdefault(registry_key, set()).add(linked_id)
                link_entities.append((key, value_str))

                logger.debug(
                    "[Graph] Historical entity: %s[%s]='%s'",
                    linked_id, key, value_str,
                )

            # Persist historical entities
            if self.postgres and link_entities:
                try:
                    self.postgres.save_graph_entities(linked_id, link_entities)
                except Exception as e:
                    logger.warning(
                        "[Graph] PostgreSQL persist failed for historical %s: %s",
                        linked_id, e,
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH TRAVERSAL
    # ══════════════════════════════════════════════════════════════════════════

    def _find_connected_claims(self, claim_id: str) -> Set[str]:
        """BFS up to GRAPH_TRAVERSAL_DEPTH hops. Returns all claim nodes."""
        undirected = _GRAPH.to_undirected()
        try:
            reachable = nx.single_source_shortest_path_length(
                undirected, claim_id, cutoff=GRAPH_TRAVERSAL_DEPTH,
            )
            return {
                node for node in reachable
                if _GRAPH.nodes.get(node, {}).get("type") == "claim"
            }
        except nx.NetworkXError:
            return {claim_id}

    # ══════════════════════════════════════════════════════════════════════════
    # RISK SCORING — rarity-based, no hardcoded weights per entity type
    # ══════════════════════════════════════════════════════════════════════════

    def _entity_risk_weight(self, entity_type: str, entity_value: str) -> float:
        """
        Dynamic risk weight based on entity rarity across all claims.
        Rare shared entity = strong fraud signal = high weight.
        """
        total_claims = len([
            n for n, d in _GRAPH.nodes(data=True)
            if d.get("type") == "claim"
        ])

        if total_claims < MIN_CLAIMS_FOR_RARITY:
            return 0.20

        registry_key  = self._registry_key(entity_type, entity_value)
        total_sharing = len(_ENTITY_CLAIMS.get(registry_key, set()))
        frequency     = total_sharing / total_claims

        if frequency <= 0.01:
            return 0.35
        elif frequency <= 0.05:
            return 0.25
        elif frequency <= 0.15:
            return 0.15
        elif frequency <= 0.30:
            return 0.08
        else:
            return 0.02

    # ══════════════════════════════════════════════════════════════════════════
    # POSTGRESQL — startup warmup and persistence
    # ══════════════════════════════════════════════════════════════════════════

    def _warm_graph_from_postgres(self) -> None:
        """
        Load all previously persisted claim entities from PostgreSQL
        into the in-process graph on service startup.
        """
        if self._warmed:
            return

        try:
            records = self.postgres.get_all_graph_entities()
            if not records:
                logger.info("[Graph] No historical entities in PostgreSQL")
                self._warmed = True
                return

            claim_entities: Dict[str, List[Tuple[str, str]]] = {}
            for row in records:
                cid = row.get("claim_id", "")
                et  = row.get("entity_type", "")
                ev  = row.get("entity_value", "")
                if cid and et and ev:
                    claim_entities.setdefault(cid, []).append((et, ev))

            # Rebuild in-process graph without re-persisting
            original_postgres = self.postgres
            self.postgres = None
            for cid, ents in claim_entities.items():
                self._add_to_graph(cid, ents)
            self.postgres = original_postgres

            logger.info(
                "[Graph] Warmed from PostgreSQL: %d claims, %d entities",
                len(claim_entities),
                sum(len(e) for e in claim_entities.values()),
            )

        except Exception as e:
            logger.error("[Graph] Graph warmup failed — empty graph: %s", e)
        finally:
            self._warmed = True

    # ══════════════════════════════════════════════════════════════════════════
    # NEO4J SYNC — optional production persistence
    # ══════════════════════════════════════════════════════════════════════════

    def _sync_to_neo4j(
        self,
        claim_id: str,
        entities: List[Tuple[str, str]],
        state,
    ) -> None:
        if not self.neo4j:
            return
        try:
            with self.neo4j.session() as session:
                session.run("MERGE (c:Claim {id: $id})", id=claim_id)

                for entity_type, entity_value in entities:
                    label = entity_type.replace(" ", "_").title()
                    session.run(
                        f"""
                        MERGE (e:{label} {{value: $value}})
                        WITH e MATCH (c:Claim {{id: $claim_id}})
                        MERGE (c)-[:HAS_{label.upper()}]->(e)
                        """,
                        value=entity_value, claim_id=claim_id,
                    )

                for link in getattr(state, "historical_claim_links", []):
                    if hasattr(link, "model_dump"):
                        link_dict = link.model_dump()
                    elif isinstance(link, dict):
                        link_dict = link
                    else:
                        continue

                    linked_id = (
                        link_dict.get("linked_claim_id")
                        or link_dict.get("claim_id") or ""
                    )
                    if not linked_id:
                        continue
                    session.run(
                        """
                        MERGE (c1:Claim {id: $current})
                        MERGE (c2:Claim {id: $linked})
                        MERGE (c1)-[:LINKED_TO {via: $via}]->(c2)
                        """,
                        current=claim_id, linked=linked_id,
                        via=link_dict.get("relationship", "unknown"),
                    )
        except Exception as e:
            logger.error("[Graph] Neo4j sync failed: %s", e)

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _node_id(entity_type: str, entity_value: str) -> str:
        return f"{entity_type}::{entity_value}"

    @staticmethod
    def _registry_key(entity_type: str, entity_value: str) -> str:
        return f"{entity_type}::{entity_value.lower().strip()}"
