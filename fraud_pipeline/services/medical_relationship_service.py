"""
services/medical_relationship_service.py

4-layer cascade to determine whether two medical terms are related.

Layer 1 — OLS exact match / synonym resolution (EBI OLS4)
Layer 2 — OLS ontology ancestor / IS-A hierarchy
Layer 3 — Disease Ontology Relation Ontology (RO) causal edges
Layer 4 — Manual review fallback (always returns, never fails)

Each layer returns dict | None. The first non-None result wins.
Layer 4 always returns a dict (final safety net).
"""

from __future__ import annotations

import json
import logging
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

# OLS4 base URL
_OLS_BASE = "https://www.ebi.ac.uk/ols4/api"

# Relationship types we look for in Layer 3 (RO / BFO)
_CAUSAL_RELATIONS = {
    "RO:0004026": "disease causes",
    "RO:0004027": "disease has basis in dysfunction of",
    "RO:0004028": "disease has basis in disruption of",
    "BFO:0000050": "part of",
}


class MedicalRelationshipService:
    """
    Determines whether two medical / clinical terms are related
    using a 4-layer cascade of ontology lookups.

    Usage:
        svc = MedicalRelationshipService(redis_client=None)
        result = svc.are_related("chronic obstructive pulmonary disease", "respiratory failure")
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.cache_ttl = 2592000  # 30 days in seconds

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ClaImosAI-FraudPipeline/1.0",
        })

    # ── Public API ─────────────────────────────────────────────────────────────

    def normalize(self, term: str) -> str:
        import re
        return re.sub(r"[^\w\s]", "", term.lower().strip())

    def are_related(self, term_a: str, term_b: str) -> dict:
        """
        Main entry point. Runs the 4-layer cascade and returns a standardised
        result dict. Never raises an exception.
        """
        norm_a = self.normalize(term_a)
        norm_b = self.normalize(term_b)

        # ── Cache check ──────────────────────────────────────────────────────
        cache_key = f"medrel:v1:{hash(norm_a)}:{hash(norm_b)}"
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    logger.debug("Cache hit: %s", cache_key)
                    return json.loads(cached)
            except Exception as e:
                logger.warning("Redis read failed: %s", e)

        # ── Cascade ──────────────────────────────────────────────────────────
        result = (
            self._layer1_exact_match(norm_a, norm_b)
            or self._layer2_ontology_ancestor(norm_a, norm_b)
            or self._layer3_disease_ontology_ro(norm_a, norm_b)
            or self._layer4_manual_review(norm_a, norm_b)
        )

        # ── Cache write ──────────────────────────────────────────────────────
        if self.redis and not result.get("api_failed"):
            try:
                self.redis.setex(cache_key, self.cache_ttl, json.dumps(result))
            except Exception as e:
                logger.warning("Redis write failed: %s", e)

        return result

    # ── Layer 1: OLS Exact Match / Synonym ────────────────────────────────────

    def _layer1_exact_match(self, norm_a: str, norm_b: str) -> dict | None:
        """
        Resolve both terms to OLS concepts.
        If they share the same IRI or any synonym label → related.
        """
        try:
            concept_a = self._ols_search(norm_a)
            concept_b = self._ols_search(norm_b)

            if not concept_a or not concept_b:
                return None

            # Identical IRI
            if concept_a["iri"] == concept_b["iri"]:
                return self._result(
                    related=True,
                    rel_type="identical",
                    confidence=1.0,
                    reasoning=(
                        f"'{concept_a['label']}' and '{concept_b['label']}' "
                        f"resolve to the same ontology concept ({concept_a['iri']})"
                    ),
                    layer=1,
                    label_a=concept_a["label"],
                    label_b=concept_b["label"],
                )

            # Synonym overlap
            labels_a = {concept_a["label"].lower()} | {s.lower() for s in concept_a["synonyms"]}
            labels_b = {concept_b["label"].lower()} | {s.lower() for s in concept_b["synonyms"]}
            shared = labels_a & labels_b
            if shared:
                return self._result(
                    related=True,
                    rel_type="synonym",
                    confidence=0.97,
                    reasoning=(
                        f"'{concept_a['label']}' and '{concept_b['label']}' share "
                        f"synonym label(s): {', '.join(list(shared)[:3])}"
                    ),
                    layer=1,
                    label_a=concept_a["label"],
                    label_b=concept_b["label"],
                )

            return None  # No match — proceed to layer 2

        except Exception as e:
            logger.warning("Layer 1 failed: %s", e)
            return None

    # ── Layer 2: OLS Ancestor / IS-A Hierarchy ────────────────────────────────

    def _layer2_ontology_ancestor(self, norm_a: str, norm_b: str) -> dict | None:
        """
        Fetch ancestor chains for both concepts.
        Check direct parent/child, then sibling (shared ancestor within 2 hops).
        """
        try:
            concept_a = self._ols_search(norm_a)
            concept_b = self._ols_search(norm_b)

            if not concept_a or not concept_b:
                return None

            iri_a = concept_a["iri"]
            iri_b = concept_b["iri"]
            ontology_a = concept_a.get("ontology", "mondo")
            ontology_b = concept_b.get("ontology", "mondo")

            ancestors_a = self._ols_ancestors(iri_a, ontology_a)  # set of IRIs
            ancestors_b = self._ols_ancestors(iri_b, ontology_b)

            # A is a parent of B  (A's IRI appears in B's ancestor chain)
            if iri_a in ancestors_b:
                return self._result(
                    related=True,
                    rel_type="is_a_parent",
                    confidence=0.92,
                    reasoning=(
                        f"'{concept_a['label']}' is an ancestor (parent class) of "
                        f"'{concept_b['label']}' in the ontology hierarchy"
                    ),
                    layer=2,
                    label_a=concept_a["label"],
                    label_b=concept_b["label"],
                )

            # B is a parent of A
            if iri_b in ancestors_a:
                return self._result(
                    related=True,
                    rel_type="is_a_parent",
                    confidence=0.92,
                    reasoning=(
                        f"'{concept_b['label']}' is an ancestor (parent class) of "
                        f"'{concept_a['label']}' in the ontology hierarchy"
                    ),
                    layer=2,
                    label_a=concept_a["label"],
                    label_b=concept_b["label"],
                )

            # Shared ancestor (sibling in same disease family)
            # Both terms share common ancestors — they're in the same disease subtree.
            # Filter out generic upper-level ontology roots (BFO, OGMS, etc.) by
            # requiring the shared ancestor IRIs to be from a recognised disease
            # ontology namespace (MONDO, DOID, HP, EFO).
            shared_ancestors = ancestors_a & ancestors_b
            disease_shared = {
                a for a in shared_ancestors
                if any(ns in a for ns in (
                    "MONDO_", "DOID_", "HP_", "EFO_",
                    "obo/MONDO", "obo/DOID", "obo/HP",
                ))
            }
            if disease_shared:
                return self._result(
                    related=True,
                    rel_type="disease_family",
                    confidence=0.78,
                    reasoning=(
                        f"'{concept_a['label']}' and '{concept_b['label']}' share "
                        f"{len(disease_shared)} common disease ancestor(s) — "
                        f"both belong to the same disease family"
                    ),
                    layer=2,
                    label_a=concept_a["label"],
                    label_b=concept_b["label"],
                )

            return None  # proceed to layer 3

        except Exception as e:
            logger.warning("Layer 2 failed: %s", e)
            return None

    # ── Layer 3: Disease Ontology RO Causal Edges ─────────────────────────────

    def _layer3_disease_ontology_ro(self, norm_a: str, norm_b: str) -> dict | None:
        """
        Check Disease Ontology Relation Ontology edges for explicit causal
        relationships (RO:0004026 disease causes, BFO:0000050 part of, etc.)
        """
        try:
            concept_a = self._ols_search(norm_a, preferred_ontology="doid")
            concept_b = self._ols_search(norm_b, preferred_ontology="doid")

            if not concept_a or not concept_b:
                return None

            iri_a = concept_a["iri"]
            iri_b = concept_b["iri"]

            relations_a = self._do_relations(iri_a)
            relations_b = self._do_relations(iri_b)

            # Check A causes B or B causes A via RO:0004026
            for rel_id, rel_name in _CAUSAL_RELATIONS.items():
                targets_of_a = relations_a.get(rel_id, set())
                targets_of_b = relations_b.get(rel_id, set())

                rel_type = "direct_cause" if "causes" in rel_name else "complication"
                conf = 0.95 if rel_type == "direct_cause" else 0.90

                if iri_b in targets_of_a:
                    return self._result(
                        related=True,
                        rel_type=rel_type,
                        confidence=conf,
                        reasoning=(
                            f"Disease Ontology encodes that '{concept_a['label']}' "
                            f"'{rel_name}' '{concept_b['label']}' (relation {rel_id})"
                        ),
                        layer=3,
                        label_a=concept_a["label"],
                        label_b=concept_b["label"],
                    )

                if iri_a in targets_of_b:
                    return self._result(
                        related=True,
                        rel_type=rel_type,
                        confidence=conf,
                        reasoning=(
                            f"Disease Ontology encodes that '{concept_b['label']}' "
                            f"'{rel_name}' '{concept_a['label']}' (relation {rel_id})"
                        ),
                        layer=3,
                        label_a=concept_a["label"],
                        label_b=concept_b["label"],
                    )

            return None  # proceed to layer 4

        except Exception as e:
            logger.warning("Layer 3 failed: %s", e)
            return None

    # ── Layer 4: Manual Review Fallback ───────────────────────────────────────

    def _layer4_manual_review(self, norm_a: str, norm_b: str) -> dict:
        """
        Final safety net. Always returns. Never calls external services.
        """
        return {
            "related": False,
            "relationship_type": "unresolved",
            "confidence": 0.0,
            "reasoning": (
                f"Could not determine medical relationship between "
                f"'{norm_a}' and '{norm_b}' through any available layer. "
                f"Flagged for manual clinical review."
            ),
            "layer_resolved": 4,
            "concept_a_resolved": norm_a,
            "concept_b_resolved": norm_b,
            "api_failed": False,
        }

    # ── OLS Helpers ───────────────────────────────────────────────────────────

    def _ols_search(self, term: str, preferred_ontology: str = "mondo") -> dict | None:
        """
        Search OLS4 for a term. Returns concept dict with iri, label, synonyms.
        Tries preferred_ontology first, falls back to mondo,doid,efo,hp.
        """
        cache_key = f"ols:concept:v1:{hash(term)}:{preferred_ontology}"
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        ontologies = preferred_ontology
        if preferred_ontology not in ("mondo,doid,efo,hp",):
            ontologies = f"{preferred_ontology},mondo,doid,efo,hp"

        try:
            resp = self.session.get(
                f"{_OLS_BASE}/search",
                params={
                    "q": term,
                    "ontology": ontologies,
                    "rows": 5,
                    "exact": "false",
                    "type": "class",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("response", {}).get("docs", [])

            if not docs:
                return None

            doc = docs[0]
            concept = {
                "iri": doc.get("iri", ""),
                "label": doc.get("label", term),
                "synonyms": doc.get("synonym", []),
                "ontology": doc.get("ontology_name", preferred_ontology),
            }

            if self.redis and concept["iri"]:
                try:
                    self.redis.setex(cache_key, self.cache_ttl, json.dumps(concept))
                except Exception:
                    pass

            return concept

        except Exception as e:
            logger.warning("OLS search failed for '%s': %s", term, e)
            return None

    def _ols_ancestors(self, iri: str, ontology: str) -> set:
        """
        Fetch all ancestor IRIs for a concept via the OLS4 ancestors endpoint.
        IRI must be double URL-encoded.
        Returns a set of ancestor IRIs.
        """
        cache_key = f"ols:ancestors:v1:{hash(iri)}"
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    return set(json.loads(cached))
            except Exception:
                pass

        # Double-encode the IRI as required by OLS4
        encoded_iri = quote(quote(iri, safe=""), safe="")
        ancestors: set = set()

        try:
            url = f"{_OLS_BASE}/ontologies/{ontology}/terms/{encoded_iri}/ancestors"
            page = 0
            while True:
                resp = self.session.get(
                    url,
                    params={"size": 200, "page": page},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                embedded = data.get("_embedded", {})
                terms = embedded.get("terms", [])

                for t in terms:
                    if t.get("iri"):
                        ancestors.add(t["iri"])

                # Pagination
                links = data.get("_links", {})
                if "next" not in links:
                    break
                page += 1
                if page > 10:  # safety cap
                    break

            if self.redis and ancestors:
                try:
                    self.redis.setex(
                        cache_key, self.cache_ttl, json.dumps(list(ancestors))
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.warning("OLS ancestors failed for IRI '%s': %s", iri, e)

        return ancestors

    def _do_relations(self, iri: str) -> dict:
        """
        Fetch RO/BFO relations for a Disease Ontology term.
        Returns dict: { relation_id: set_of_target_IRIs }
        """
        cache_key = f"do:relations:v1:{hash(iri)}"
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    raw = json.loads(cached)
                    return {k: set(v) for k, v in raw.items()}
            except Exception:
                pass

        encoded_iri = quote(quote(iri, safe=""), safe="")
        relations: dict = {}

        try:
            resp = self.session.get(
                f"{_OLS_BASE}/ontologies/doid/terms/{encoded_iri}/relations",
                params={"size": 200},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            embedded = data.get("_embedded", {})

            for rel_block in embedded.get("relations", []):
                rel_iri = rel_block.get("relation", {}).get("iri", "")
                # Convert full IRI to short form: .../RO_0004026 → RO:0004026
                rel_id = rel_iri.rsplit("/", 1)[-1].replace("_", ":")
                target_iri = rel_block.get("target", {}).get("iri", "")
                if rel_id and target_iri:
                    relations.setdefault(rel_id, set()).add(target_iri)

        except Exception as e:
            logger.warning("DO relations failed for IRI '%s': %s", iri, e)

        if self.redis:
            try:
                serialisable = {k: list(v) for k, v in relations.items()}
                self.redis.setex(
                    cache_key, self.cache_ttl, json.dumps(serialisable)
                )
            except Exception:
                pass

        return relations

    # ── Result Builder ────────────────────────────────────────────────────────

    @staticmethod
    def _result(
        related: bool,
        rel_type: str,
        confidence: float,
        reasoning: str,
        layer: int,
        label_a: str,
        label_b: str,
        api_failed: bool = False,
    ) -> dict:
        return {
            "related": related,
            "relationship_type": rel_type,
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "layer_resolved": layer,
            "concept_a_resolved": label_a,
            "concept_b_resolved": label_b,
            "api_failed": api_failed,
        }
