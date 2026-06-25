"""
MedicalInferenceService
Given any diagnosis string, returns all conditions it causally implies.

Source cascade:
  1. Disease Ontology RO causal edges  — strongest, most explicit
  2. Wikidata P828 fuzzy entity search — broad coverage, handles variants
  3. OLS description causal parsing    — fallback, no external model needed

All sources run and results are combined.
Each source wraps in try/except — any single failure is logged and skipped.
"""
import json
import re
import logging
import urllib.parse
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── API endpoints ─────────────────────────────────────────────────────────────
OLS_SEARCH      = "https://www.ebi.ac.uk/ols4/api/search"
OLS_TERM        = "https://www.ebi.ac.uk/ols4/api/ontologies/{ontology}/terms/{iri}"
OLS_RELATIONS   = "https://www.ebi.ac.uk/ols4/api/ontologies/{ontology}/terms/{iri}/relations"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# ── Relation Ontology properties that express causal/basis relationships ──────
CAUSAL_RO = {
    "RO:0004026": "disease_has_basis_in",
    "RO:0002410": "has_condition",
    "RO:0004027": "has_basis_in_dysfunction_of",
    "BFO:0000051": "has_part",
    "RO:0003304": "contributes_to_condition",
}

# ── Text patterns for description parsing — no hardcoded medical terms ────────
CAUSAL_TEXT_PATTERNS = [
    r"caused\s+by\s+([^,.;]{4,60})",
    r"due\s+to\s+([^,.;]{4,60})",
    r"resulting\s+from\s+([^,.;]{4,60})",
    r"secondary\s+to\s+([^,.;]{4,60})",
    r"associated\s+with\s+([^,.;]{4,60})",
    r"attributable\s+to\s+([^,.;]{4,60})",
    r"consequence\s+of\s+([^,.;]{4,60})",
    r"complication\s+of\s+([^,.;]{4,60})",
]

CACHE_TTL = 2592000  # 30 days


class MedicalInferenceService:

    def __init__(self, redis_client=None, timeout: int = 12):
        self.redis   = redis_client
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept":     "application/json",
            "User-Agent": "ClaImosAI-FraudPipeline/1.0",
        })

    # ── Public ────────────────────────────────────────────────────────────────

    def get_implied_conditions(self, diagnosis: str) -> List[Dict]:
        normalized = self._normalize(diagnosis)
        cache_key  = f"inference:implied:v3:{hash(normalized)}"

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Split compound diagnoses ("COPD, lung cancer, chronic smoking") and
        # resolve each sub-term independently — combine all results
        sub_terms = self._split_compound(diagnosis)
        all_implied: List[Dict] = []
        for term in sub_terms:
            all_implied.extend(self._resolve_single(term))

        result = self._deduplicate_sort(all_implied)
        self._cache_set(cache_key, result)
        return result

    @staticmethod
    def _split_compound(diagnosis: str) -> List[str]:
        """Split on commas/semicolons only when each segment is long enough
        to be a meaningful medical term (> 4 chars). Always include the full
        original string as the first candidate."""
        parts = [diagnosis]
        for sep in (",", ";"):
            if sep in diagnosis:
                parts += [p.strip() for p in diagnosis.split(sep) if len(p.strip()) > 4]
        return list(dict.fromkeys(parts))  # deduplicate, preserve order

    def _resolve_single(self, diagnosis: str) -> List[Dict]:
        normalized = self._normalize(diagnosis)
        concept = self._resolve_concept(normalized)
        if not concept:
            logger.warning("[Inference] Cannot resolve concept: %s", diagnosis)
            return []

        implied = []

        # Source 1 — Disease Ontology RO edges
        try:
            implied.extend(self._from_do_ro(concept))
        except Exception as e:
            logger.warning("[Inference] DO RO failed for '%s': %s", diagnosis, e)

        # Source 2 — Wikidata P828 with fuzzy entity search
        try:
            implied.extend(self._from_wikidata(concept["label"]))
        except Exception as e:
            logger.warning("[Inference] Wikidata failed for '%s': %s", diagnosis, e)

        # Source 3 — OLS description causal phrase parsing
        try:
            implied.extend(self._from_ols_description(concept))
        except Exception as e:
            logger.warning("[Inference] OLS desc failed for '%s': %s", diagnosis, e)

        # Source 4 — OLS4 ancestor terms (always returns results if concept resolves)
        # Used as a fallback when Sources 1-3 yield nothing
        try:
            implied.extend(self._from_ols_ancestors(concept))
        except Exception as e:
            logger.warning("[Inference] OLS ancestors failed for '%s': %s", diagnosis, e)

        # Source 5 — Synonyms from OLS search result (broadens matching surface)
        try:
            implied.extend(self._from_synonyms(concept))
        except Exception as e:
            logger.warning("[Inference] Synonyms failed for '%s': %s", diagnosis, e)

        return implied

    # ── Concept resolution ────────────────────────────────────────────────────

    def _resolve_concept(self, term: str) -> Optional[Dict]:
        cache_key = f"ols:concept:v2:{hash(term)}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            response = self.session.get(
                OLS_SEARCH,
                params={
                    "q":        term,
                    "ontology": "doid,mondo",
                    "rows":     5,
                    "exact":    "false",
                },
                timeout=self.timeout,
            )
            docs = self._parse_ols_search(response.json())
            if not docs:
                return None

            concept = {
                "iri":      docs[0]["iri"],
                "label":    docs[0]["label"],
                "ontology": docs[0]["ontology_name"],
                "synonyms": docs[0].get("synonym", []),
            }
            self._cache_set(cache_key, concept)
            return concept

        except Exception as e:
            logger.warning("[Inference] OLS resolve failed for '%s': %s", term, e)
            return None

    # ── Source 1 — Disease Ontology RO causal edges ───────────────────────────

    def _from_do_ro(self, concept: Dict) -> List[Dict]:
        encoded = self._double_encode(concept["iri"])
        url = OLS_RELATIONS.format(ontology=concept["ontology"], iri=encoded)
        response = self.session.get(url, timeout=self.timeout)
        results = []
        for term in self._parse_ols_embedded(response.json()):
            rel = term.get("relation_type", "")
            if rel in CAUSAL_RO:
                label = term.get("label", "")
                if label:
                    results.append({
                        "text":         label,
                        "concept_iri":  term.get("iri", ""),
                        "relationship": CAUSAL_RO[rel],
                        "confidence":   0.95,
                        "source":       "DO_RO",
                    })
        return results

    # ── Source 2 — Wikidata P828 with fuzzy entity search ────────────────────

    def _from_wikidata(self, label: str) -> List[Dict]:
        entity_ids = self._wikidata_search_entities(label)
        if not entity_ids:
            return []

        results = []
        for entity_id in entity_ids[:2]:
            query = f"""
            SELECT ?cause ?causeLabel WHERE {{
              wd:{entity_id} wdt:P828 ?cause .
              SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" .
              }}
            }}
            LIMIT 10
            """
            try:
                response = self.session.get(
                    WIKIDATA_SPARQL,
                    params={"query": query, "format": "json"},
                    timeout=15,
                )
                for binding in response.json()["results"]["bindings"]:
                    cause_label = binding.get("causeLabel", {}).get("value", "")
                    cause_uri   = binding.get("cause", {}).get("value", "")
                    if cause_label and not cause_label.startswith("Q"):
                        results.append({
                            "text":         cause_label,
                            "concept_iri":  cause_uri,
                            "relationship": "wikidata_has_cause",
                            "confidence":   0.88,
                            "source":       "WIKIDATA",
                        })
            except Exception as e:
                logger.warning("[Inference] Wikidata P828 query failed: %s", e)

        return results

    def _wikidata_search_entities(self, label: str) -> List[str]:
        """
        Uses Wikidata MediaWiki entity search — handles fuzzy matching,
        synonyms, abbreviations. Returns list of Wikidata entity IDs (Qxxx).
        """
        query = f"""
        SELECT ?disease WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:api "EntitySearch" .
            bd:serviceParam wikibase:endpoint "www.wikidata.org" .
            bd:serviceParam mwapi:search "{label}" .
            bd:serviceParam mwapi:language "en" .
            bd:serviceParam mwapi:limit "5" .
            ?disease wikibase:apiOutputItem mwapi:item .
          }}
          ?disease wdt:P31/wdt:P279* wd:Q12136 .
        }}
        LIMIT 3
        """
        try:
            response = self.session.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                timeout=15,
            )
            ids = []
            for binding in response.json()["results"]["bindings"]:
                uri = binding.get("disease", {}).get("value", "")
                if uri:
                    entity_id = uri.split("/")[-1]
                    if entity_id.startswith("Q"):
                        ids.append(entity_id)
            return ids
        except Exception as e:
            logger.warning("[Inference] Wikidata entity search failed: %s", e)
            return []

    # ── Source 3 — OLS description causal phrase parsing ─────────────────────

    def _from_ols_description(self, concept: Dict) -> List[Dict]:
        encoded = self._double_encode(concept["iri"])
        url = OLS_TERM.format(ontology=concept["ontology"], iri=encoded)
        response = self.session.get(url, timeout=self.timeout)
        descriptions = response.json().get("description", [])
        if not descriptions:
            return []

        text = descriptions[0].lower()
        results = []
        for pattern in CAUSAL_TEXT_PATTERNS:
            for match in re.findall(pattern, text):
                clean = match.strip().rstrip(".,;")
                if len(clean) > 3:
                    results.append({
                        "text":         clean,
                        "concept_iri":  None,
                        "relationship": "description_parse",
                        "confidence":   0.75,
                        "source":       "OLS_DESC",
                    })
        return results

    # ── Source 4 — OLS4 ancestors as implied conditions ──────────────────────
    # Every ancestor of a disease is an implied broader condition.
    # Filters out generic BFO/OGMS roots — only disease-namespace IRIs.

    def _from_ols_ancestors(self, concept: Dict) -> List[Dict]:
        encoded = self._double_encode(concept["iri"])
        url = f"https://www.ebi.ac.uk/ols4/api/ontologies/{concept['ontology']}/terms/{encoded}/ancestors"
        results = []
        page = 0
        while page < 3:  # max 3 pages = 600 ancestors
            response = self.session.get(url, params={"size": 200, "page": page},
                                        timeout=self.timeout)
            data = response.json()
            terms = data.get("_embedded", {}).get("terms", [])
            for t in terms:
                iri = t.get("iri", "")
                label = t.get("label", "")
                # Only include disease-ontology ancestors (not generic BFO/OGMS roots)
                if label and any(ns in iri for ns in
                                 ("MONDO_", "DOID_", "HP_", "EFO_",
                                  "obo/MONDO", "obo/DOID", "obo/HP")):
                    results.append({
                        "text":         label,
                        "concept_iri":  iri,
                        "relationship": "is_a_ancestor",
                        "confidence":   0.70,
                        "source":       "OLS_ANCESTORS",
                    })
            if "_links" not in data or "next" not in data["_links"]:
                break
            page += 1
        return results

    # ── Source 5 — Synonyms from OLS search result ──────────────────────────
    # Synonyms from OLS are known aliases for the concept. Returning them as
    # implied conditions broadens the matching surface against denied terms.

    def _from_synonyms(self, concept: Dict) -> List[Dict]:
        results = []
        for syn in concept.get("synonyms", []):
            if isinstance(syn, str) and len(syn) > 4:
                results.append({
                    "text":         syn,
                    "concept_iri":  concept["iri"],
                    "relationship": "synonym",
                    "confidence":   0.80,
                    "source":       "OLS_SYNONYM",
                })
        return results

    # ── OLS response parsing — handles both OLS3 and OLS4 formats ─────────────

    @staticmethod
    def _parse_ols_search(data: dict) -> list:
        if "response" in data:
            return data["response"].get("docs", [])
        if "_embedded" in data:
            return data["_embedded"].get("terms", [])
        return []

    @staticmethod
    def _parse_ols_embedded(data: dict) -> list:
        if "_embedded" in data:
            return data["_embedded"].get("terms", [])
        if "response" in data:
            return data["response"].get("docs", [])
        return []

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(term: str) -> str:
        return re.sub(r"[^\w\s]", "", term.lower().strip())

    @staticmethod
    def _double_encode(iri: str) -> str:
        return urllib.parse.quote(urllib.parse.quote(iri, safe=""))

    @staticmethod
    def _deduplicate_sort(items: List[Dict]) -> List[Dict]:
        seen, unique = set(), []
        for item in sorted(items, key=lambda x: x["confidence"], reverse=True):
            key = item["text"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _cache_get(self, key: str):
        if not self.redis:
            return None
        try:
            val = self.redis.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    def _cache_set(self, key: str, value):
        if not self.redis or value is None:
            return
        try:
            self.redis.setex(key, CACHE_TTL, json.dumps(value))
        except Exception:
            pass
