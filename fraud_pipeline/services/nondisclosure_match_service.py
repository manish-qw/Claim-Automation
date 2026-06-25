"""
NonDisclosureMatchService
Compares denied conditions from proposal against implied conditions
from diagnosis. Four match levels in priority order.

Critical fix: substring match uses word boundary regex to prevent
false positives like "liver" matching "alcoholic liver disease"
when only liver disease was denied — not alcohol.
"""
import re
import json
import logging
import urllib.parse
import requests
from typing import List, Dict

logger = logging.getLogger(__name__)

CACHE_TTL = 2592000


class NonDisclosureMatchService:

    def __init__(self, redis_client=None, timeout: int = 10):
        self.redis   = redis_client
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept":     "application/json",
            "User-Agent": "ClaImosAI-FraudPipeline/1.0",
        })

    def find_contradiction(
        self,
        denied_conditions: List[Dict],
        implied_conditions: List[Dict],
    ) -> Dict:
        """
        Compares all denied vs all implied concepts.
        Returns first match found at highest confidence level.
        Stops at first match — caller iterates over all diagnoses.
        """
        for denied in denied_conditions:
            for implied in implied_conditions:

                # Level 1 — exact concept IRI match (strongest possible)
                if (denied.get("concept_iri")
                        and implied.get("concept_iri")
                        and denied["concept_iri"] == implied["concept_iri"]):
                    return self._result(denied, implied, "exact_iri", 0.97)

                d_norm = denied["normalized"]
                i_norm = implied["text"].lower().strip()

                # Level 2 — exact normalized text match
                if d_norm == i_norm:
                    return self._result(denied, implied, "text_match", 0.93)

                # Level 3 — word-boundary substring match
                # Uses \b to prevent "liver" matching "alcoholic liver disease"
                shorter = min(d_norm, i_norm, key=len)
                longer  = max(d_norm, i_norm, key=len)
                if (len(shorter) > 4
                        and re.search(rf"\b{re.escape(shorter)}\b", longer)):
                    return self._result(denied, implied, "substring_match", 0.85)

                # Level 4 — OLS synonym overlap
                if denied.get("concept_iri") and implied.get("concept_iri"):
                    syns_d = self._get_synonyms(denied["concept_iri"])
                    syns_i = self._get_synonyms(implied["concept_iri"])
                    if set(syns_d) & set(syns_i):
                        return self._result(denied, implied, "synonym_match", 0.88)

        return {
            "matched":    False,
            "confidence": 0.0,
            "reasoning":  "No contradiction found between proposal and medical records",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _result(
        self,
        denied:     Dict,
        implied:    Dict,
        match_type: str,
        confidence: float,
    ) -> Dict:
        return {
            "matched":           True,
            "match_type":        match_type,
            "confidence":        confidence,
            "denied_condition":  denied["text"],
            "implied_condition": implied["text"],
            "reasoning": (
                f"Diagnosis implies '{implied['text']}' "
                f"via {implied.get('source', 'unknown')} "
                f"({implied.get('relationship', '')}) "
                f"which contradicts declared denial of "
                f"'{denied['text']}' (field: {denied.get('field', 'unknown')}). "
                f"Match type: {match_type}, confidence: {confidence}"
            ),
        }

    def _get_synonyms(self, concept_iri: str) -> List[str]:
        cache_key = f"synonyms:v2:{hash(concept_iri)}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            encoded = urllib.parse.quote(urllib.parse.quote(concept_iri, safe=""))
            url = f"https://www.ebi.ac.uk/ols4/api/ontologies/mondo/terms/{encoded}"
            response = self.session.get(url, timeout=self.timeout)
            synonyms = [s.lower() for s in response.json().get("synonyms", [])]
            self._cache_set(cache_key, synonyms)
            return synonyms
        except Exception as e:
            logger.warning("[Match] Synonym lookup failed: %s", e)
            return []

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
