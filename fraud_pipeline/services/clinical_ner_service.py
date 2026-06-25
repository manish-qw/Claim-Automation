"""
ClinicalNERService
Extracts denied medical conditions from any proposal structure.
Three passes: structured dict, dataclass fields, free text.
Falls back gracefully if scispaCy not installed.
"""
import re
import dataclasses
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# All surface forms that mean "patient denied this condition"
DENIAL_VALUES = {False, "no", "none", "never", "0", "negative",
                 "non", "absent", "nil", "not applicable", "na"}

# All negation patterns for free text — covers abbreviations,
# prefixes, adverbs — no hardcoded medical terms
NEGATION_PATTERNS = [
    r"no\s+(?:history\s+of\s+)?([a-z][a-z\s\-]{3,40})(?:\.|,|;|\n|$)",
    r"never\s+([a-z][a-z\s\-]{3,40})(?:\.|,|;|\n|$)",
    r"denies?\s+(?:any\s+)?([a-z][a-z\s\-]{3,40})(?:\.|,|;|\n|$)",
    r"non[- ]([a-z][a-z\s\-]{3,30})(?:\.|,|;|\n|$)",
    r"does\s+not\s+([a-z][a-z\s\-]{3,30})(?:\.|,|;|\n|$)",
    r"did\s+not\s+([a-z][a-z\s\-]{3,30})(?:\.|,|;|\n|$)",
    r"no\s+h/o\s+([a-z][a-z\s\-]{3,40})(?:\.|,|;|\n|$)",
    r"negative\s+for\s+([a-z][a-z\s\-]{3,40})(?:\.|,|;|\n|$)",
    r"absence\s+of\s+([a-z][a-z\s\-]{3,40})(?:\.|,|;|\n|$)",
    r"without\s+([a-z][a-z\s\-]{3,40})(?:\.|,|;|\n|$)",
]

# Free text fields that may contain health declarations
FREE_TEXT_FIELDS = [
    "health_declaration", "medical_history",
    "lifestyle_declaration", "additional_info",
    "remarks", "notes", "declaration",
]


class ClinicalNERService:

    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_sm")
            self.available = True
            logger.info("scispaCy loaded — clinical NER active")
        except Exception as e:
            self.nlp = None
            self.available = False
            logger.warning(
                "scispaCy not available (%s) — using regex fallback", e
            )

    # ── Public ────────────────────────────────────────────────────────────────

    def extract_denied_conditions(self, state) -> List[Dict]:
        denied = []
        denied.extend(self._pass1_structured_form(state))
        denied.extend(self._pass2_state_fields(state))
        denied.extend(self._pass3_free_text(state))
        return self._deduplicate(denied)

    def extract_entities(self, text: str) -> List[Dict]:
        if not text:
            return []
        if self.available:
            doc = self.nlp(text)
            entities = [
                {
                    "text": ent.text,
                    "normalized": ent.text.lower().strip(),
                    "label": ent.label_,
                }
                for ent in doc.ents
            ]
            # If scispaCy found nothing, treat whole text as entity
            if not entities:
                entities = self._raw_entity(text)
        else:
            entities = self._raw_entity(text)
        return entities

    # ── Pass 1 — structured proposal_form (Pydantic model or dict) ─────────────

    def _pass1_structured_form(self, state) -> List[Dict]:
        results = []
        proposal = state.proposal_form
        if proposal is None:
            return results

        # Convert Pydantic model → flat dict; fall back if already a dict
        if hasattr(proposal, "model_dump"):
            form_dict = proposal.model_dump()
        elif hasattr(proposal, "__dict__"):
            form_dict = vars(proposal)
        elif isinstance(proposal, dict):
            form_dict = proposal
        else:
            return results

        # Also merge additional_disclosures sub-dict if present
        extra = form_dict.pop("additional_disclosures", None) or {}
        if isinstance(extra, dict):
            form_dict.update(extra)

        for field_name, value in form_dict.items():
            # Skip non-scalar values (lists, nested dicts)
            if isinstance(value, (list, dict)):
                continue
            if self._is_denial(value):
                term = self._field_to_term(field_name)
                for ent in self.extract_entities(term):
                    results.append({
                        "text":       ent["text"],
                        "normalized": ent["normalized"],
                        "source":     "proposal_form",
                        "field":      field_name,
                    })
        return results

    # ── Pass 2 — ClaimState Pydantic fields starting with proposal_ ────────────
    # Automatically picks up any new fields added to ClaimState

    def _pass2_state_fields(self, state) -> List[Dict]:
        results = []
        # ClaimState is a Pydantic model — use model_fields
        try:
            field_names = list(state.model_fields.keys())
        except AttributeError:
            # Fallback for plain dataclasses
            try:
                import dataclasses
                field_names = [f.name for f in dataclasses.fields(state)]
            except TypeError:
                return results

        for fname in field_names:
            if not fname.startswith("proposal_"):
                continue
            if fname == "proposal_form":
                continue
            value = getattr(state, fname, None)
            if self._is_denial(value):
                term = fname.replace("proposal_", "").replace("_", " ").strip()
                for ent in self.extract_entities(term):
                    results.append({
                        "text":       ent["text"],
                        "normalized": ent["normalized"],
                        "source":     "state_field",
                        "field":      fname,
                    })
        return results

    # ── Pass 3 — free text negation detection ────────────────────────────────

    def _pass3_free_text(self, state) -> List[Dict]:
        results = []
        proposal = state.proposal_form
        if proposal is None:
            return results

        # Build a plain dict regardless of whether proposal is Pydantic or dict
        if hasattr(proposal, "model_dump"):
            form_dict = proposal.model_dump()
        elif isinstance(proposal, dict):
            form_dict = proposal
        else:
            form_dict = {}

        for field_name in FREE_TEXT_FIELDS:
            text = form_dict.get(field_name, "") or ""
            if not text or len(text) < 5:
                continue
            for phrase in self._extract_negated_phrases(text):
                for ent in self.extract_entities(phrase):
                    results.append({
                        "text":       ent["text"],
                        "normalized": ent["normalized"],
                        "source":     "free_text_negation",
                        "field":      field_name,
                    })
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_negated_phrases(self, text: str) -> List[str]:
        phrases = []
        text_lower = text.lower()

        if self.available:
            # scispaCy dependency parse — catches neg tokens
            doc = self.nlp(text)
            for token in doc:
                if token.dep_ == "neg":
                    for chunk in doc.noun_chunks:
                        if token.head in chunk:
                            phrases.append(chunk.text)
            # Also run regex on top — catches patterns dep parse misses
            for pattern in NEGATION_PATTERNS:
                for match in re.findall(pattern, text_lower):
                    phrases.append(match.strip())
        else:
            for pattern in NEGATION_PATTERNS:
                for match in re.findall(pattern, text_lower):
                    phrases.append(match.strip())

        return [p for p in phrases if len(p) > 3]

    @staticmethod
    def _is_denial(value) -> bool:
        if value is False:
            return True
        if isinstance(value, str):
            return value.lower().strip() in DENIAL_VALUES
        return False

    @staticmethod
    def _field_to_term(field_name: str) -> str:
        return field_name.replace("_", " ").replace("-", " ").strip()

    @staticmethod
    def _raw_entity(text: str) -> List[Dict]:
        return [{"text": text,
                 "normalized": text.lower().strip(),
                 "label": "UNKNOWN"}]

    @staticmethod
    def _deduplicate(items: List[Dict]) -> List[Dict]:
        seen, unique = set(), []
        for item in items:
            if item["normalized"] not in seen:
                seen.add(item["normalized"])
                unique.append(item)
        return unique
