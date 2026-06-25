"""
Agent 5 — Conflict Resolution Agent

Responsibilities:
  - Detect contradictions between FIR, PMR, Death Certificate, Medical Reports
  - Resolve medical term conflicts using MedicalRelationshipService (4-layer cascade)
  - Resolve non-medical conflicts using trust_score × confidence_score
  - Escalate on ties or unresolvable conflicts

Medical cause-of-death comparison is handled entirely by MedicalRelationshipService.
No hardcoded medical term lists anywhere in this file.
"""

import logging
from typing import Dict, List, Tuple, Optional

from ..schemas import ClaimState, ConflictResolutionOutput
from ..services.llm_service import LLMService
import os

logger = logging.getLogger(__name__)

# Document source trust priors (tuned from domain knowledge)
# Adjusted at runtime by OCR confidence and external verification
SOURCE_TRUST_PRIORS: Dict[str, float] = {
    "death_certificate":    0.85,
    "pmr":                  0.90,    # Post-Mortem Report — highest authority
    "medical_report":       0.80,
    "fir":                  0.75,
    "hospital_discharge":   0.78,
    "claim_form":           0.50,    # Self-reported; lowest prior
}


class ConflictResolutionAgent:
    """
    Resolves document-level conflicts using trust-weighted confidence.
    Uses MedicalRelationshipService for all medical term comparisons.
    Enriches ClaimState.conflict_resolution.
    """

    TIE_THRESHOLD = 0.05   # If scores within this range, escalate

    def __init__(self):
        # Instantiate once — not inside run() — so it can be shared across calls
        self._llm: Optional[LLMService] = None

        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'agent5_conflict.txt')
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = LLMService(redis_client=None)
        return self._llm

    def run(self, state: ClaimState) -> ClaimState:
        logger.info("[Agent 5] Conflict Resolution → claim %s", state.claim_case_id)

        # Build document evidence map
        evidence = self._build_evidence_map(state)
        conflicts: List[str] = []
        resolutions: List[Tuple[str, str, str]] = []  # (conflict, winner, reasoning)

        # Track MedicalRelationshipService results for output
        med_rel_found = False
        med_rel_type = ""
        med_rel_reasoning = ""
        med_rel_layer = 0

        for source_a, source_b, field in self._generate_conflict_pairs(evidence):
            val_a = evidence[source_a].get(field)
            val_b = evidence[source_b].get(field)

            if not val_a or not val_b:
                continue

            # ── Medical field → use MedicalRelationshipService ────────────────
            if field == "cause_of_death":
                # Fast path: if values already agree via string matching, no conflict
                if self._values_agree(val_a, val_b):
                    continue

                # Slow path: values differ — use LLM
                prompt = self.prompt_template.replace("{cause_A}", val_a).replace("{cause_B}", val_b)
                llm_result = self.llm.route_to_gemini(prompt, enforce_json=True)
                
                if not llm_result:
                    # Nothing could resolve — send to human
                    conflict_desc = (
                        f"{field}: {source_a} says '{val_a}' vs {source_b} says '{val_b}'"
                    )
                    conflicts.append(conflict_desc)
                    resolutions.append((conflict_desc, "MANUAL_REVIEW", "LLM API failed. Manual review required."))

                elif llm_result.get("are_compatible") and llm_result.get("confidence", 0.0) >= 0.70:
                    # Terms are medically related — not a genuine conflict
                    med_rel_found = True
                    med_rel_type = "llm_synonym"
                    med_rel_reasoning = llm_result.get("reasoning", "")
                    med_rel_layer = 1
                    logger.info(
                        "[Agent 5] Cause-of-death terms medically related: '%s' ↔ '%s'",
                        val_a, val_b,
                    )

                elif not llm_result.get("are_compatible"):
                    # Genuinely unrelated — flag as conflict
                    conflict_desc = (
                        f"{field}: {source_a} says '{val_a}' vs {source_b} says '{val_b}'"
                    )
                    conflicts.append(conflict_desc)
                    score_a = self._weighted_score(source_a, state)
                    score_b = self._weighted_score(source_b, state)
                    if abs(score_a - score_b) <= self.TIE_THRESHOLD:
                        resolutions.append((conflict_desc, "TIE", "ESCALATE — sources equally trusted"))
                    elif score_a > score_b:
                        resolutions.append((conflict_desc, source_a,
                                            f"{source_a} wins (score {score_a:.2f} > {score_b:.2f})"))
                    else:
                        resolutions.append((conflict_desc, source_b,
                                            f"{source_b} wins (score {score_b:.2f} > {score_a:.2f})"))

                else:
                    # Related but low confidence
                    conflict_desc = (
                        f"{field}: {source_a} says '{val_a}' vs {source_b} says '{val_b}' "
                        f"(low-confidence medical relationship: {llm_result.get('confidence', 0):.2f})"
                    )
                    conflicts.append(conflict_desc)
                    resolutions.append((conflict_desc, "MANUAL_REVIEW",
                                        f"Low confidence ({llm_result.get('confidence', 0):.2f}) — manual review"))

            # ── Non-medical field → standard value agreement check ─────────────
            else:
                if not self._values_agree(val_a, val_b):
                    conflict_desc = (
                        f"{field}: {source_a} says '{val_a}' vs {source_b} says '{val_b}'"
                    )
                    conflicts.append(conflict_desc)

                    score_a = self._weighted_score(source_a, state)
                    score_b = self._weighted_score(source_b, state)

                    if abs(score_a - score_b) <= self.TIE_THRESHOLD:
                        resolutions.append((conflict_desc, "TIE", "ESCALATE"))
                    elif score_a > score_b:
                        resolutions.append((conflict_desc, source_a,
                                            f"{source_a} wins (score {score_a:.2f} > {score_b:.2f})"))
                    else:
                        resolutions.append((conflict_desc, source_b,
                                            f"{source_b} wins (score {score_b:.2f} > {score_a:.2f})"))

        # ── Determine final action ─────────────────────────────────────────────
        conflict_detected = len(conflicts) > 0
        has_manual = any(r[1] == "MANUAL_REVIEW" for r in resolutions)
        has_tie = any(r[1] == "TIE" for r in resolutions)

        if not conflicts:
            resolved_action = "ACCEPT"
            winning_source = "CONSENSUS"
            reasoning = "No conflicts detected across documents"
            confidence = 0.92
        elif has_manual:
            resolved_action = "MANUAL_REVIEW"
            winning_source = "MANUAL_REVIEW"
            reasoning = "Medical relationship unresolvable via ontology — clinical review required"
            confidence = 0.50
        elif has_tie:
            resolved_action = "ESCALATE"
            winning_source = "ESCALATED"
            reasoning = "Tie detected — human review required"
            confidence = 0.60
        else:
            winners = [r[1] for r in resolutions if r[1] not in ("TIE", "MANUAL_REVIEW")]
            winning_source = max(set(winners), key=winners.count) if winners else "UNKNOWN"
            resolved_action = "ACCEPT" if all(
                r[1] not in ("TIE", "MANUAL_REVIEW") for r in resolutions
            ) else "ESCALATE"
            reasoning = "; ".join(r[2] for r in resolutions[:3])
            confidence = 0.78

        trust = 0.85 if resolved_action == "ACCEPT" else 0.45

        output = ConflictResolutionOutput(
            conflict_detected=conflict_detected,
            conflicts_found=conflicts,
            resolved_action=resolved_action,
            winning_source=winning_source,
            resolution_reasoning=reasoning,
            confidence_score=confidence,
            trust_score=trust,
            validation_flags=["CONFLICT_DETECTED"] if conflict_detected else [],
            # Medical Relationship Service output
            medical_relationship_found=med_rel_found,
            relationship_type=med_rel_type,
            ontology_reasoning=med_rel_reasoning,
            resolution_layer=med_rel_layer,
        )

        state.conflict_resolution = output.model_dump()
        if conflict_detected:
            state.validation_flags.append("CONFLICT_DETECTED")
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_evidence_map(self, state: ClaimState) -> Dict[str, Dict]:
        """Extract key fields from each document source."""
        evidence: Dict[str, Dict] = {}

        # Death certificate
        evidence["death_certificate"] = {
            "cause_of_death": state.death_information.cause_of_death,
            "date_of_death":  state.death_information.date_of_death,
            "hospital_name":  state.death_information.hospital_name,
        }

        # FIR — try date_filed first, fall back to date
        if state.fir_records:
            fir = state.fir_records[0]
            fir_date = fir.date_filed or fir.date or ""
            fir_cause = fir.description or fir.incident_description or ""
            evidence["fir"] = {
                "fir_number":    fir.fir_number,
                "date_of_death": fir_date,
                "hospital_name": "",
                "cause_of_death": fir_cause,
            }

        # Medical records / PMR — extract doctor name properly
        if state.medical_records:
            pmr = next(
                (r for r in state.medical_records
                 if r.record_type.lower() in ("pmr", "post_mortem", "post mortem")),
                None,
            )
            if pmr:
                evidence["pmr"] = {
                    "cause_of_death": pmr.diagnosis,
                    "date_of_death":  pmr.date,
                    "doctor_name":    pmr.doctor_name or pmr.treating_doctor or "",
                }
            else:
                med = state.medical_records[0]
                evidence["medical_report"] = {
                    "cause_of_death": med.diagnosis,
                    "date_of_death":  med.date,
                    "doctor_name":    med.doctor_name or med.treating_doctor or "",
                }

        return evidence

    def _generate_conflict_pairs(self, evidence: Dict):
        """Generate all source pairs that share a checkable field."""
        sources = list(evidence.keys())
        shared_fields = ["cause_of_death", "date_of_death", "hospital_name"]
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                for field in shared_fields:
                    if (field in evidence[sources[i]]
                            and evidence[sources[i]].get(field)
                            and field in evidence[sources[j]]
                            and evidence[sources[j]].get(field)):
                        yield sources[i], sources[j], field

    def _values_agree(self, val_a: str, val_b: str) -> bool:
        """
        Two values agree if identical or one is a substring of the other.
        Also handles date parsing to match formats like '12/05/2026' with '2026-05-12'.
        """
        a, b = val_a.lower().strip(), val_b.lower().strip()
        if a == b:
            return True
        if a in b or b in a:
            return True
        if len(a) >= 10 and len(b) >= 10 and a[:10] == b[:10]:
            return True
            
        # Try robust date parsing if simple string match fails
        from dateutil import parser
        try:
            # If the string contains a slash, assume DD/MM/YYYY (Indian format)
            # Otherwise use ISO standard (YYYY-MM-DD)
            date_a = parser.parse(a, dayfirst=("/" in a))
            date_b = parser.parse(b, dayfirst=("/" in b))
            
            # Allow a 1-day leeway for edge cases (e.g., late-night death vs next-day autopsy report)
            if abs((date_a.date() - date_b.date()).days) <= 1:
                return True
        except Exception:
            pass
            
        return False

    def _weighted_score(self, source: str, state: ClaimState) -> float:
        """Compute trust × OCR confidence for a document source."""
        base_trust = SOURCE_TRUST_PRIORS.get(source, 0.60)
        ocr_conf = state.ocr_confidence_scores.get(source, 1.0)
        ext = state.external_verification or {}
        ext_trust = ext.get("trust_score", 1.0)
        return round(base_trust * ocr_conf * ext_trust, 4)
