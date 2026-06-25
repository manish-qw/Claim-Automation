"""
Agent 4 — Non-Disclosure Detection Agent

Pipeline:
  ClinicalNERService        → extract what patient denied from any proposal
  MedicalInferenceService   → extract what each diagnosis implies (ontology)
  NonDisclosureMatchService → find contradiction between the two lists

Fails safe — API failures never silently approve a claim.
"""
import logging
import re
from ..schemas import ClaimState, NonDisclosureOutput
from ..services.clinical_ner_service import ClinicalNERService
from ..services.llm_service import LLMService
import os
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.75


def _normalize_doc_type(doc_type: str) -> str:
    if not doc_type:
        return ""
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", str(doc_type))
    return s.lower().replace("-", "_").replace(" ", "_")


class NonDisclosureDetectionAgent:

    def __init__(self, redis_client=None):
        self.redis     = redis_client
        self.ner       = ClinicalNERService()
        self.llm       = LLMService(redis_client=redis_client)
        
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'agent4_nondisclosure.txt')
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

    def run(self, state: ClaimState) -> ClaimState:
        logger.info("[Agent 4] Non-Disclosure Detection → %s",
                    state.claim_case_id)

        # Step 1 — extract everything patient denied
        denied_conditions = self.ner.extract_denied_conditions(state)

        if not denied_conditions:
            logger.info("[Agent 4] No denied conditions found in proposal")
            state.non_disclosure_analysis = NonDisclosureOutput(
                contradiction_detected=False,
                non_disclosure_score=0.0,
                validation_flags=["NO_PROPOSAL_DECLARATIONS_FOUND"],
                inference_reasoning=(
                    "No denied conditions extractable from proposal — "
                    "cannot assess non-disclosure"
                ),
            ).model_dump()
            return state

        logger.info("[Agent 4] Denied conditions found: %s",
                    [d["text"] for d in denied_conditions])

        findings     = []
        max_score    = 0.0
        api_failures = []

        denied_str = ", ".join([d["text"] for d in denied_conditions])
        
        # Build a list of all text sources that might contain historical medical/lifestyle info
        records_to_check = []
        for record in state.medical_records:
            if not record.diagnosis and not record.content_summary:
                continue
            records_to_check.append({
                "date": record.date or record.admission_date or "UNKNOWN",
                "text": f"Diagnosis: {record.diagnosis or 'None'}\nSummary: {record.content_summary or 'None'}\nTreatment: {record.treatment or 'None'}",
                "obj": record,
                "is_medical": True
            })
            
        for fir in state.fir_records:
            summary = getattr(fir, "summary", "") or fir.description or fir.incident_description or ""
            if summary:
                records_to_check.append({
                    "date": fir.date_filed or fir.date or "UNKNOWN",
                    "text": f"Police FIR Summary: {summary}",
                    "obj": fir,
                    "is_medical": False
                })

        for doc in state.submitted_documents:
            # Check raw OCR for PMR, FIR, or medical reports
            doc_t = _normalize_doc_type(doc.doc_type or doc.document_type or "")
            if doc.ocr_text and doc_t in (
                "fir_copy",
                "fir",
                "first_information_report",
                "final_police_investigation_report",
                "medical_report",
                "medical_attendant_hospital_certificate",
                "medico_legal_cause_of_death_certificate",
                "pmr",
                "post_mortem",
                "postmortem_report",
                "discharge_summary",
                "past_medical_records_and_treatment_papers",
                "indoor_case_papers",
            ):
                records_to_check.append({
                    "date": str(doc.metadata.get("creation_date") or "UNKNOWN"),
                    "text": f"Raw Document Text ({doc.doc_type}): {doc.ocr_text[:1000]}",
                    "obj": doc,
                    "is_medical": False
                })

        # Step 2 — check every record text against denied conditions
        for item in records_to_check:
            record_text = item["text"]
            record_date = item["date"]
            record_obj = item["obj"]

            logger.info("[Agent 4] LLM checking record snippet: '%s'", record_text[:50].replace('\n', ' '))
            
            prompt = self.prompt_template.replace(
                "{denied_conditions_list}", denied_str
            ).replace(
                "{medical_record_diagnosis}", record_text
            ).replace(
                "{policy_issue_date}", state.policy_issue_date or "UNKNOWN"
            ).replace(
                "{medical_record_date}", record_date
            )
            
            # Call Gemini
            llm_result = self.llm.route_to_gemini(prompt, enforce_json=True)
            
            if not llm_result:
                logger.warning("[Agent 4] Gemini failed for '%s' — applying rule-based fallback", record_text[:50].replace('\n', ' '))
                api_failures.append(record_text[:50])

                if item["is_medical"]:
                    record = record_obj
                    # ── Rule-based fallback ──────────────────────────────────────
                    # Check structured boolean fields directly — no LLM needed
                    denied_lower = {d["normalized"] for d in denied_conditions}
                    rule_findings = []

                    # Alcohol non-disclosure
                    alcohol_denied = any(
                        kw in " ".join(denied_lower)
                        for kw in ("alcohol", "drink", "liquor")
                    )
                    if record.alcohol_history and alcohol_denied:
                        rule_findings.append(
                            "Proposal denied alcohol use — medical record confirms alcohol_history=True"
                        )
                        max_score = max(max_score, 0.85)

                    # Smoking non-disclosure
                    smoking_denied = any(
                        kw in " ".join(denied_lower)
                        for kw in ("smok", "tobacco", "cigarette")
                    )
                    if record.smoking_history and smoking_denied:
                        rule_findings.append(
                            "Proposal denied smoking — medical record confirms smoking_history=True"
                        )
                        max_score = max(max_score, 0.80)

                    # Chronic condition keyword match
                    chronic = " ".join(record.chronic_conditions or []).lower()
                    for denied in denied_lower:
                        if len(denied) > 4 and denied in chronic:
                            rule_findings.append(
                                f"Proposal denied '{denied}' — found in medical chronic_conditions"
                            )
                            max_score = max(max_score, 0.75)

                    if rule_findings:
                        logger.info("[Agent 4] Rule-based fallback found: %s", rule_findings)
                        findings.extend(rule_findings)
                continue
                
            if llm_result.get("contradiction_detected"):
                findings.extend(llm_result.get("findings", []))
                score = llm_result.get("non_disclosure_score", 0.0)
                max_score = max(max_score, score)
                logger.info(f"[Agent 4] Contradiction found: {llm_result.get('reasoning')}")

        # Step 5 — build output
        contradiction_detected = len(findings) > 0
        has_api_failures       = len(api_failures) > 0

        flags = []
        if has_api_failures:
            flags.append("ONTOLOGY_PARTIAL_FAILURE")
        if has_api_failures and not contradiction_detected:
            # Could not verify — do not auto-approve
            flags.append("MANUAL_REVIEW_RECOMMENDED")

        output = NonDisclosureOutput(
            contradiction_detected=contradiction_detected,
            non_disclosure_findings=findings,
            non_disclosure_score=round(max_score, 4),
            api_failed=has_api_failures,
            inference_reasoning=(
                "; ".join(findings)
                if findings
                else "No contradictions detected"
            ),
            validation_flags=flags,
            confidence_score=round(max_score, 4),
            trust_score=(
                round(1.0 - max_score, 4)
                if contradiction_detected
                else 1.0
            ),
        )

        if contradiction_detected:
            state.escalation_required = True
            state.validation_flags.append("NON_DISCLOSURE_DETECTED")

        state.non_disclosure_analysis = output.model_dump()
        return state
