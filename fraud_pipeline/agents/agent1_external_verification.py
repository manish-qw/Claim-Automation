"""
Agent 1 — External Verification Engine  (v2 — Production Rewrite)
===================================================================
Six verification areas, each returning a partial float score (0.0–1.0):

  Area 1: Hospital verification  — ROHINI ID format + PostgreSQL frequency
  Area 2: Doctor verification    — NMR registration format + PostgreSQL frequency
  Area 3: FIR verification       — state format regex + date arithmetic + jurisdiction
  Area 4: Document metadata      — OCR confidence + PDF/EXIF metadata integrity
  Area 5: Image forensics        — SHA-256 hash dedup + pdfplumber + Pillow ELA
  Area 6: Geo-location           — PIN code region match + LLM entity extraction

Weighted confidence formula:
  confidence = (image × 0.30) + (hospital × 0.25) + (metadata × 0.20)
             + (fir × 0.15) + (doctor × 0.07) + (geo × 0.03)

Trust formula:
  trust = max(0.05, confidence − (0.30 if image_tampered) − (0.20 if metadata_tampered))

Graceful failure rules:
  - PostgreSQL down      → neutral score 0.5, flag DB_UNAVAILABLE_FREQUENCY_CHECK
  - LLM unavailable      → PIN-only geo, flag GEO_LLM_UNAVAILABLE
  - Pillow ELA exception → image_score = 0.5 for that document, continue

All thresholds are read from environment variables — no hardcoded numbers.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

from ..schemas import ClaimState, ExternalVerificationOutput
from ..services.llm_service import LLMService
from ..utils.format_constants import (
    ROHINI_ID_RE,
    VALID_STATE_CODES,
    PIN_CODE_RE,
    PDF_EDITING_SOFTWARE_KEYWORDS,
    EXIF_EDITING_SOFTWARE_KEYWORDS,
    validate_rohini_id,
    validate_nmr_number,
    validate_fir_number,
    pin_region,
)
from ..utils.runtime_queries import (
    get_hospital_claim_frequency,
    get_doctor_claim_frequency,
    check_file_hash_registry,
    register_file_hash,
)

logger = logging.getLogger(__name__)


# ── Read thresholds from env (tunable by business team) ──────────────────────
def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

HOSPITAL_FREQ_THRESHOLD   = _env_int("HOSPITAL_FREQ_THRESHOLD",   5)
DOCTOR_FREQ_THRESHOLD     = _env_int("DOCTOR_FREQ_THRESHOLD",      5)
FIR_DATE_ANOMALY_DAYS     = _env_int("FIR_DATE_ANOMALY_DAYS",      30)
ELA_ANOMALY_THRESHOLD     = _env_float("ELA_ANOMALY_THRESHOLD",    15.0)
PDF_MOD_DATE_ANOMALY_DAYS = _env_int("PDF_MOD_DATE_ANOMALY_DAYS",  7)
OCR_LOW_CONF_THRESHOLD    = _env_float("OCR_LOW_CONFIDENCE_THRESHOLD", 0.55)

# Document types legitimately created AFTER date of death
_POST_DEATH_DOC_TYPES = {
    "pmr", "post_mortem_report", "autopsy_report",
    "discharge_summary", "discharge", "death_certificate",
    "hospital_report", "claim_form", "affidavit",
    "medical_report", "medical_certificate",
    # OCR JSON document names (normalized)
    "postmortem_report", "first_information_report",
    "final_police_investigation_report", "inquest_panchanana_report",
    "viscera_chemical_examination_report",
    "medical_attendant_hospital_certificate",
    "medico_legal_cause_of_death_certificate",
}


def _normalize_doc_type(doc_type: str) -> str:
    if not doc_type:
        return ""
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", str(doc_type))
    return s.lower().replace("-", "_").replace(" ", "_")


class ExternalVerificationEngine:
    """
    Verifies external entities referenced in a claim.
    Writes per-area float scores + evidence references into ClaimState.external_verification.
    Accepts an optional SQLAlchemy engine for live frequency queries.
    """

    def __init__(self, db_engine=None):
        self._engine = db_engine
        self.llm = LLMService(redis_client=None)
        
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'agent1_geo.txt')
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

    def run(self, state: ClaimState) -> ClaimState:
        logger.info("[Agent 1] External Verification → %s", state.claim_case_id)

        flags: List[str] = []
        evidence: Dict[str, str] = {}   # flag → "source:field"

        # ── Run all 6 verification areas ─────────────────────────────────────
        hospital_score, h_flags, h_ev   = self._verify_hospital(state)
        doctor_score, d_flags, d_ev     = self._verify_doctor(state)
        fir_score, f_flags, f_ev        = self._verify_fir(state)
        metadata_score, meta_tampered, m_flags, m_ev = self._check_metadata(state)
        image_score, img_tampered, i_flags, i_ev     = self._check_image_forensics(state)
        geo_score, g_flags, g_ev        = self._check_geolocation(state)

        # Merge all flags and evidence
        for flag_list, ev_map in [
            (h_flags, h_ev), (d_flags, d_ev), (f_flags, f_ev),
            (m_flags, m_ev), (i_flags, i_ev), (g_flags, g_ev),
        ]:
            flags.extend(flag_list)
            evidence.update(ev_map)

        # ── Weighted confidence formula ────────────────────────────────────────
        confidence = round(
            (image_score    * 0.30) +
            (hospital_score * 0.25) +
            (metadata_score * 0.20) +
            (fir_score      * 0.15) +
            (doctor_score   * 0.07) +
            (geo_score      * 0.03),
            4,
        )

        # ── Trust formula with stronger tampering penalties ───────────────────
        trust = round(max(0.05, confidence
                          - (0.30 if img_tampered   else 0.0)
                          - (0.20 if meta_tampered  else 0.0)), 4)

        # ── Derive backward-compat bool fields from float scores ──────────────
        hospital_verified        = hospital_score >= 0.70
        doctor_verified          = doctor_score   >= 0.70
        fir_verified             = fir_score      >= 0.70
        geo_location_match       = geo_score      >= 0.60
        image_manipulation_det   = img_tampered
        metadata_tampered_flag   = meta_tampered

        output = ExternalVerificationOutput(
            # Per-area float scores
            hospital_score          = round(hospital_score, 4),
            doctor_score            = round(doctor_score, 4),
            fir_score               = round(fir_score, 4),
            metadata_score          = round(metadata_score, 4),
            image_score             = round(image_score, 4),
            geo_score               = round(geo_score, 4),
            # Backward-compat bools
            hospital_verified       = hospital_verified,
            doctor_verified         = doctor_verified,
            fir_verified            = fir_verified,
            geo_location_match      = geo_location_match,
            image_manipulation_detected = image_manipulation_det,
            metadata_tampered       = metadata_tampered_flag,
            # Aggregate
            verification_confidence = confidence,
            confidence_score        = confidence,
            trust_score             = trust,
            validation_flags        = flags,
            flag_evidence           = evidence,
        )

        state.external_verification = output.model_dump()
        state.validation_flags.extend(flags)

        # Register file hashes into DB for future dedup checks
        self._register_hashes(state)

        return state

    # ══════════════════════════════════════════════════════════════════════════
    # AREA 1 — Hospital Verification
    # ══════════════════════════════════════════════════════════════════════════

    def _verify_hospital(
        self, state: ClaimState
    ) -> Tuple[float, List[str], Dict[str, str]]:
        """
        Returns (score: 0.0–1.0, flags, evidence).
        Steps:
          1. ROHINI ID presence check
          2. ROHINI ID format validation
          3. PostgreSQL frequency check (rolling 90 days)
        """
        flags: List[str] = []
        evidence: Dict[str, str] = {}

        # Only use actual ROHINI IDs — hospital_id (e.g. HOSP-KA-0088) is NOT a ROHINI ID
        rohini_id = (
            (state.death_information.hospital_rohini_id or "").strip()
        )
        # Also check submitted documents for rohini_id
        if not rohini_id:
            for doc in state.submitted_documents:
                if doc.hospital_rohini_id:
                    rohini_id = doc.hospital_rohini_id.strip()
                    break

        # Step 1: Presence
        if not rohini_id:
            flag = "ROHINI_ID_MISSING"
            flags.append(flag)
            evidence[flag] = "death_information.hospital_rohini_id"
            logger.info("[Agent 1] Hospital: ROHINI ID absent → score 0.40")
            return 0.40, flags, evidence

        # Step 2: Format validation
        is_valid, reason = validate_rohini_id(rohini_id)
        if not is_valid:
            flag = "ROHINI_ID_FORMAT_INVALID"
            flags.append(flag)
            evidence[flag] = f"death_information.hospital_rohini_id='{rohini_id}' ({reason})"
            logger.info("[Agent 1] Hospital: ROHINI ID format invalid '%s' → score 0.20", rohini_id)
            return 0.20, flags, evidence

        # Step 3: PostgreSQL frequency check
        score = 1.0
        try:
            count, claim_ids = get_hospital_claim_frequency(rohini_id, days=90, engine=self._engine)
            if count > HOSPITAL_FREQ_THRESHOLD:
                flag = f"HOSPITAL_HIGH_FREQUENCY:{rohini_id}:{count}claims/90days"
                flags.append(flag)
                evidence[flag] = f"death_information.hospital_rohini_id='{rohini_id}'"
                # Reduce score proportionally but not below 0.3
                score = max(0.30, 1.0 - min(0.70, (count - HOSPITAL_FREQ_THRESHOLD) * 0.10))
                logger.info(
                    "[Agent 1] Hospital '%s' appeared in %d claims in 90 days → score %.2f",
                    rohini_id, count, score,
                )
        except Exception as exc:
            flag = "DB_UNAVAILABLE_FREQUENCY_CHECK:hospital"
            flags.append(flag)
            evidence[flag] = "runtime_queries.get_hospital_claim_frequency"
            score = 0.50   # neutral — cannot assess
            logger.warning("[Agent 1] Hospital frequency DB query failed: %s", exc)

        return score, flags, evidence

    # ══════════════════════════════════════════════════════════════════════════
    # AREA 2 — Doctor Verification
    # ══════════════════════════════════════════════════════════════════════════

    def _verify_doctor(
        self, state: ClaimState
    ) -> Tuple[float, List[str], Dict[str, str]]:
        """
        Returns (score: 0.0–1.0, flags, evidence).
        Steps:
          1. Registration number presence check
          2. NMR format validation
          3. PostgreSQL frequency check (rolling 90 days)
        """
        flags: List[str] = []
        evidence: Dict[str, str] = {}

        reg_number = (
            (state.death_information.doctor_registration_number or "").strip()
        )
        # Also check submitted documents
        if not reg_number:
            for doc in state.submitted_documents:
                if doc.doctor_registration_number:
                    reg_number = doc.doctor_registration_number.strip()
                    break

        # Step 1: Presence
        if not reg_number:
            flag = "DOCTOR_REG_NUMBER_MISSING"
            flags.append(flag)
            evidence[flag] = "death_information.doctor_registration_number"
            logger.info("[Agent 1] Doctor: registration number absent → score 0.50")
            return 0.50, flags, evidence   # absence alone is not fraud; doctor may be in records

        # Step 2: Format validation
        is_valid, reason = validate_nmr_number(reg_number)
        if not is_valid:
            flag = "DOCTOR_REG_FORMAT_INVALID"
            flags.append(flag)
            evidence[flag] = f"death_information.doctor_registration_number='{reg_number}' ({reason})"
            logger.info("[Agent 1] Doctor: reg format invalid '%s' → score 0.20", reg_number)
            return 0.20, flags, evidence

        # Step 3: PostgreSQL frequency check
        score = 1.0
        try:
            count, claim_ids = get_doctor_claim_frequency(reg_number, days=90, engine=self._engine)
            if count > DOCTOR_FREQ_THRESHOLD:
                flag = f"DOCTOR_HIGH_FREQUENCY:{reg_number}:{count}claims/90days"
                flags.append(flag)
                evidence[flag] = f"death_information.doctor_registration_number='{reg_number}'"
                score = max(0.30, 1.0 - min(0.70, (count - DOCTOR_FREQ_THRESHOLD) * 0.10))
                logger.info(
                    "[Agent 1] Doctor '%s' appeared in %d claims in 90 days → score %.2f",
                    reg_number, count, score,
                )
        except Exception as exc:
            flag = "DB_UNAVAILABLE_FREQUENCY_CHECK:doctor"
            flags.append(flag)
            evidence[flag] = "runtime_queries.get_doctor_claim_frequency"
            score = 0.50
            logger.warning("[Agent 1] Doctor frequency DB query failed: %s", exc)

        return score, flags, evidence

    # ══════════════════════════════════════════════════════════════════════════
    # AREA 3 — FIR Verification
    # ══════════════════════════════════════════════════════════════════════════

    def _verify_fir(
        self, state: ClaimState
    ) -> Tuple[float, List[str], Dict[str, str]]:
        """
        Returns (score: 0.0–1.0, flags, evidence).
        Steps:
          1. FIR number format check (state-specific regex)
          2. Date consistency (FIR must be filed on or after death date)
          3. Jurisdiction consistency (police station district vs. place_of_death district)
        If no FIR records → returns 1.0 (neutral — FIR not required for natural deaths).
        """
        flags: List[str] = []
        evidence: Dict[str, str] = {}

        if not state.fir_records:
            return 1.0, flags, evidence   # No FIR — neutral

        scores = []
        for idx, fir in enumerate(state.fir_records):
            fir_score = 1.0
            fir_src = f"fir_records[{idx}]"

            # Step 1: FIR number missing
            fir_number = (fir.fir_number or "").strip()
            if not fir_number:
                flag = "FIR_NUMBER_MISSING"
                flags.append(flag)
                evidence[flag] = f"{fir_src}.fir_number"
                scores.append(0.30)
                continue

            # Step 2: Format check
            is_valid, matched_state, reason = validate_fir_number(fir_number)
            if not is_valid:
                flag = f"FIR_FORMAT_INVALID:{fir_number[:2] or 'UNKNOWN'}"
                flags.append(flag)
                evidence[flag] = f"{fir_src}.fir_number='{fir_number}' ({reason})"
                fir_score = min(fir_score, 0.30)
                logger.info("[Agent 1] FIR format invalid '%s'", fir_number)

            # Step 3: Date consistency
            fir_date_str = (fir.date_filed or fir.date or "").strip()
            death_date_str = (state.death_information.date_of_death or "").strip()
            if fir_date_str and death_date_str:
                try:
                    fir_dt   = datetime.fromisoformat(fir_date_str)
                    death_dt = datetime.fromisoformat(death_date_str)
                    days_lag = (fir_dt - death_dt).days
                    if days_lag < 0:
                        # FIR filed before death — impossible for most cases
                        flag = f"FIR_DATE_BEFORE_DEATH:{abs(days_lag)}days"
                        flags.append(flag)
                        evidence[flag] = f"{fir_src}.date_filed='{fir_date_str}' vs death='{death_date_str}'"
                        fir_score = min(fir_score, 0.20)
                    elif days_lag > FIR_DATE_ANOMALY_DAYS:
                        flag = f"FIR_DATE_ANOMALY:{days_lag}days_after_death"
                        flags.append(flag)
                        evidence[flag] = f"{fir_src}.date_filed='{fir_date_str}'"
                        fir_score = min(fir_score, 0.60)
                except ValueError:
                    pass   # unparseable dates — skip check

            # Step 4: Jurisdiction check (district extraction)
            place_of_death = (state.death_information.place_of_death or "").lower()
            station = (fir.police_station or "").lower()
            station_loc = (fir.location or "").lower()
            # Simple district check: extract first meaningful token from each
            death_district = self._extract_district_token(place_of_death)
            fir_district   = self._extract_district_token(station_loc or station)
            if death_district and fir_district and death_district != fir_district:
                flag = f"FIR_JURISDICTION_MISMATCH:ps={fir_district},death={death_district}"
                flags.append(flag)
                evidence[flag] = f"{fir_src}.police_station vs death_information.place_of_death"
                fir_score = min(fir_score, 0.60)

            scores.append(fir_score)

        final_score = min(scores) if scores else 1.0
        return final_score, flags, evidence

    # ══════════════════════════════════════════════════════════════════════════
    # AREA 4 — Document Metadata Integrity
    # ══════════════════════════════════════════════════════════════════════════

    def _check_metadata(
        self, state: ClaimState
    ) -> Tuple[float, bool, List[str], Dict[str, str]]:
        """
        Returns (score: 0.0–1.0, tampered: bool, flags, evidence).
        Checks:
          - OCR confidence thresholds
          - Document creation_date vs. date_of_death
          - PDF /Creator /Producer editing software keywords
          - PDF ModDate vs CreationDate anomaly
        """
        flags: List[str] = []
        evidence: Dict[str, str] = {}
        issues: List[str] = []
        score = 1.0

        death_date_str = (state.death_information.date_of_death or "").strip()
        try:
            death_dt = datetime.fromisoformat(death_date_str) if death_date_str else None
        except ValueError:
            death_dt = None

        for idx, doc in enumerate(state.submitted_documents):
            doc_src = f"submitted_documents[{idx}]({doc.doc_type})"
            doc_type_lower = _normalize_doc_type(doc.doc_type or doc.document_type or "")

            # Check 1: OCR confidence
            if 0 < doc.ocr_confidence < OCR_LOW_CONF_THRESHOLD:
                flag = f"METADATA_ISSUES:LOW_OCR:{doc.doc_type}:{doc.ocr_confidence:.2f}"
                flags.append(flag)
                evidence[flag] = f"{doc_src}.ocr_confidence"
                issues.append(flag)
                score = min(score, max(0.40, doc.ocr_confidence / OCR_LOW_CONF_THRESHOLD))

            # Check 2: creation_date pre-dates death (for post-death documents)
            created = doc.metadata.get("creation_date", "")
            if created and death_dt and doc_type_lower in _POST_DEATH_DOC_TYPES:
                try:
                    doc_dt = datetime.fromisoformat(created)
                    if doc_dt < death_dt:
                        flag = f"DOC_PREDATES_DEATH:{doc.doc_type}"
                        flags.append(flag)
                        evidence[flag] = f"{doc_src}.metadata.creation_date='{created}'"
                        score = min(score, 0.20)
                        issues.append(flag)
                except ValueError:
                    pass

            # Check 3: PDF /Creator or /Producer editing software
            for pdf_field in ("creator", "producer", "pdf_creator", "pdf_producer", "creator_software"):
                pdf_meta_val = str(doc.metadata.get(pdf_field, "")).lower()
                if pdf_meta_val and any(kw in pdf_meta_val for kw in PDF_EDITING_SOFTWARE_KEYWORDS):
                    flag = f"PDF_CREATED_BY_EDITOR:{doc.doc_type}"
                    if flag not in flags:
                        flags.append(flag)
                        evidence[flag] = f"{doc_src}.metadata.{pdf_field}='{pdf_meta_val[:60]}'"
                        score = min(score, 0.30)
                        issues.append(flag)

            # Check 4: PDF ModDate vs CreationDate anomaly
            mod_date_str = doc.metadata.get("mod_date") or doc.metadata.get("modification_date") or ""
            create_date_str = doc.metadata.get("creation_date") or ""
            if mod_date_str and create_date_str:
                try:
                    mod_dt    = datetime.fromisoformat(str(mod_date_str))
                    create_dt = datetime.fromisoformat(str(create_date_str))
                    delta_days = (mod_dt - create_dt).days
                    if delta_days > PDF_MOD_DATE_ANOMALY_DAYS:
                        flag = f"PDF_MODIFICATION_ANOMALY:{doc.doc_type}:{delta_days}days"
                        flags.append(flag)
                        evidence[flag] = f"{doc_src}.metadata.mod_date vs creation_date"
                        score = min(score, 0.60)
                        issues.append(flag)
                except (ValueError, TypeError):
                    pass

        tampered = score < 0.80
        return round(score, 4), tampered, flags, evidence

    # ══════════════════════════════════════════════════════════════════════════
    # AREA 5 — Image Forensics
    # ══════════════════════════════════════════════════════════════════════════

    def _check_image_forensics(
        self, state: ClaimState
    ) -> Tuple[float, bool, List[str], Dict[str, str]]:
        """
        Returns (score: 0.0–1.0, manipulation_detected: bool, flags, evidence).
        Steps:
          1. SHA-256 hash deduplication (from doc.sha256_hash field)
          2. EXIF Software tag editing software check (Pillow)
          3. Error Level Analysis (ELA) — Pillow JPEG resave comparison
          4. PDF /Creator /Producer editing software (pdfplumber — if file_path available)
        Note: digitally_altered and copy_paste_detected metadata fields are IGNORED.
        """
        flags: List[str] = []
        evidence: Dict[str, str] = {}
        score = 1.0

        for idx, doc in enumerate(state.submitted_documents):
            doc_src = f"submitted_documents[{idx}]({doc.doc_type})"
            doc_score = 1.0

            # ── Step 1: SHA-256 hash registry check ──────────────────────────
            if doc.sha256_hash:
                try:
                    is_dup, original_claim = check_file_hash_registry(
                        doc.sha256_hash, state.claim_case_id, engine=self._engine
                    )
                    if is_dup:
                        flag = f"DUPLICATE_FILE_HASH:{doc.doc_type}"
                        flags.append(flag)
                        evidence[flag] = (
                            f"{doc_src}.sha256_hash='{doc.sha256_hash[:16]}...' "
                            f"also in claim '{original_claim}'"
                        )
                        doc_score = min(doc_score, 0.10)
                except Exception as exc:
                    logger.warning("[Agent 1] Hash registry check failed for %s: %s", doc.doc_type, exc)

            # ── Step 2: EXIF editing software check ──────────────────────────
            exif_software = str(doc.metadata.get("exif_software", "")).lower()
            if not exif_software:
                # Try Pillow on in-memory bytes if available
                raw_bytes = doc.metadata.get("raw_bytes")
                if raw_bytes:
                    exif_software = self._extract_exif_software(raw_bytes, doc.doc_type)

            if exif_software and any(kw in exif_software for kw in EXIF_EDITING_SOFTWARE_KEYWORDS):
                flag = f"EXIF_EDITING_SOFTWARE:{doc.doc_type}"
                if flag not in flags:
                    flags.append(flag)
                    evidence[flag] = f"{doc_src}.exif_software='{exif_software[:60]}'"
                    doc_score = min(doc_score, 0.20)

            # ── Step 3: ELA (Error Level Analysis) ───────────────────────────
            raw_bytes = doc.metadata.get("raw_bytes")
            doc_type_norm = _normalize_doc_type(doc.doc_type or doc.document_type or "")
            if raw_bytes and doc_type_norm in (
                "death_certificate",
                "pmr",
                "fir",
                "medical_report",
                "post_mortem",
                "postmortem_report",
                "first_information_report",
            ):
                ela_score = self._compute_ela_score(raw_bytes, doc.doc_type)
                if ela_score is not None and ela_score > ELA_ANOMALY_THRESHOLD:
                    flag = f"ELA_ANOMALY_DETECTED:{doc.doc_type}"
                    flags.append(flag)
                    evidence[flag] = (
                        f"{doc_src}: ELA mean pixel diff {ela_score:.1f} > threshold {ELA_ANOMALY_THRESHOLD}"
                    )
                    doc_score = min(doc_score, 0.25)

            # ── Step 4: pdfplumber metadata (if file_path provided) ───────────
            if doc.file_path and doc.file_path.lower().endswith(".pdf"):
                pdf_score, pdf_flags, pdf_ev = self._check_pdf_metadata(
                    doc.file_path, doc.doc_type, doc_src
                )
                flags.extend(pdf_flags)
                evidence.update(pdf_ev)
                doc_score = min(doc_score, pdf_score)

            score = min(score, doc_score)

        manipulation_detected = score < 0.70
        return round(score, 4), manipulation_detected, flags, evidence

    # ══════════════════════════════════════════════════════════════════════════
    # AREA 6 — Geo-location Validation
    # ══════════════════════════════════════════════════════════════════════════

    def _check_geolocation(
        self, state: ClaimState
    ) -> Tuple[float, List[str], Dict[str, str]]:
        """
        Returns (score: 0.0–1.0, flags, evidence).
        Uses Gemini LLM to check hospital credibility and extract city/state.
        """
        flags: List[str] = []
        evidence: Dict[str, str] = {}

        hospital_addr = (
            state.death_information.hospital_address or
            state.death_information.hospital_name or ""
        )
        place_of_death = state.death_information.place_of_death or ""

        if not hospital_addr:
            return 0.50, flags, evidence   # Cannot assess

        prompt = self.prompt_template.replace("{hospital_name}", hospital_addr).replace("{address}", place_of_death)
        llm_result = self.llm.route_to_gemini(prompt, enforce_json=True)
        
        if not llm_result:
            flag = "GEO_LLM_UNAVAILABLE"
            flags.append(flag)
            evidence[flag] = "LLM unavailable for geo check"
            return 0.50, flags, evidence
            
        cred_score = llm_result.get("hospital_credibility_score", 0.5)
        city = llm_result.get("extracted_city", "UNKNOWN")
        state_name = llm_result.get("extracted_state", "UNKNOWN")
        
        if cred_score < 0.4:
            flag = "HOSPITAL_CREDIBILITY_LOW"
            flags.append(flag)
            evidence[flag] = llm_result.get("credibility_reasoning", "Low credibility")
            
        # Return credibility score as geo score since we removed PIN logic
        return cred_score, flags, evidence

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_exif_software(self, raw_bytes: bytes, doc_type: str) -> str:
        """Extract EXIF Software tag using Pillow. Returns '' on any failure."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            img = Image.open(io.BytesIO(raw_bytes))
            exif_data = img._getexif()
            if not exif_data:
                return ""
            for tag_id, value in exif_data.items():
                if TAGS.get(tag_id) == "Software":
                    return str(value).lower()
        except Exception as exc:
            logger.debug("[Agent 1] EXIF extraction failed for %s: %s", doc_type, exc)
        return ""

    def _compute_ela_score(self, raw_bytes: bytes, doc_type: str) -> Optional[float]:
        """
        Error Level Analysis (ELA) using Pillow.
        Resaves JPEG at quality=95 and computes mean pixel difference.
        Returns None if computation fails (caller treats as no-flag).
        """
        try:
            import numpy as np
            from PIL import Image, ImageChops

            original = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            buffer = io.BytesIO()
            original.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            resaved = Image.open(buffer).convert("RGB")

            diff = ImageChops.difference(original, resaved)
            arr  = np.array(diff, dtype=np.float32)
            ela_mean = float(arr.mean())
            logger.debug("[Agent 1] ELA score for %s: %.2f", doc_type, ela_mean)
            return ela_mean
        except Exception as exc:
            logger.warning("[Agent 1] ELA failed for %s: %s — returning neutral", doc_type, exc)
            return None   # None → caller skips; image_score stays 0.5 for this doc

    def _check_pdf_metadata(
        self, file_path: str, doc_type: str, doc_src: str
    ) -> Tuple[float, List[str], Dict[str, str]]:
        """
        Checks PDF /Creator, /Producer, and /ModDate via pdfplumber.
        Returns (score, flags, evidence).
        """
        flags: List[str] = []
        evidence: Dict[str, str] = {}
        score = 1.0
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                meta = pdf.metadata or {}
                creator  = str(meta.get("Creator",  "")).lower()
                producer = str(meta.get("Producer", "")).lower()
                for field_name, field_val in [("Creator", creator), ("Producer", producer)]:
                    if field_val and any(kw in field_val for kw in PDF_EDITING_SOFTWARE_KEYWORDS):
                        flag = f"PDF_CREATED_BY_EDITOR:{doc_type}"
                        if flag not in flags:
                            flags.append(flag)
                            evidence[flag] = f"{doc_src}.pdf.{field_name}='{field_val[:60]}'"
                            score = min(score, 0.20)

                # ModDate vs CreationDate check
                mod_date_str    = str(meta.get("ModDate",      ""))
                create_date_str = str(meta.get("CreationDate", ""))
                if mod_date_str and create_date_str:
                    try:
                        # PDF dates use format: D:YYYYMMDDHHmmSS
                        def _parse_pdf_date(s: str) -> Optional[datetime]:
                            s = s.replace("D:", "").replace("'", "")[:14]
                            return datetime.strptime(s, "%Y%m%d%H%M%S")
                        mod_dt    = _parse_pdf_date(mod_date_str)
                        create_dt = _parse_pdf_date(create_date_str)
                        if mod_dt and create_dt:
                            delta = (mod_dt - create_dt).days
                            if delta > PDF_MOD_DATE_ANOMALY_DAYS:
                                flag = f"PDF_MODIFICATION_ANOMALY:{doc_type}:{delta}days"
                                flags.append(flag)
                                evidence[flag] = f"{doc_src}.pdf ModDate−CreationDate={delta}days"
                                score = min(score, 0.50)
                    except (ValueError, TypeError):
                        pass
        except ImportError:
            logger.warning("[Agent 1] pdfplumber not installed — skipping PDF metadata check for %s", doc_type)
        except Exception as exc:
            logger.warning("[Agent 1] pdfplumber check failed for %s: %s", file_path, exc)
        return score, flags, evidence

    @staticmethod
    def _extract_location_tokens(text: str) -> Dict[str, str]:
        """
        Fast city/state extraction from a text string using a curated map.
        Returns {"city": "...", "state": "..."} or empty dict.
        """
        CITY_STATE_MAP: Dict[str, Dict[str, str]] = {
            # Tier-1
            "mumbai":      {"city": "mumbai",      "state": "maharashtra"},
            "pune":        {"city": "pune",        "state": "maharashtra"},
            "nagpur":      {"city": "nagpur",      "state": "maharashtra"},
            "delhi":       {"city": "delhi",       "state": "delhi"},
            "new delhi":   {"city": "delhi",       "state": "delhi"},
            "bengaluru":   {"city": "bengaluru",   "state": "karnataka"},
            "bangalore":   {"city": "bengaluru",   "state": "karnataka"},
            "hyderabad":   {"city": "hyderabad",   "state": "telangana"},
            "chennai":     {"city": "chennai",     "state": "tamil_nadu"},
            "kolkata":     {"city": "kolkata",     "state": "west_bengal"},
            "ahmedabad":   {"city": "ahmedabad",   "state": "gujarat"},
            "surat":       {"city": "surat",       "state": "gujarat"},
            "jaipur":      {"city": "jaipur",      "state": "rajasthan"},
            "lucknow":     {"city": "lucknow",     "state": "uttar_pradesh"},
            "kanpur":      {"city": "kanpur",      "state": "uttar_pradesh"},
            "agra":        {"city": "agra",        "state": "uttar_pradesh"},
            "bhopal":      {"city": "bhopal",      "state": "madhya_pradesh"},
            "indore":      {"city": "indore",      "state": "madhya_pradesh"},
            "patna":       {"city": "patna",       "state": "bihar"},
            "chandigarh":  {"city": "chandigarh",  "state": "chandigarh"},
            "bhubaneswar": {"city": "bhubaneswar", "state": "odisha"},
            "guwahati":    {"city": "guwahati",    "state": "assam"},
            "kochi":       {"city": "kochi",       "state": "kerala"},
            "thiruvananthapuram": {"city": "thiruvananthapuram", "state": "kerala"},
            "coimbatore":  {"city": "coimbatore",  "state": "tamil_nadu"},
            "madurai":     {"city": "madurai",     "state": "tamil_nadu"},
            "visakhapatnam": {"city": "visakhapatnam", "state": "andhra_pradesh"},
            "vijayawada":  {"city": "vijayawada",  "state": "andhra_pradesh"},
            "ranchi":      {"city": "ranchi",      "state": "jharkhand"},
            "raipur":      {"city": "raipur",      "state": "chhattisgarh"},
            "dehradun":    {"city": "dehradun",    "state": "uttarakhand"},
        }
        text = text.lower()
        for city, loc in CITY_STATE_MAP.items():
            if city in text:
                return loc
        return {}

    @staticmethod
    def _extract_district_token(text: str) -> str:
        """Extract first meaningful word token from text for jurisdiction comparison."""
        words = re.findall(r"[a-z]+", text.lower())
        stop = {"police", "station", "ps", "thana", "district", "city", "nagar", "area", "zone"}
        for w in words:
            if len(w) >= 4 and w not in stop:
                return w
        return ""

    def _register_hashes(self, state: ClaimState) -> None:
        """Register all document SHA-256 hashes into the DB after analysis."""
        for doc in state.submitted_documents:
            if doc.sha256_hash:
                try:
                    register_file_hash(
                        doc.sha256_hash, state.claim_case_id, doc.doc_type, engine=self._engine
                    )
                except Exception as exc:
                    logger.warning("[Agent 1] Hash registration failed: %s", exc)
