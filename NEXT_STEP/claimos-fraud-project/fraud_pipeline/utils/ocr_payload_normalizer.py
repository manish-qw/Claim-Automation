"""
Normalize OCR-centric document payloads into the pipeline ClaimState shape.

This keeps integration inside project code (no external adapter service) and
lets uploaded OCR JSONs flow through the existing agents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import re


_SCALAR_TYPES = (str, int, float, bool)
_FIELD_META_KEYS = {"value", "ocr_confidence", "extraction_confidence", "source_text"}
_NEGATIVE_TOKENS = {
    "no",
    "none",
    "na",
    "n/a",
    "nil",
    "negative",
    "absent",
    "false",
    "0",
    "never",
    "does not",
    "not known",
    "not available",
}


def _snake_doc_type(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", str(name))
    s = s.replace("-", "_").replace(" ", "_")
    return s.lower()


def _is_doc_object(v: Any) -> bool:
    if not isinstance(v, dict):
        return False
    return any(k in v for k in ("document_type", "DocumentType", "extracted_entities"))


def _extract_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, _SCALAR_TYPES):
        s = str(value).strip()
        return s if s else None
    if isinstance(value, dict):
        # OCR entity leafs are usually {value, ocr_confidence, extraction_confidence, source_text}
        if "value" in value:
            return _extract_scalar(value.get("value"))
        if "source_text" in value:
            return _extract_scalar(value.get("source_text"))
    return None


def _extract_field_with_meta(value: Any) -> Dict[str, Any]:
    """
    Extract a scalar field while preserving OCR/extraction/source metadata.
    """
    field = {
        "value": None,
        "ocr_confidence": None,
        "extraction_confidence": None,
        "source_text": None,
    }
    if value is None:
        return field

    if isinstance(value, dict) and "value" in value:
        field["value"] = _extract_scalar(value.get("value"))
        field["ocr_confidence"] = value.get("ocr_confidence")
        field["extraction_confidence"] = value.get("extraction_confidence")
        field["source_text"] = _extract_scalar(value.get("source_text"))
        return field

    scalar = _extract_scalar(value)
    field["value"] = scalar
    return field


def _walk_key_match(obj: Any, key_candidates: List[str]) -> Optional[str]:
    cand = {c.lower() for c in key_candidates}

    def _walk(x: Any) -> Optional[str]:
        if isinstance(x, dict):
            for k, v in x.items():
                if k.lower() in cand:
                    val = _extract_scalar(v)
                    if val:
                        return val
                hit = _walk(v)
                if hit:
                    return hit
        elif isinstance(x, list):
            for item in x:
                hit = _walk(item)
                if hit:
                    return hit
        return None

    return _walk(obj)


def _collect_field_array(source: Any) -> List[str]:
    out: List[str] = []
    if isinstance(source, list):
        for item in source:
            val = _extract_scalar(item)
            if val:
                out.append(val)
    elif isinstance(source, dict):
        for key in ("LowConfidenceFields", "UnmappedFields"):
            if isinstance(source.get(key), list):
                for item in source.get(key) or []:
                    val = _extract_scalar(item)
                    if val:
                        out.append(val)
    return out


def _collect_ocr_field_map(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Lossless map of OCR fields with confidence/source metadata.
    """
    root = doc.get("extracted_entities") if isinstance(doc.get("extracted_entities"), dict) else doc
    if not isinstance(root, dict):
        return {}

    field_map: Dict[str, Dict[str, Any]] = {}
    skip_keys = {
        "document_id",
        "document_type",
        "file_name",
        "uploaded_by",
        "uploaded_at",
        "ocr_confidence",
        "trust_score",
        "document_language",
        "pages",
        "is_handwritten",
        "is_blurry",
        "is_tampered",
        "verification_status",
        "validation_flags",
        "LowConfidenceFields",
        "UnmappedFields",
    }

    def _walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            if _FIELD_META_KEYS.issubset(node.keys()):
                meta = _extract_field_with_meta(node)
                if any(meta.get(k) is not None for k in ("value", "ocr_confidence", "extraction_confidence", "source_text")):
                    field_map[prefix] = meta
                return
            for key, value in node.items():
                if not prefix and key in skip_keys:
                    continue
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(value, next_prefix)
            return

        if isinstance(node, list):
            for idx, item in enumerate(node):
                next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                _walk(item, next_prefix)

    _walk(root, "")
    return field_map


def _collect_ocr_text(doc: Dict[str, Any]) -> str:
    chunks: List[str] = []

    def _walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            if "value" in obj:
                val = _extract_scalar(obj.get("value"))
                if val:
                    chunks.append(f"{prefix}: {val}" if prefix else val)
            else:
                for k, v in obj.items():
                    nxt = f"{prefix}.{k}" if prefix else k
                    _walk(nxt, v)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk(f"{prefix}[{idx}]", item)

    _walk("", doc.get("extracted_entities", {}))
    text = " | ".join(chunks)
    return text[:12000]


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_bool_from_text(x: Optional[str]) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("true", "yes", "y", "1", "present", "positive")


def _extract_docs_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload.get("submitted_documents"), list):
        docs = []
        for idx, d in enumerate(payload.get("submitted_documents", [])):
            if isinstance(d, dict):
                docs.append(d)
        return docs

    docs: List[Dict[str, Any]] = []

    docs_container = payload.get("documents")
    if isinstance(docs_container, dict):
        for key, value in docs_container.items():
            if _is_doc_object(value):
                d = dict(value)
                d.setdefault("_top_level_key", key)
                docs.append(d)
        if docs:
            return docs
    if isinstance(docs_container, list):
        for idx, value in enumerate(docs_container):
            if _is_doc_object(value):
                d = dict(value)
                d.setdefault("_top_level_key", f"documents[{idx}]")
                docs.append(d)
        if docs:
            return docs

    for k, v in payload.items():
        if _is_doc_object(v):
            d = dict(v)
            d.setdefault("_top_level_key", k)
            docs.append(d)
    return docs


def _norm_type_raw(doc: Dict[str, Any]) -> str:
    raw = (
        doc.get("document_type")
        or doc.get("DocumentType")
        or doc.get("_top_level_key")
        or ""
    )
    return str(raw)


def _make_submitted_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    raw_type = _norm_type_raw(doc)
    norm_type = _snake_doc_type(raw_type)
    extracted = doc.get("extracted_entities", {}) if isinstance(doc.get("extracted_entities"), dict) else {}
    search_root = extracted if extracted else doc

    # Pull optional technical fields if present
    hospital_rohini_id = (
        _walk_key_match(search_root, ["HospitalRohiniId", "RohiniId", "ROHINIID"])
        or doc.get("hospital_rohini_id")
    )
    doctor_reg = (
        _walk_key_match(
            search_root,
            ["DoctorRegistrationNumber", "RegistrationNumber", "NMRNumber", "DoctorRegNo"],
        )
        or doc.get("doctor_registration_number")
    )

    metadata: Dict[str, Any] = {}
    for key in ("creation_date", "mod_date", "creator", "producer", "pdf_creator", "pdf_producer"):
        if key in doc:
            metadata[key] = doc.get(key)

    metadata["ocr_field_map"] = _collect_ocr_field_map(doc)
    metadata["low_confidence_fields"] = _collect_field_array(doc.get("LowConfidenceFields")) or _collect_field_array(extracted.get("LowConfidenceFields"))
    metadata["unmapped_fields"] = _collect_field_array(doc.get("UnmappedFields")) or _collect_field_array(extracted.get("UnmappedFields"))
    metadata["document_trust_score"] = _to_float(doc.get("trust_score"), 0.0)
    metadata["is_handwritten"] = bool(doc.get("is_handwritten", False))
    metadata["is_blurry"] = bool(doc.get("is_blurry", False))
    metadata["is_tampered"] = bool(doc.get("is_tampered", False))
    metadata["verification_status"] = _extract_scalar(doc.get("verification_status")) or ""
    metadata["validation_flags"] = doc.get("validation_flags", [])
    metadata["raw_document"] = doc

    return {
        "doc_type": norm_type,
        "document_type": raw_type,
        "ocr_confidence": _to_float(doc.get("ocr_confidence"), 0.0),
        "ocr_text": doc.get("ocr_text") or _collect_ocr_text(doc),
        "metadata": metadata,
        "validation_flags": doc.get("validation_flags", []),
        "hospital_rohini_id": hospital_rohini_id,
        "doctor_registration_number": doctor_reg,
    }


def _to_optional_bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    if s in _NEGATIVE_TOKENS or any(token in s for token in (" not ", "none", "nil", "non smoker", "non-smoker")):
        return False
    if s in ("yes", "true", "y", "1", "present", "positive", "current", "active"):
        return True
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?", s):
        return float(s) > 0
    if any(ch.isdigit() for ch in s):
        nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", s)
        if nums:
            return any(float(n) > 0 for n in nums)
    return True


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%Y", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _collect_claim_core(docs_raw: List[Dict[str, Any]], docs_norm: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Search raw doc trees for semantic fields
    joined = {"docs": docs_raw}

    claimant_name = _walk_key_match(joined, ["ClaimantName", "Claimant", "Name"])
    claimant_rel = _walk_key_match(joined, ["RelationshipWithLifeAssured", "RelationshipWithDeceased", "Relationship"])
    bank_account = _walk_key_match(joined, ["AccountNumber"])
    bank_name = _walk_key_match(joined, ["BankName"])

    la_name = _walk_key_match(joined, ["LifeAssuredName", "LifeAssured", "Deceased", "Patient"])
    la_age = _walk_key_match(joined, ["Age"])
    la_occ = _walk_key_match(joined, ["Occupation", "OccupationCategory", "Designation"])
    sum_assured = _walk_key_match(joined, ["SumAssured"])

    date_of_death = _walk_key_match(joined, ["DateOfDeath", "DateOfDischargeOrDeath"])
    cause_of_death = _walk_key_match(
        joined,
        ["CauseOfDeath", "PrimaryCauseOfDeath", "ImmediateCause", "FinalOpinion"],
    )
    place_of_death = _walk_key_match(joined, ["PlaceOfDeath"])
    hospital_name = _walk_key_match(joined, ["HospitalName", "InstitutionName"])
    hospital_address = _walk_key_match(joined, ["HospitalAddress", "InstitutionAddress"])
    doctor_name = _walk_key_match(joined, ["DoctorName", "AttendingDoctor", "DiagnosingDoctorName"])
    doctor_reg = _walk_key_match(joined, ["DoctorRegistrationNumber", "RegistrationNumber"])
    fir_number = _walk_key_match(joined, ["FIRNumber"])
    death_category = _walk_key_match(joined, ["DeathCategory"])
    manner_of_death = _walk_key_match(joined, ["NatureOfDeath", "MannerOfDeath", "ModeOfDeath"])

    smoking_raw = _walk_key_match(
        joined,
        ["SmokingDuration", "SmokingQuantity", "TobaccoDuration", "TobaccoQuantity"],
    )
    alcohol_raw = _walk_key_match(
        joined,
        ["DrinkingDuration", "DrinkingQuantity", "AlcoholUse", "AlcoholConsumption"],
    )
    proposal_smoking = _to_optional_bool(smoking_raw)
    proposal_alcohol = _to_optional_bool(alcohol_raw)

    condition_candidates = [
        ("Cancer", "CancerHistory"),
        ("HeartDisease", "HeartDiseaseHistory"),
        ("LiverDisease", "LiverDiseaseHistory"),
        ("KidneyDisease", "KidneyDiseaseHistory"),
        ("LungDisease", "LungDiseaseHistory"),
        ("HypertensionOrDiabetes", "HypertensionOrDiabetesHistory"),
    ]
    proposal_pre_existing_conditions: List[str] = []
    for label, key in condition_candidates:
        raw = _walk_key_match(joined, [key])
        mark = _to_optional_bool(raw)
        if mark:
            proposal_pre_existing_conditions.append(label)
    other_history = _walk_key_match(joined, ["OtherDiseaseHistory"])
    if other_history:
        other_norm = str(other_history).strip()
        if other_norm and _to_optional_bool(other_norm) is not False:
            proposal_pre_existing_conditions.append(other_norm)

    # Promote direct technical fields if present in normalized docs
    if not doctor_reg:
        for d in docs_norm:
            if d.get("doctor_registration_number"):
                doctor_reg = d.get("doctor_registration_number")
                break

    return {
        "claimant": {
            "name": claimant_name or "",
            "relationship_to_life_assured": claimant_rel or "",
            "bank_account_number": bank_account or "",
            "bank_name": bank_name or "",
        },
        "life_assured": {
            "name": la_name or "",
            "age": int(float(la_age)) if la_age and str(la_age).replace(".", "", 1).isdigit() else 0,
            "occupation": la_occ or "",
            "sum_assured": _to_float(sum_assured, 0.0),
        },
        "death_information": {
            "date_of_death": date_of_death or "",
            "cause_of_death": cause_of_death or "",
            "place_of_death": place_of_death or "",
            "manner_of_death": manner_of_death or "",
            "hospital_name": hospital_name or "",
            "hospital_address": hospital_address or "",
            "doctor_name": doctor_name or "",
            "doctor_registration_number": doctor_reg or "",
            "fir_number": fir_number or "",
        },
        "death_category": death_category or "",
        "proposal_smoking": proposal_smoking,
        "proposal_alcohol_use": proposal_alcohol,
        "proposal_pre_existing_conditions": proposal_pre_existing_conditions,
    }


def _build_fir_records(docs_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in docs_raw:
        dt = _snake_doc_type(_norm_type_raw(d))
        if dt not in (
            "first_information_report",
            "final_police_investigation_report",
            "inquest_panchanana_report",
        ):
            continue
        row = {
            "fir_number": _walk_key_match(d, ["FIRNumber", "RegistrationNumber"]) or "",
            "police_station": _walk_key_match(d, ["PoliceStationName", "PoliceStation"]) or "",
            "date_filed": _walk_key_match(d, ["DateOfFIR", "DateOfRegistration"]) or "",
            "date": _walk_key_match(d, ["DateOfFIR", "DateOfRegistration"]) or "",
            "description": _walk_key_match(d, ["Description", "NatureOfIncident", "IncidentDescription"]) or "",
            "incident_description": _walk_key_match(d, ["Description", "NatureOfIncident", "IncidentDescription"]) or "",
            "location": _walk_key_match(d, ["Location", "PlaceOfIncident"]) or "",
        }
        if any(row.values()):
            rows.append(row)
    return rows


def _build_medical_records(docs_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    med_types = {
        "medical_attendant_hospital_certificate",
        "medico_legal_cause_of_death_certificate",
        "discharge_summary",
        "admission_form",
        "indoor_case_papers",
        "diagnostic_test_report",
        "past_medical_records_and_treatment_papers",
        "postmortem_report",
        "viscera_chemical_examination_report",
    }
    rows: List[Dict[str, Any]] = []
    for d in docs_raw:
        dt = _snake_doc_type(_norm_type_raw(d))
        if dt not in med_types:
            continue
        diagnosis = _walk_key_match(d, ["FinalDiagnosis", "ProvisionalDiagnosis", "ImmediateCause", "CauseOfDeath", "Diagnosis"]) or ""
        treatment = _walk_key_match(d, ["TreatmentGiven", "Treatment"]) or ""
        date = _walk_key_match(d, ["DateOfAdmission", "DateOfDischargeOrDeath", "Date", "DateOfReport"]) or ""
        doctor_name = _walk_key_match(d, ["DiagnosingDoctorName", "AttendingDoctor", "DoctorName"]) or ""
        smoke_raw = _walk_key_match(d, ["SmokingDuration", "SmokingQuantity"])
        alc_raw = _walk_key_match(d, ["DrinkingDuration", "DrinkingQuantity", "Alcohol"])
        chronic = []
        hist = _walk_key_match(d, ["ContributoryConditions", "History", "OtherDiseaseHistory"])
        if hist:
            chronic = [str(hist)]
        row = {
            "record_type": "pmr" if dt in ("postmortem_report", "viscera_chemical_examination_report") else "medical_report",
            "hospital_name": _walk_key_match(d, ["HospitalName", "InstitutionName"]) or "",
            "doctor_name": doctor_name,
            "diagnosis": diagnosis,
            "treatment": treatment,
            "content_summary": _collect_ocr_text(d),
            "admission_date": _walk_key_match(d, ["DateOfAdmission"]) or "",
            "discharge_date": _walk_key_match(d, ["DateOfDischargeOrDeath"]) or "",
            "date": date,
            "smoking_history": _to_bool_from_text(smoke_raw),
            "alcohol_history": _to_bool_from_text(alc_raw),
            "chronic_conditions": chronic,
        }
        if any(
            [
                row["diagnosis"],
                row["treatment"],
                row["content_summary"],
                row["date"],
                row["hospital_name"],
            ]
        ):
            rows.append(row)
    return rows


def normalize_claim_case_payload(claim_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize either:
      1) existing ClaimState-like payload (returned mostly as-is), or
      2) OCR document-bundle payload (top-level document objects)
    into ClaimState-compatible dict.
    """
    if not isinstance(claim_case, dict):
        return claim_case

    # Already in canonical claim shape
    canonical_keys = {"claimant", "life_assured", "death_information", "submitted_documents"}
    if any(k in claim_case for k in canonical_keys):
        # If submitted docs already exist, just ensure doc_type normalization for CamelCase names
        docs = claim_case.get("submitted_documents")
        if isinstance(docs, list):
            source_docs: List[Dict[str, Any]] = []
            for d in docs:
                if isinstance(d, dict):
                    raw = d.get("doc_type") or d.get("document_type") or ""
                    if raw and not d.get("doc_type"):
                        d["doc_type"] = _snake_doc_type(raw)
                    metadata = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
                    source_doc = metadata.get("raw_document") if isinstance(metadata.get("raw_document"), dict) else d
                    if isinstance(source_doc, dict):
                        source_docs.append(source_doc)
                        metadata.setdefault("ocr_field_map", _collect_ocr_field_map(source_doc))
                        metadata.setdefault("low_confidence_fields", _collect_field_array(source_doc.get("LowConfidenceFields")))
                        metadata.setdefault("unmapped_fields", _collect_field_array(source_doc.get("UnmappedFields")))
                        metadata.setdefault("raw_document", source_doc)
                    if d.get("ocr_text") in (None, "") and isinstance(source_doc, dict):
                        d["ocr_text"] = _collect_ocr_text(source_doc)
                    d["metadata"] = metadata

            if source_docs:
                joined = {"docs": source_docs}
                death_category = (claim_case.get("death_category") or _walk_key_match(joined, ["DeathCategory"]) or "").strip()
                if death_category:
                    claim_case["death_category"] = death_category
                    dc = death_category.upper()
                    if dc == "UNNATURAL":
                        claim_case.setdefault("death_type", "ACCIDENTAL")
                        claim_case.setdefault("incident_type", "ACCIDENTAL")
                    elif dc == "NATURAL_OR_MEDICAL":
                        claim_case.setdefault("death_type", "NATURAL")
                        claim_case.setdefault("incident_type", "NATURAL")

                if claim_case.get("proposal_smoking") is None:
                    smoking_raw = _walk_key_match(
                        joined,
                        ["SmokingDuration", "SmokingQuantity", "TobaccoDuration", "TobaccoQuantity"],
                    )
                    claim_case["proposal_smoking"] = _to_optional_bool(smoking_raw)

                if claim_case.get("proposal_alcohol_use") is None:
                    alcohol_raw = _walk_key_match(
                        joined,
                        ["DrinkingDuration", "DrinkingQuantity", "AlcoholUse", "AlcoholConsumption"],
                    )
                    claim_case["proposal_alcohol_use"] = _to_optional_bool(alcohol_raw)

                if not claim_case.get("proposal_pre_existing_conditions"):
                    conditions: List[str] = []
                    for label, key in (
                        ("Cancer", "CancerHistory"),
                        ("HeartDisease", "HeartDiseaseHistory"),
                        ("LiverDisease", "LiverDiseaseHistory"),
                        ("KidneyDisease", "KidneyDiseaseHistory"),
                        ("LungDisease", "LungDiseaseHistory"),
                        ("HypertensionOrDiabetes", "HypertensionOrDiabetesHistory"),
                    ):
                        mark = _to_optional_bool(_walk_key_match(joined, [key]))
                        if mark:
                            conditions.append(label)
                    other_history = _walk_key_match(joined, ["OtherDiseaseHistory"])
                    if other_history and _to_optional_bool(other_history) is not False:
                        conditions.append(str(other_history).strip())
                    claim_case["proposal_pre_existing_conditions"] = conditions

        if "months_since_inception" not in claim_case:
            issue_dt = _parse_date(str(claim_case.get("policy_issue_date") or ""))
            death_dt = _parse_date(
                str(
                    (claim_case.get("death_information") or {}).get("date_of_death")
                    if isinstance(claim_case.get("death_information"), dict)
                    else ""
                )
            )
            if issue_dt and death_dt:
                days = (death_dt - issue_dt).days
                if days >= 0:
                    claim_case["months_since_inception"] = int(days / 30)
        return claim_case

    docs_raw = _extract_docs_from_payload(claim_case)
    if not docs_raw:
        return claim_case

    docs_norm = [_make_submitted_doc(d) for d in docs_raw]
    core = _collect_claim_core(docs_raw, docs_norm)
    death_category = (claim_case.get("death_category") or core.get("death_category") or "").strip()

    ocr_conf = {}
    for d in docs_norm:
        dt = d.get("doc_type") or d.get("document_type") or "unknown_document"
        ocr_conf[dt] = _to_float(d.get("ocr_confidence"), 0.0)

    normalized: Dict[str, Any] = {
        "claim_case_id": claim_case.get("claim_case_id", ""),
        "policy_number": claim_case.get("policy_number", ""),
        "policy_issue_date": claim_case.get("policy_issue_date", ""),
        "policy_age_days": int(claim_case.get("policy_age_days", 0) or 0),
        "policy_revival_detected": bool(claim_case.get("policy_revival_detected", False)),
        "policy_sum_assured": _to_float(claim_case.get("policy_sum_assured", 0.0)),
        "policy_premium": _to_float(claim_case.get("policy_premium", 0.0)),
        "premium_payment_history": claim_case.get("premium_payment_history", []),
        "claimant": core["claimant"],
        "life_assured": core["life_assured"],
        "death_information": core["death_information"],
        "submitted_documents": docs_norm,
        "medical_records": _build_medical_records(docs_raw),
        "fir_records": _build_fir_records(docs_raw),
        "ocr_confidence_scores": ocr_conf,
        "proposal_smoking": claim_case.get("proposal_smoking", core.get("proposal_smoking")),
        "proposal_alcohol_use": claim_case.get("proposal_alcohol_use", core.get("proposal_alcohol_use")),
        "proposal_pre_existing_conditions": claim_case.get(
            "proposal_pre_existing_conditions",
            core.get("proposal_pre_existing_conditions", []),
        ),
    }
    if death_category:
        normalized["death_category"] = death_category

    # Carry optional policy-side fields if supplied
    for k in (
        "incident_type",
        "death_type",
        "months_since_inception",
        "is_policy_in_force",
        "premium_payment_type",
        "payout_option_chosen",
    ):
        if k in claim_case:
            normalized[k] = claim_case[k]

    # Map OCR death category to policy routing defaults when caller did not provide them.
    if death_category:
        dc = death_category.upper()
        if dc == "UNNATURAL":
            normalized.setdefault("death_type", "ACCIDENTAL")
            normalized.setdefault("incident_type", "ACCIDENTAL")
        elif dc == "NATURAL_OR_MEDICAL":
            normalized.setdefault("death_type", "NATURAL")
            normalized.setdefault("incident_type", "NATURAL")

    # Derive months_since_inception when policy/death dates are available.
    if "months_since_inception" not in normalized:
        issue_dt = _parse_date(str(normalized.get("policy_issue_date") or ""))
        death_dt = _parse_date(str(normalized.get("death_information", {}).get("date_of_death") or ""))
        if issue_dt and death_dt:
            days = (death_dt - issue_dt).days
            if days >= 0:
                normalized["months_since_inception"] = int(days / 30)

    return normalized
