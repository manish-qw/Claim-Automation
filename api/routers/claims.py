import os
# Force PaddleOCR to not explode CPU/RAM thread counts and avoid OneDNN C++ crashes
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import io
from typing import List, Dict, Any, Tuple, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
import time
import asyncio
import threading
import queue
from PIL import Image
import numpy as np
import cv2
import json
import uuid
import re
import httpx
from openai import OpenAI
from api.face_verification import (
    is_face_status_terminal,
    run_stage1_face_verification,
    should_run_stage1_face_verification,
)
from api.validation_rules import validate_document

FACE_ONLY_DOC_TYPES = {"CLAIMANT_RECENT_PHOTOGRAPH"}

SCHEMA_METADATA_DEFAULTS: Dict[str, Any] = {
    "document_id": "",
    "document_type": "",
    "file_name": "",
    "uploaded_by": "",
    "uploaded_at": "",
    "ocr_confidence": 0,
    "trust_score": 0,
    "document_language": "",
    "pages": 0,
    "is_handwritten": False,
    "is_blurry": False,
    "is_tampered": False,
    "verification_status": "",
    "validation_flags": [],
}
SCHEMA_METADATA_FIELDS = list(SCHEMA_METADATA_DEFAULTS.keys())

DOC_TYPE_MAPPING = {
    "CLAIMANT_STATEMENT_FORM": "ClaimantStatementForm",
    "DEATH_CERTIFICATE": "DeathCertificate",
    "AADHAAR_CARD": "AadhaarCard",
    "PASSPORT": "Passport",
    "DRIVING_LICENCE": "DrivingLicence",
    "VOTER_ID": "VoterID",
    "PAN_CARD": "PANCard",
    "BANK_PROOF": "BankProof",
    
    # Step 2 Natural / Medical Docs
    "MEDICO_LEGAL_CERT": "MedicoLegalCauseOfDeathCertificate",
    "HOSPITALIZATION_RECORDS": "DischargeSummary",
    "TREATING_DOCTOR_CERT": "PastMedicalRecordsAndTreatmentPapers",
    "HOSPITAL_ATTENDANT_CERT": "MedicalAttendantHospitalCertificate",
    "EMPLOYER_CERT": "EmployerCertificate",
    
    # Step 2 Unnatural Docs
    "FIR": "FirstInformationReport",
    "INQUEST_REPORT": "InquestPanchananaReport",
    "FINAL_POLICE_REPORT": "FinalPoliceInvestigationReport",
    "POSTMORTEM_REPORT": "PostmortemReport",
    "VISCERA_REPORT": "VisceraChemicalExaminationReport",
    "NEWSPAPER_CUTTING": "NewspaperCutting",
    "DRIVING_LICENCE_STEP2": "DrivingLicence"
}

DOC_TYPE_CANONICAL_ALIASES: Dict[str, str] = {
    "DEATHCERTIFICATE": "DEATH_CERTIFICATE",
    "CLAIMANTSTATEMENTFORM": "CLAIMANT_STATEMENT_FORM",
    "AADHAARCARD": "AADHAAR_CARD",
    "AADHARCARD": "AADHAAR_CARD",
    "PASSPORT": "PASSPORT",
    "DRIVINGLICENCE": "DRIVING_LICENCE",
    "DRIVINGLICENSE": "DRIVING_LICENCE",
    "VOTERID": "VOTER_ID",
    "PANCARD": "PAN_CARD",
    "BANKPROOF": "BANK_PROOF",
    "MEDICOLEGALCERT": "MEDICO_LEGAL_CERT",
    "HOSPITALIZATIONRECORDS": "HOSPITALIZATION_RECORDS",
    "TREATINGDOCTORCERT": "TREATING_DOCTOR_CERT",
    "HOSPITALATTENDANTCERT": "HOSPITAL_ATTENDANT_CERT",
    "EMPLOYERCERT": "EMPLOYER_CERT",
    "FIR": "FIR",
    "INQUESTREPORT": "INQUEST_REPORT",
    "FINALPOLICEREPORT": "FINAL_POLICE_REPORT",
    "POSTMORTEMREPORT": "POSTMORTEM_REPORT",
    "VISCERAREPORT": "VISCERA_REPORT",
    "NEWSPAPERCUTTING": "NEWSPAPER_CUTTING",
    "CLAIMANTRECENTPHOTOGRAPH": "CLAIMANT_RECENT_PHOTOGRAPH",
}

STAGE1_INTERNAL_DOC_TYPES = {
    "CLAIMANT_STATEMENT_FORM",
    "DEATH_CERTIFICATE",
    "AADHAAR_CARD",
    "PASSPORT",
    "DRIVING_LICENCE",
    "VOTER_ID",
    "PAN_CARD",
    "BANK_PROOF",
    "IDENTITY_PROOF",
    "ADDRESS_PROOF",
}

STAGE2_INTERNAL_DOC_TYPES = {
    "MEDICO_LEGAL_CERT",
    "HOSPITALIZATION_RECORDS",
    "TREATING_DOCTOR_CERT",
    "HOSPITAL_ATTENDANT_CERT",
    "EMPLOYER_CERT",
    "FIR",
    "INQUEST_REPORT",
    "FINAL_POLICE_REPORT",
    "POSTMORTEM_REPORT",
    "VISCERA_REPORT",
    "NEWSPAPER_CUTTING",
    "DRIVING_LICENCE_STEP2",
}

STAGE2_NATURAL_DOC_TYPES = {
    "medical_attendant_hospital_certificate",
    "employer_certificate",
    "medico_legal_cause_of_death_certificate",
    "discharge_summary",
    "admission_form",
    "indoor_case_papers",
    "diagnostic_test_report",
    "past_medical_records_and_treatment_papers",
}

STAGE2_UNNATURAL_DOC_TYPES = {
    "first_information_report",
    "inquest_panchanana_report",
    "final_police_investigation_report",
    "postmortem_report",
    "viscera_chemical_examination_report",
    "newspaper_cutting",
}


def _normalize_doc_type_token(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", token).upper()


def canonicalize_doc_type(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw in DOC_TYPE_MAPPING or raw in FACE_ONLY_DOC_TYPES:
        return raw

    upper = raw.upper()
    normalized_sep = re.sub(r"[\s\-]+", "_", upper)
    if normalized_sep in DOC_TYPE_MAPPING or normalized_sep in FACE_ONLY_DOC_TYPES:
        return normalized_sep

    flattened = _normalize_doc_type_token(raw)
    if flattened in DOC_TYPE_CANONICAL_ALIASES:
        return DOC_TYPE_CANONICAL_ALIASES[flattened]

    # Allow schema-style names (e.g. DeathCertificate, PanCard)
    for internal_key, schema_name in DOC_TYPE_MAPPING.items():
        if _normalize_doc_type_token(schema_name) == flattened:
            return internal_key

    return raw


def normalize_doc_type_list(values: List[str]) -> List[str]:
    expanded: List[str] = []
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        # Swagger sometimes sends list as one CSV field: "DeathCertificate,PanCard"
        if "," in text:
            parts = [part.strip() for part in text.split(",") if part.strip()]
            expanded.extend(parts)
        else:
            expanded.append(text)
    return [canonicalize_doc_type(item) for item in expanded]


def is_stage1_internal_doc_type(value: Any) -> bool:
    doc_type = canonicalize_doc_type(value)
    return doc_type in STAGE1_INTERNAL_DOC_TYPES


def is_stage2_internal_doc_type(value: Any) -> bool:
    doc_type = canonicalize_doc_type(value)
    return doc_type in STAGE2_INTERNAL_DOC_TYPES

CLASSIFIER_PROMPT_TEMPLATE = """You are a document classifier. 
Given this OCR text, identify what type of document this is.

Return ONLY this JSON:
{{
  "detected_type": "<AadhaarCard|Passport|DrivingLicence|VoterID|PANCard|DeathCertificate|ClaimantStatementForm|MedicoLegalCauseOfDeathCertificate|DischargeSummary|PastMedicalRecordsAndTreatmentPapers|MedicalAttendantHospitalCertificate|EmployerCertificate|FirstInformationReport|InquestPanchananaReport|FinalPoliceInvestigationReport|PostmortemReport|VisceraChemicalExaminationReport|NewspaperCutting|Unknown>",
  "confidence": <0.0 to 1.0>,
  "reason": "<one line why>"
}}

OCR Text:
{OCR_TEXT}
"""

PROMPT_TEMPLATE = """You are a precise document data extraction engine for insurance claim processing. 
Your output will be used for autonomous claim verification where accuracy is critical.

## YOUR TASK
Extract structured data from OCR-extracted text of a {DOCUMENT_TYPE} document.
Return ONLY a valid JSON object. No explanation, no markdown, no preamble, no trailing text.

## INPUT FORMAT
You will receive two inputs:
1. CLEAN_TEXT: The OCR extracted text with special characters removed, words in sequence.
2. CONF_TEXT: The same text with per-word OCR confidence scores in parentheses.

Use CONF_TEXT to compute ocr_confidence for each field.
Use CLEAN_TEXT for actual value extraction.

## OUTPUT FORMAT
Every field in the schema must appear in your output.
Standard metadata fields must follow the primitive/list type shown in the schema:
document_id, document_type, file_name, uploaded_by, uploaded_at,
ocr_confidence, trust_score, document_language, pages, is_handwritten,
is_blurry, is_tampered, verification_status, validation_flags.

Every document content field must be an object with exactly these four keys:

{{
  "value": <extracted value as string, or null if not found>,
  "ocr_confidence": <float 0-100, average OCR confidence of source words, or null if not found>,
  "extraction_confidence": <float 0.0-1.0, YOUR confidence this mapping is correct>,
  "source_text": <the exact raw substring from CONF_TEXT you used, or null if not found>
}}

For array fields (e.g. PolicyNumbers, OtherInsurancePolicies), return a JSON array 
where each element follows the same four-key structure for scalar values, 
or is an object of four-key structures for object arrays.

## EXTRACTION RULES

1. NEVER invent or hallucinate values. If a field is not present in the text, 
   set value to null, ocr_confidence to null, extraction_confidence to 0.0, 
   source_text to null.

2. VALUE NORMALIZATION:
   - Dates: always output as DD/MM/YYYY. If only month/year visible, output MM/YYYY.
   - Phone numbers: strip spaces and dashes, keep country code if present.
   - Names: Title Case. Remove extra spaces.
   - Boolean fields (Yes/No, checkboxes): output as true/false.
   - Enum/checkbox fields: output the selected option label exactly as listed in the schema/options.
   - PAN: uppercase, no spaces.
   - Pincode: 6 digits as string, no spaces.
   - Aadhaar: output masked if masked in source (XXXX XXXX 1234 format).

3. OCR CONFIDENCE CALCULATION:
   - Find the words in CONF_TEXT that correspond to the extracted value.
   - Average their confidence percentages.
   - Round to 2 decimal places.
   - If the value spans multiple OCR words, average all of them.
   - Example: "DEATH (99.98%) CERTIFICATE (99.98%)" → ocr_confidence: 99.98

4. EXTRACTION CONFIDENCE GUIDANCE:
   - 0.95–1.0 : Field label clearly present, value unambiguous, OCR confidence high.
   - 0.80–0.94: Field found but value slightly ambiguous (e.g. partially cut off, low OCR).
   - 0.60–0.79: Field inferred from context, not explicitly labeled.
   - 0.40–0.59: Multiple possible values, best guess chosen.
   - 0.0–0.39 : Very uncertain. Flag for human review.
   Set extraction_confidence independently of ocr_confidence. 
   A field can have high OCR confidence but low extraction confidence 
   (clear text, wrong field mapping) and vice versa.

5. HANDLING AMBIGUITY:
   - If two fields could share the same source text (e.g. a name appears in both 
     claimant and life assured sections), use surrounding context and field labels 
     to disambiguate.
   - If genuinely ambiguous, pick the most likely mapping and set 
     extraction_confidence below 0.65.

6. UNMAPPED FIELDS:
   Populate the UnmappedFields array with any information present in the document 
   that does not map to any schema field. Each entry must follow this structure:
   {{
     "key": <descriptive field name you assign>,
     "value": <extracted value>,
     "ocr_confidence": <float>,
     "extraction_confidence": <float>,
     "source_text": <raw substring>,
     "reason": <one sentence: why this did not map to any schema field>
   }}
   Only include genuinely present text. Do NOT add entries for absent information.

7. CAUSE OF DEATH CLASSIFICATION (applies to Claim Form and Death Certificate only):
   In addition to extracting CauseOfDeath and NatureOfDeath as text values, 
   you must populate the field "DeathCategory" with one of exactly two values:

   "NATURAL_OR_MEDICAL" — if the death was due to illness, disease, or medical 
   condition, whether at home or hospital. Examples: cancer, heart failure, 
   fever, organ failure, old age.

   "UNNATURAL" — if the death involved any external cause. Examples: road accident, 
   rail accident, air accident, fall, drowning, murder, homicide, suicide, 
   electrocution, burns, poisoning.

   If the cause of death is not mentioned or is completely illegible, 
   set DeathCategory value to null and extraction_confidence to 0.0.
   
   This classification must be based on CauseOfDeath and NatureOfDeath text 
   extracted from the document. Do not guess from other fields.

8. LOW CONFIDENCE FLAGGING:
   Populate the "LowConfidenceFields" array (top level, alongside UnmappedFields) 
   with the key path of any field where EITHER:
   - ocr_confidence is below 70.0, OR
   - extraction_confidence is below 0.65
   Example: ["LifeAssured.DateOfDeath", "Claimant.PAN", "Medical.CancerDiagnosisDate"]
   This allows downstream systems to instantly know which fields need human review 
   without scanning the full JSON.

9. RELATIONSHIP PREFIX RULE (CRITICAL FOR INDIAN DOCUMENTS):
   Indian documents encode family relationships via prefixes before names.
   You MUST read the prefix to determine which field the name belongs to:
   - S/O or Son of      → FatherName
   - D/O or Daughter of → FatherName
   - W/O or Wife of     → SpouseName
   - H/O or Husband of  → SpouseName
   - C/O or Care of     → GuardianName
   - M/O or Mother of   → MotherName
   If source_text contains "W/O: Mangeram", the name "Mangeram" maps to 
   SpouseName, NOT FatherName. FatherName must be null in this case.
   Never assign a name to a field whose prefix contradicts that field.

10. ADDRESS PREFIX RULE:
   - Vill. or Village → VillageTownCity
   - Po. or P.O.      → PostOffice
   - Dist.            → District
   - Pin              → Pincode
   - Mob. or Ph.      → MobileNumber (differentiate from LandlineNumber)

## SCHEMA
The target schema is below. Extract only the fields defined here 
(plus UnmappedFields and LowConfidenceFields).

{SCHEMA}

## INPUT

CLEAN_TEXT:
{CLEAN_TEXT}

CONF_TEXT:
{CONF_TEXT}

## OUTPUT
Return only the completed JSON. Start your response with {{ and end with }}.
"""

CHUNK_PROMPT_TEMPLATE = """Extract structured data from this {DOCUMENT_TYPE} OCR chunk.
Return ONLY valid JSON matching the schema. No markdown or explanation.

Rules:
- Use only CURRENT_CHUNK_TEXT for extraction. PREVIOUS_CONTEXT is only to resolve split labels.
- If a schema field is absent in CURRENT_CHUNK_TEXT, return null for that field.
- Do not infer missing values from outside this chunk.
- Metadata fields keep the primitive/list type shown in the schema.
- For document content scalar fields, return {{"value": string|null, "ocr_confidence": number|null, "extraction_confidence": number, "source_text": string|null}}.
- Use CONF_TEXT to estimate ocr_confidence.
- Keep arrays in page order and include only values present in this chunk.

Schema:
{SCHEMA}

Chunk: {CHUNK_INDEX}/{CHUNK_TOTAL}
Pages: {PAGE_RANGE}

PREVIOUS_CONTEXT:
{PREVIOUS_CONTEXT}

CURRENT_CHUNK_TEXT:
{CLEAN_TEXT}

CONF_TEXT:
{CONF_TEXT}
"""

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Try importing OCR libraries
try:
    import pytesseract
    # Configure Tesseract path for Windows
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_RAPIDOCR = True
except ImportError:
    HAS_RAPIDOCR = False

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

router = APIRouter()

# Hugging Face API Setup
hf_keys = []
for i in range(1, 6):
    k = os.getenv(f"HF_API_KEY_{i}")
    if k and k.strip():
        hf_keys.append(k.strip())

hf_key_index = 0
hf_key_lock = threading.Lock()

def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except Exception:
        return default

def parse_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except Exception:
        return default

def parse_csv_env(name: str, default: str = "") -> List[str]:
    value = os.getenv(name, default)
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

def is_transient_llm_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    transient_markers = [
        "timeout",
        "timed out",
        "connection",
        "temporarily",
        "service unavailable",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "gateway",
        "network",
    ]
    return any(marker in message for marker in transient_markers)

def get_next_hf_key():
    global hf_key_index
    if not hf_keys:
        return None
    with hf_key_lock:
        key = hf_keys[hf_key_index]
        hf_key_index = (hf_key_index + 1) % len(hf_keys)
        return key

def clean_json_response(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content

def estimate_llm_tokens(text: str) -> int:
    # Cheap approximation; good enough for routing without adding tokenizer latency.
    return max(1, (len(text or "") + 3) // 4)

def normalize_compare_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

def is_extraction_field_object(value: Any) -> bool:
    return isinstance(value, dict) and "value" in value and (
        "ocr_confidence" in value or "extraction_confidence" in value or "source_text" in value
    )

def field_has_value(field: Any) -> bool:
    if is_extraction_field_object(field):
        value = field.get("value")
    else:
        value = field
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True

def extraction_confidence_of(field: Any) -> float:
    if not isinstance(field, dict):
        return 0.0
    try:
        return float(field.get("extraction_confidence") or 0.0)
    except Exception:
        return 0.0

def stable_json_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return str(value)

def normalize_ground_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

def is_value_grounded(value: Any, raw_text: str) -> bool:
    if value is None:
        return True
    value_text = str(value).strip()
    if not value_text:
        return True
    normalized_value = normalize_ground_text(value_text)
    if len(normalized_value) < 3:
        return True
    return normalized_value in normalize_ground_text(raw_text)

def should_ground_value(field_path: str, value: Any) -> bool:
    if value is None:
        return False
    value_text = str(value).strip()
    if len(normalize_ground_text(value_text)) < 4:
        return False
    path = field_path.lower()
    critical_markers = [
        "aadhaar", "account", "amount", "application", "certificate", "date",
        "ifsc", "micr", "number", "pan", "pincode", "policy", "registration",
        "uid", "vid",
    ]
    return bool(re.search(r"\d", value_text)) or any(marker in path for marker in critical_markers)

def tail_lines(text: str, count: int) -> str:
    if count <= 0:
        return ""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines[-count:])

def fast_classify_identity_document(ocr_text: str) -> Optional[Dict[str, Any]]:
    text = f" {ocr_text or ''} "
    lower_text = text.lower()
    upper_text = text.upper()
    compact_lower = re.sub(r"[^a-z0-9]+", " ", lower_text)

    candidates = {
        "AadhaarCard": {"score": 0, "reasons": []},
        "PANCard": {"score": 0, "reasons": []},
        "Passport": {"score": 0, "reasons": []},
        "DrivingLicence": {"score": 0, "reasons": []},
        "VoterID": {"score": 0, "reasons": []},
    }

    def add(schema_key: str, points: int, reason: str) -> None:
        candidates[schema_key]["score"] += points
        candidates[schema_key]["reasons"].append(reason)

    aadhaar_keywords = ["aadhaar", "aadhar","uidai","unique identification authority of india", "your aadhaar no","aadhaar no","vid"]
    if any(keyword in compact_lower for keyword in aadhaar_keywords):
        add("AadhaarCard", 4, "aadhaar_keyword")
    if re.search(r"\bvirtual\s+id\b", compact_lower) or re.search(r"\bvid\b", compact_lower):
        add("AadhaarCard", 3, "aadhaar_vid_keyword")
    if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text):
        add("AadhaarCard", 2, "aadhaar_12_digit_pattern")
    if re.search(r"(virtual\s+id|vid)\D{0,12}\d{4}\s?\d{4}\s?\d{4}\s?\d{4}", compact_lower):
        add("AadhaarCard", 4, "aadhaar_vid_16_digit_pattern")

    pan_keywords = ["permanent account number", "income tax department", "income tax", "pan card"]
    if any(keyword in compact_lower for keyword in pan_keywords):
        add("PANCard", 4, "pan_keyword")
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", upper_text):
        add("PANCard", 3, "pan_number_pattern")

    passport_keywords = ["passport", "pass port", "republic of india", "passport no", "passport number"]
    if any(keyword in compact_lower for keyword in passport_keywords):
        add("Passport", 4, "passport_keyword")
    if re.search(r"\bP<IND", upper_text) or re.search(r"\b[A-Z][0-9]{7}\b", upper_text):
        add("Passport", 2, "passport_number_or_mrz_pattern")

    dl_keywords = [
        "driving licence",
        "driving license",
        "driver licence",
        "driver license",
        "transport department",
        "licence no",
        "license no",
        "dl no",
        "d l no",
    ]
    if any(keyword in compact_lower for keyword in dl_keywords):
        add("DrivingLicence", 4, "driving_licence_keyword")
    if re.search(r"\b[A-Z]{2}\s?\d{2}\s?\d{4}\s?\d{6,8}\b", upper_text):
        add("DrivingLicence", 2, "driving_licence_number_pattern")

    voter_keywords = [
        "voter id",
        "election commission",
        "elector photo identity card",
        "elector",
        "epic no",
        "epic number",
    ]
    if any(keyword in compact_lower for keyword in voter_keywords):
        add("VoterID", 4, "voter_keyword")
    if re.search(r"\b[A-Z]{3}[0-9]{7}\b", upper_text):
        add("VoterID", 2, "epic_number_pattern")

    ranked = sorted(
        candidates.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )
    best_key, best_data = ranked[0]
    second_score = ranked[1][1]["score"] if len(ranked) > 1 else 0
    best_score = best_data["score"]

    if best_score < 4 or best_score - second_score < 2:
        return None

    confidence = min(0.98, 0.70 + (best_score * 0.04))
    return {
        "detected_type": best_key,
        "confidence": round(confidence, 2),
        "reason": ",".join(best_data["reasons"]),
        "source": "local_keyword_classifier",
    }

SCHEMA_FILES = [
    "document_schemas.json",
    "document_schemas_natural_death.json",
    "document_schemas_unnatural_death.json",
]
CANONICAL_CROSS_VERIFY_MAP_FILE = os.path.join("api", "cross_verify_field_map.json")
CANONICAL_CROSS_VERIFY_STAGE2_MAP_FILE = os.path.join("api", "cross_verify_field_map_stage2.json")
_SCHEMAS_CACHE: Dict[str, Any] = {}
_SCHEMAS_CACHE_READY = False
_SCHEMAS_CACHE_LOCK = threading.Lock()
_CROSS_VERIFY_MAP_CACHE: Dict[str, Any] = {}
_CROSS_VERIFY_MAP_READY = False
_CROSS_VERIFY_MAP_LOCK = threading.Lock()
_CROSS_VERIFY_STAGE2_MAP_CACHE: Dict[str, Any] = {}
_CROSS_VERIFY_STAGE2_MAP_READY = False
_CROSS_VERIFY_STAGE2_MAP_LOCK = threading.Lock()

def get_combined_schemas() -> Dict[str, Any]:
    global _SCHEMAS_CACHE_READY, _SCHEMAS_CACHE
    if _SCHEMAS_CACHE_READY:
        return _SCHEMAS_CACHE

    with _SCHEMAS_CACHE_LOCK:
        if _SCHEMAS_CACHE_READY:
            return _SCHEMAS_CACHE

        merged: Dict[str, Any] = {}
        for schema_file in SCHEMA_FILES:
            try:
                with open(os.path.join("api", schema_file), "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    merged.update(file_data)
            except Exception as e:
                print(f"Error loading {schema_file}: {e}")

        _SCHEMAS_CACHE = merged
        _SCHEMAS_CACHE_READY = True
        print(f"[SCHEMA CACHE] loaded={len(_SCHEMAS_CACHE)}")
        return _SCHEMAS_CACHE


def get_canonical_cross_verify_map() -> Dict[str, Any]:
    global _CROSS_VERIFY_MAP_CACHE, _CROSS_VERIFY_MAP_READY
    if _CROSS_VERIFY_MAP_READY:
        return _CROSS_VERIFY_MAP_CACHE

    with _CROSS_VERIFY_MAP_LOCK:
        if _CROSS_VERIFY_MAP_READY:
            return _CROSS_VERIFY_MAP_CACHE
        try:
            with open(CANONICAL_CROSS_VERIFY_MAP_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                loaded = {}
        except Exception as exc:
            print(f"[CROSS VERIFY MAP] load_error={exc}")
            loaded = {}
        loaded.setdefault("fields", {})
        loaded.setdefault("documents", {})
        loaded.setdefault("version", 0)
        _CROSS_VERIFY_MAP_CACHE = loaded
        _CROSS_VERIFY_MAP_READY = True
        print(
            "[CROSS VERIFY MAP] "
            f"fields={len(_CROSS_VERIFY_MAP_CACHE.get('fields', {}))} "
            f"documents={len(_CROSS_VERIFY_MAP_CACHE.get('documents', {}))}"
        )
        return _CROSS_VERIFY_MAP_CACHE


def get_stage2_cross_verify_map() -> Dict[str, Any]:
    global _CROSS_VERIFY_STAGE2_MAP_CACHE, _CROSS_VERIFY_STAGE2_MAP_READY
    if _CROSS_VERIFY_STAGE2_MAP_READY:
        return _CROSS_VERIFY_STAGE2_MAP_CACHE

    with _CROSS_VERIFY_STAGE2_MAP_LOCK:
        if _CROSS_VERIFY_STAGE2_MAP_READY:
            return _CROSS_VERIFY_STAGE2_MAP_CACHE
        try:
            with open(CANONICAL_CROSS_VERIFY_STAGE2_MAP_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                loaded = {}
        except Exception as exc:
            print(f"[CROSS VERIFY STAGE2 MAP] load_error={exc}")
            loaded = {}
        loaded.setdefault("natural_death", {})
        loaded.setdefault("unnatural_death", {})
        for key in ("natural_death", "unnatural_death"):
            node = loaded.get(key)
            if not isinstance(node, dict):
                node = {}
                loaded[key] = node
            node.setdefault("fields", {})
            node.setdefault("documents", {})
        loaded.setdefault("version", 0)
        _CROSS_VERIFY_STAGE2_MAP_CACHE = loaded
        _CROSS_VERIFY_STAGE2_MAP_READY = True
        print(
            "[CROSS VERIFY STAGE2 MAP] "
            f"natural_docs={len(_CROSS_VERIFY_STAGE2_MAP_CACHE.get('natural_death', {}).get('documents', {}))} "
            f"unnatural_docs={len(_CROSS_VERIFY_STAGE2_MAP_CACHE.get('unnatural_death', {}).get('documents', {}))}"
        )
        return _CROSS_VERIFY_STAGE2_MAP_CACHE

def clone_json_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return value

def strip_schema_prompt_comments(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_schema_prompt_comments(item) for item in value]
    if isinstance(value, dict):
        removable_keys = re.compile(r"^_(?:comment\d*|bucket|format|note|required_critical|required_important)$")
        return {
            key: strip_schema_prompt_comments(item)
            for key, item in value.items()
            if not removable_keys.match(str(key))
        }
    return value

def prepare_schema_for_extraction(schema_key: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    clean_schema = strip_schema_prompt_comments(clone_json_value(schema))
    prepared: Dict[str, Any] = {}
    for field in SCHEMA_METADATA_FIELDS:
        if field == "document_type":
            prepared[field] = clean_schema.get(field) or schema_key
        else:
            prepared[field] = clean_schema.get(field, clone_json_value(SCHEMA_METADATA_DEFAULTS[field]))

    for key, value in clean_schema.items():
        if key in SCHEMA_METADATA_FIELDS or key in {"LowConfidenceFields", "UnmappedFields"}:
            continue
        prepared[key] = value

    prepared["LowConfidenceFields"] = clean_schema.get("LowConfidenceFields", [])
    prepared["UnmappedFields"] = clean_schema.get("UnmappedFields", [])
    return prepared

def calculate_document_ocr_confidence(pages_result: List[Dict[str, Any]]) -> float:
    total_words = 0
    weighted_confidence = 0.0
    for page in pages_result or []:
        metrics = page.get("metrics") or {}
        try:
            word_count = int(metrics.get("word_count") or len(page.get("words") or []))
            confidence = float(metrics.get("overall_confidence"))
        except Exception:
            continue
        if word_count <= 0:
            continue
        total_words += word_count
        weighted_confidence += confidence * word_count
    return round(weighted_confidence / total_words, 2) if total_words > 0 else 0.0

def apply_backend_schema_metadata(
    payload: Dict[str, Any],
    schema_key: str,
    filename: str,
    request_id: str,
    pages_result: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ocr_confidence = calculate_document_ocr_confidence(pages_result)
    payload["document_id"] = request_id or ""
    payload["document_type"] = schema_key or payload.get("document_type") or ""
    payload["file_name"] = filename or payload.get("file_name") or ""
    payload["uploaded_by"] = payload.get("uploaded_by") if isinstance(payload.get("uploaded_by"), str) else ""
    payload["uploaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["ocr_confidence"] = ocr_confidence
    payload["trust_score"] = round(ocr_confidence / 100.0, 4) if ocr_confidence else 0.0
    payload["document_language"] = payload.get("document_language") if isinstance(payload.get("document_language"), str) else ""
    payload["pages"] = len(pages_result or [])
    for key in ("is_handwritten", "is_blurry", "is_tampered"):
        payload[key] = payload.get(key) if isinstance(payload.get(key), bool) else False
    payload["verification_status"] = payload.get("verification_status") if isinstance(payload.get("verification_status"), str) else "EXTRACTED"
    if not payload["verification_status"]:
        payload["verification_status"] = "EXTRACTED"
    payload["validation_flags"] = payload.get("validation_flags") if isinstance(payload.get("validation_flags"), list) else []
    return payload

# Lazy loaded OCR instances
RAPID_OCR_INSTANCE = None
ocr_init_lock = threading.Lock()
try:
    _rapid_concurrency = int(os.getenv("RAPID_OCR_MAX_CONCURRENCY", "2").strip())
except Exception:
    _rapid_concurrency = 2
RAPID_OCR_MAX_CONCURRENCY = max(1, min(_rapid_concurrency, 8))
rapid_inference_lock = threading.Semaphore(RAPID_OCR_MAX_CONCURRENCY)
LLM_MAX_CONCURRENCY = max(1, min(parse_int_env("LLM_MAX_CONCURRENCY", 3), 8))
llm_request_lock = threading.Semaphore(LLM_MAX_CONCURRENCY)

PIPELINE_BATCHES: Dict[str, Any] = {}
PIPELINE_LOCK = threading.Lock()
OCR_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
LLM_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
VALIDATION_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
FACE_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
PIPELINE_WORKERS_STARTED = False
PIPELINE_WORKERS_LOCK = threading.Lock()
PIPELINE_TERMINAL_STATUSES = {"success", "failed", "validation_error"}

# ---------------------------------------------------------------------------
# Auto-trigger: Fraud + Policy pipeline runs automatically when a batch
# completes.  Runs in its own daemon thread — does NOT block the OCR response.
# ---------------------------------------------------------------------------

# def _run_fraud_and_policy_background(batch_id: str, claim_payload: Dict[str, Any]) -> None:
#     """Background thread: normalize OCR payload → FraudPipeline → Policy orchestrator."""
#     try:
#         import sys, os
#         sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#         _policy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'policy_agent'))
#         if _policy_dir not in sys.path:
#             sys.path.insert(0, _policy_dir)

#         from fraud_pipeline.utils.ocr_payload_normalizer import normalize_claim_case_payload
#         from fraud_pipeline.schemas import ClaimState
#         from fraud_pipeline.pipeline import FraudPipeline

#         print(f"[AUTO-ANALYSIS START] batch={batch_id}")

#         # Stage 1 — normalise OCR payload into ClaimState
#         normalized = normalize_claim_case_payload(claim_payload)
#         state = ClaimState(**normalized)

#         # Stage 2 — 7-agent fraud pipeline
#         package = FraudPipeline().run(state)
#         fraud_result = package.model_dump()
#         print(f"[AUTO-ANALYSIS FRAUD DONE] batch={batch_id} recommendation={package.final_recommendation}")

#         # Stage 3 — Policy orchestrator (LangGraph)
#         policy_result: Dict[str, Any] = {}
#         try:
#             from mock_data import get_initial_state   # bare import — policy_dir on sys.path
#             from orchestrator import app as policy_app

#             # Build policy payload from fraud output
#             docs_verified = [
#                 f"{d.doc_type} (ocr_conf: {d.ocr_confidence})"
#                 for d in state.submitted_documents
#             ]
#             fraud_payload = {
#                 "claim_case_id": package.claim_case_id,
#                 "status": "success",
#                 "result": {
#                     "claim_case_id": package.claim_case_id,
#                     "claim_profile": {
#                         "policy_id":                 state.policy_number,
#                         "claimant_name":             state.claimant.name,
#                         "life_assured_name":         state.life_assured.name,
#                         "life_assured_age_at_entry": state.life_assured.age,
#                         "incident_type":             normalized.get("incident_type", "NATURAL"),
#                         "incident_date":             state.death_information.date_of_death,
#                         "cause_of_death":            state.death_information.cause_of_death,
#                         "death_type":                normalized.get("death_type", "NATURAL"),
#                         "months_since_inception":    normalized.get("months_since_inception", 0),
#                         "is_policy_in_force":        normalized.get("is_policy_in_force", True),
#                         "premium_payment_type":      normalized.get("premium_payment_type", "REGULAR"),
#                         "sum_assured":               state.policy_sum_assured,
#                         "annualised_premium":        state.policy_premium,
#                         "payout_option_chosen":      normalized.get("payout_option_chosen", "LUMP_SUM"),
#                         "documents_verified":        docs_verified,
#                         "document_flags":            state.validation_flags,
#                     },
#                     "fraud_analysis":          package.fraud_analysis or {},
#                     "trust_analysis":          package.trust_analysis or {},
#                     "external_verification":   package.external_verification or {},
#                     "early_claim_analysis":    package.early_claim_analysis or {},
#                     "non_disclosure_analysis": package.non_disclosure_analysis or {},
#                     "conflict_resolution":     package.conflict_resolution or {},
#                     "graph_analysis":          package.graph_analysis or {},
#                     "final_recommendation":    package.final_recommendation,
#                     "escalation_required":     package.escalation_required,
#                 },
#             }
#             initial_state = get_initial_state(fraud_payload=fraud_payload)
#             final_state = policy_app.invoke(initial_state)
#             policy_result = {
#                 "policy_decision_report": final_state.get("claim_decision_report"),
#                 "policy_final_status":    final_state.get("final_status"),
#                 "detailed_summary":       final_state.get("detailed_summary"),
#                 "audit_log":              final_state.get("audit_log", []),
#             }
#             print(f"[AUTO-ANALYSIS POLICY DONE] batch={batch_id} status={policy_result.get('policy_final_status')}")
#         except Exception as policy_exc:
#             print(f"[AUTO-ANALYSIS POLICY ERROR] batch={batch_id} error={policy_exc}")
#             policy_result = {"error": str(policy_exc), "policy_final_status": "ERROR"}

#         # Store result back into the batch for GET polling
#         with PIPELINE_LOCK:
#             batch = PIPELINE_BATCHES.get(batch_id)
#             if batch:
#                 batch["analysis_result"] = {
#                     "status": "completed",
#                     "fraud_result": fraud_result,
#                     **policy_result,
#                 }
#         print(f"[AUTO-ANALYSIS COMPLETE] batch={batch_id}")

#     except Exception as exc:
#         print(f"[AUTO-ANALYSIS ERROR] batch={batch_id} error={exc}")
#         with PIPELINE_LOCK:
#             batch = PIPELINE_BATCHES.get(batch_id)
#             if batch:
#                 batch["analysis_result"] = {"status": "error", "error": str(exc)}

def _run_fraud_and_policy_background(batch_id: str, claim_payload: Dict[str, Any]) -> None:
    started_at = time.time()
    try:
        from api.routers.fraud_bridge import run_full_analysis_sync

        print(f"[AUTO-ANALYSIS START] batch={batch_id}")
        result = run_full_analysis_sync(claim_payload)

        with PIPELINE_LOCK:
            batch = PIPELINE_BATCHES.get(batch_id)
            if batch:
                batch["analysis_result"] = {
                    "status": "completed",
                    "started_at": started_at,
                    "completed_at": time.time(),
                    "processing_ms": int((time.time() - started_at) * 1000),
                    **result,
                }
                batch["updated_at"] = time.time()
        print(
            f"[AUTO-ANALYSIS COMPLETE] batch={batch_id} "
            f"policy_status={result.get('policy_final_status')}"
        )
    except Exception as exc:
        print(f"[AUTO-ANALYSIS ERROR] batch={batch_id} error={exc}")
        with PIPELINE_LOCK:
            batch = PIPELINE_BATCHES.get(batch_id)
            if batch:
                batch["analysis_result"] = {
                    "status": "error",
                    "started_at": started_at,
                    "completed_at": time.time(),
                    "processing_ms": int((time.time() - started_at) * 1000),
                    "error": str(exc),
                }
                batch["updated_at"] = time.time()


def get_rapid_ocr():
    global RAPID_OCR_INSTANCE
    if not HAS_RAPIDOCR:
        raise HTTPException(
            status_code=500,
            detail="RapidOCR is not installed. Please install 'rapidocr-onnxruntime'."
        )
    if RAPID_OCR_INSTANCE is None:
        with ocr_init_lock:
            if RAPID_OCR_INSTANCE is None:
                RAPID_OCR_INSTANCE = RapidOCR()
    return RAPID_OCR_INSTANCE

def is_numeric_word(word: str) -> bool:
    w = word.strip()
    if not w:
        return False
    has_digit = any(c.isdigit() for c in w)
    has_alpha = any(c.isalpha() for c in w)
    return has_digit and not has_alpha

def needs_ocr(page_text: str) -> bool:
    # If less than 20 chars or mostly garbage, use OCR
    return len(page_text.strip()) < 20

def enhance_image_cv2(pil_img: Image.Image) -> Image.Image:
    """Enhance image using sharpening and CLAHE."""
    try:
        # Resize to max 1500px to speed up OCR
        max_dim = 1500
        if pil_img.width > max_dim or pil_img.height > max_dim:
            scale = max_dim / max(pil_img.width, pil_img.height)
            new_width = int(pil_img.width * scale)
            new_height = int(pil_img.height * scale)
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Sharpen
        sharpening_kernel = np.array([
            [ 0, -0.25,  0],
            [-0.25,  2.0, -0.25],
            [ 0, -0.25,  0]
        ])
        sharpened = cv2.filter2D(gray, -1, sharpening_kernel)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(sharpened)
        
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)
    except Exception:
        return pil_img

def detect_qr_codes(pil_img: Image.Image) -> List[str]:
    """Detects and decodes QR codes using PyZbar for robust document scanning."""
    try:
        from pyzbar.pyzbar import decode
        # Convert PIL to CV2 grayscale for pyzbar
        img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # We can also do a basic thresholding to help pyzbar on bad scans
        # thresholding actually sometimes hurts pyzbar, so we try raw grayscale first
        decoded_objects = decode(gray)
        
        # If not found, try a bit of contrast enhancement
        if not decoded_objects:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            decoded_objects = decode(enhanced_gray)
            
        codes = []
        for obj in decoded_objects:
            text = obj.data.decode('utf-8')
            if text:
                codes.append(text)
                
        return codes
    except Exception as e:
        print(f"QR Detection Error (PyZbar): {e}")
    return []

def process_tesseract_ocr(image: Image.Image) -> Dict[str, Any]:
    if not HAS_TESSERACT:
        raise HTTPException(status_code=500, detail="Tesseract (pytesseract) is not installed on this system.")
    
    start_time = time.time()
    enhanced = enhance_image_cv2(image)
    
    try:
        data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tesseract OCR failed: {str(e)}")

    words = []
    total_conf = 0.0
    text_conf = 0.0
    num_conf = 0.0
    
    total_count = 0
    text_count = 0
    num_count = 0
    
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = float(data['conf'][i])
        
        if not text or conf == -1:
            continue
            
        is_num = is_numeric_word(text)
        
        word_info = {
            "text": text,
            "confidence": conf,
            "is_numeric": is_num
        }
        words.append(word_info)
        
        total_conf += conf
        total_count += 1
        
        if is_num:
            num_conf += conf
            num_count += 1
        else:
            text_conf += conf
            text_count += 1

    # Don't throw 422 if empty, just return empty words array to match ocr_test
    if total_count == 0:
        pass

    avg_overall = round(total_conf / total_count, 2) if total_count > 0 else 0.0
    avg_text = round(text_conf / text_count, 2) if text_count > 0 else 0.0
    avg_num = round(num_conf / num_count, 2) if num_count > 0 else 0.0
    raw_text = " ".join([w["text"] for w in words])
    
    return {
        "raw_text": raw_text,
        "words": words,
        "metrics": {
            "overall_confidence": avg_overall,
            "text_confidence": avg_text,
            "number_confidence": avg_num,
            "word_count": total_count,
            "text_count": text_count,
            "number_count": num_count,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
    }

def process_rapid_ocr(image: Image.Image) -> Dict[str, Any]:
    start_time = time.time()
    ocr_engine = get_rapid_ocr()
    enhanced = enhance_image_cv2(image)
    img_np = np.array(enhanced.convert("RGB"))
    
    try:
        with rapid_inference_lock:
            result, elapse = ocr_engine(img_np)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RapidOCR processing failed: {str(e)}")
        
    words = []
    total_conf = 0.0
    text_conf = 0.0
    num_conf = 0.0
    
    total_count = 0
    text_count = 0
    num_count = 0
    
    if result:
        for line in result:
            # RapidOCR line format: [dt_boxes, rec_text, score]
            if len(line) >= 3:
                text_str = str(line[1])
                conf = float(line[2])
                conf_percent = round(conf * 100 if conf <= 1.0 else conf, 2)
                
                words_in_block = text_str.split()
                for word in words_in_block:
                    cleaned = word.strip(".,;:?!()[]{}*\"'")
                    if not cleaned:
                        continue
                    is_num = is_numeric_word(cleaned)
                    
                    words.append({
                        "text": word,
                        "confidence": conf_percent,
                        "is_numeric": is_num
                    })
                    
                    total_conf += conf_percent
                    total_count += 1
                    if is_num:
                        num_conf += conf_percent
                        num_count += 1
                    else:
                        text_conf += conf_percent
                        text_count += 1

    avg_overall = round(total_conf / total_count, 2) if total_count > 0 else 0.0
    avg_text = round(text_conf / text_count, 2) if text_count > 0 else 0.0
    avg_num = round(num_conf / num_count, 2) if num_count > 0 else 0.0
    raw_text = " ".join([w["text"] for w in words])
    
    return {
        "raw_text": raw_text,
        "words": words,
        "metrics": {
            "overall_confidence": avg_overall,
            "text_confidence": avg_text,
            "number_confidence": avg_num,
            "word_count": total_count,
            "text_count": text_count,
            "number_count": num_count,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
    }

def run_ocr_with_fallback(image: Image.Image, primary_engine: str = "rapidocr") -> Tuple[Dict[str, Any], str]:
    if primary_engine == "tesseract":
        return process_tesseract_ocr(image), "tesseract"

    try:
        return process_rapid_ocr(image), "rapidocr"
    except Exception as rapid_exc:
        fallback_enabled = parse_bool_env("OCR_FALLBACK_ENABLED", True)
        if fallback_enabled and HAS_TESSERACT:
            print(f"[OCR FALLBACK] from=rapidocr to=tesseract reason={rapid_exc}")
            try:
                return process_tesseract_ocr(image), "tesseract"
            except Exception as tesseract_exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"RapidOCR failed and Tesseract fallback also failed: {tesseract_exc}",
                )
        raise

def process_document_ocr_only(
    file_bytes: bytes,
    filename: str,
    doc_type: str,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    request_id = request_id or uuid.uuid4().hex[:8]
    request_started_at = time.time()
    pages_result = []
    ocr_provider_metadata: Dict[str, Any] = {}
    auto_engine = "rapidocr"
    pdf_parallel_threshold_pages = max(1, parse_int_env("PDF_PARALLEL_THRESHOLD_PAGES", 1))
    default_pdf_workers = min(4, os.cpu_count() or 2)
    pdf_parallel_workers = min(max(1, parse_int_env("PDF_PARALLEL_PAGE_WORKERS", default_pdf_workers)), 8)

    try:
        if not pages_result and (filename.lower().endswith(".pdf") or file_bytes.startswith(b"%PDF")):
            if not HAS_PYMUPDF:
                raise HTTPException(
                    status_code=500,
                    detail="PyMuPDF (fitz) is not installed on the server. Cannot process PDF files."
                )

            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            num_pages = len(pdf_doc)
            if num_pages == 0:
                raise HTTPException(status_code=400, detail="The uploaded PDF file has no pages.")

            auto_engine = "rapidocr"

            def process_page(page_idx):
                page_started_at = time.time()
                print(
                    f"[PDF PAGE OCR START] req={request_id} doc={doc_type} "
                    f"file={filename} page={page_idx + 1}/{num_pages}"
                )
                page = pdf_doc.load_page(page_idx)
                text = page.get_text()

                if not needs_ocr(text):
                    words = []
                    for w in text.split():
                        cleaned = w.strip(".,;:?!()[]{}*\"'")
                        if cleaned:
                            words.append({
                                "text": cleaned,
                                "confidence": 100.0,
                                "is_numeric": is_numeric_word(cleaned)
                            })
                    total_count = len(words)
                    num_count = sum(1 for w in words if w["is_numeric"])
                    text_count = total_count - num_count

                    pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    detected_qrs = detect_qr_codes(img)

                    page_ocr = {
                        "raw_text": text.strip(),
                        "words": words,
                        "qr_codes": detected_qrs,
                        "metrics": {
                            "overall_confidence": 100.0 if total_count > 0 else 0.0,
                            "text_confidence": 100.0 if text_count > 0 else 0.0,
                            "number_confidence": 100.0 if num_count > 0 else 0.0,
                            "word_count": total_count,
                            "text_count": text_count,
                            "number_count": num_count,
                            "processing_time_ms": 1
                        }
                    }
                    page_ocr["ocr_engine_used"] = "native_pdf_text"
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    detected_qrs = detect_qr_codes(img)
                    page_ocr, page_engine = run_ocr_with_fallback(img, auto_engine)
                    page_ocr["ocr_engine_used"] = page_engine
                    page_ocr["qr_codes"] = detected_qrs

                page_ocr["page_number"] = page_idx + 1
                print(
                    f"[PDF PAGE OCR DONE] req={request_id} doc={doc_type} "
                    f"file={filename} page={page_idx + 1}/{num_pages} "
                    f"words={page_ocr.get('metrics', {}).get('word_count', 0)} "
                    f"ms={int((time.time() - page_started_at) * 1000)}"
                )
                return page_ocr

            if num_pages > pdf_parallel_threshold_pages:
                with ThreadPoolExecutor(max_workers=pdf_parallel_workers) as executor:
                    future_to_idx = {executor.submit(process_page, idx): idx for idx in range(num_pages)}
                    results_map = {}
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        results_map[idx] = future.result()
                for idx in range(num_pages):
                    if idx in results_map:
                        pages_result.append(results_map[idx])
            else:
                for idx in range(num_pages):
                    pages_result.append(process_page(idx))
        elif not pages_result:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            auto_engine = "rapidocr"
            detected_qrs = detect_qr_codes(img)
            page_ocr, page_engine = run_ocr_with_fallback(img, auto_engine)
            page_ocr["ocr_engine_used"] = page_engine
            page_ocr["qr_codes"] = detected_qrs
            page_ocr["page_number"] = 1
            pages_result.append(page_ocr)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

    total_words = 0
    total_texts = 0
    total_nums = 0
    weighted_overall_conf = 0.0
    weighted_text_conf = 0.0
    weighted_num_conf = 0.0
    total_processing_time = 0
    all_qr_codes = []

    for page in pages_result:
        metrics = page["metrics"]
        total_words += metrics["word_count"]
        total_texts += metrics["text_count"]
        total_nums += metrics["number_count"]
        weighted_overall_conf += metrics["overall_confidence"] * metrics["word_count"]
        weighted_text_conf += metrics["text_confidence"] * metrics["text_count"]
        weighted_num_conf += metrics["number_confidence"] * metrics["number_count"]
        total_processing_time += metrics["processing_time_ms"]
        if page.get("qr_codes"):
            all_qr_codes.extend(page["qr_codes"])

    all_qr_codes = list(set(all_qr_codes))
    global_overall = round(weighted_overall_conf / total_words, 2) if total_words > 0 else 0.0
    global_text = round(weighted_text_conf / total_texts, 2) if total_texts > 0 else 0.0
    global_num = round(weighted_num_conf / total_nums, 2) if total_nums > 0 else 0.0
    engines_used = sorted(set(page.get("ocr_engine_used", auto_engine) for page in pages_result))
    engine_used = "+".join(engines_used) if engines_used else auto_engine
    ocr_stage_ms = int((time.time() - request_started_at) * 1000)

    print(
        f"[OCR DONE] req={request_id} doc={doc_type} file={filename} "
        f"engine={engine_used} words={total_words} ocr_ms={ocr_stage_ms}"
    )

    return {
        "success": True,
        "filename": filename,
        "doc_type": doc_type,
        "engine_used": engine_used,
        "ocr_provider_metadata": ocr_provider_metadata,
        "llm_provider": None,
        "llm_model": None,
        "llm_fallback_used": False,
        "llm_fallback_reason": None,
        "llm_metrics": {},
        "pipeline_metrics": {
            "ocr_stage_ms": ocr_stage_ms,
            "llm_stage_ms": 0,
            "rapid_ocr_max_concurrency": RAPID_OCR_MAX_CONCURRENCY,
            "pdf_parallel_workers": pdf_parallel_workers,
            "pdf_parallel_threshold_pages": pdf_parallel_threshold_pages,
            **ocr_provider_metadata,
            "total_request_ms": int((time.time() - request_started_at) * 1000),
        },
        "qr_codes": all_qr_codes,
        "extracted_data": None,
        "global_metrics": {
            "overall_confidence": global_overall,
            "text_confidence": global_text,
            "number_confidence": global_num,
            "total_pages": len(pages_result),
            "total_words": total_words,
            "total_texts": total_texts,
            "total_numbers": total_nums,
            "total_processing_time_ms": total_processing_time
        },
        "pages": pages_result
    }

def run_json_extraction_from_pages(
    doc_type: str,
    pages_result: List[Dict[str, Any]],
    filename: str = "uploaded_file",
    request_id: Optional[str] = None,
    validate_output: bool = True,
) -> Dict[str, Any]:
    request_id = request_id or uuid.uuid4().hex[:8]
    extraction_started_at = time.time()
    extracted_data = None
    llm_provider = None
    llm_model = None
    classification_ms = None
    extraction_ms = None
    fallback_extraction_ms = None
    llm_fallback_used = False
    llm_fallback_reason = None
    llm_input_char_limit = max(2000, parse_int_env("LLM_INPUT_CHAR_LIMIT", 30000))
    llm_request_timeout_s = max(5.0, parse_float_env("LLM_REQUEST_TIMEOUT_S", 45.0))
    llm_retry_count = max(0, parse_int_env("LLM_RETRY_COUNT", 1))
    llm_retry_backoff_ms = max(0, parse_int_env("LLM_RETRY_BACKOFF_MS", 400))
    llm_input_truncated = False
    clean_text_chars = 0
    conf_text_chars = 0
    total_token_estimate = 0
    chunked_extraction_used = False
    chunk_count = 0
    chunk_token_estimates: List[int] = []
    chunk_extraction_metrics: List[Dict[str, Any]] = []
    chunk_merge_conflicts: List[Dict[str, Any]] = []
    grounding_issues: List[Dict[str, Any]] = []
    fallback_state_lock = threading.Lock()

    def call_llm_with_retry(
        client: OpenAI,
        kwargs: Dict[str, Any],
        stage: str,
        provider_override: Optional[str] = None,
    ):
        provider_label = provider_override or llm_provider or "unknown"
        total_attempts = llm_retry_count + 1
        for attempt in range(1, total_attempts + 1):
            try:
                queue_started_at = time.time()
                with llm_request_lock:
                    queue_ms = int((time.time() - queue_started_at) * 1000)
                    if queue_ms > 50:
                        print(
                            f"[LLM QUEUE] req={request_id} doc={doc_type} provider={provider_label} "
                            f"stage={stage} wait_ms={queue_ms} max_concurrency={LLM_MAX_CONCURRENCY}"
                        )
                    print(
                        f"[LLM CALL START] req={request_id} doc={doc_type} provider={provider_label} "
                        f"model={kwargs.get('model')} stage={stage} attempt={attempt}/{total_attempts}"
                    )
                    call_started_at = time.time()
                    response = client.chat.completions.create(**kwargs)
                    print(
                        f"[LLM CALL DONE] req={request_id} doc={doc_type} provider={provider_label} "
                        f"model={kwargs.get('model')} stage={stage} ms={int((time.time() - call_started_at) * 1000)}"
                    )
                    return response
            except Exception as call_exc:
                transient = is_transient_llm_error(call_exc)
                print(
                    f"[LLM ERROR] req={request_id} doc={doc_type} provider={provider_label} stage={stage} "
                    f"attempt={attempt}/{total_attempts} transient={transient} error={call_exc}"
                )
                if attempt >= total_attempts or not transient:
                    raise
                time.sleep((llm_retry_backoff_ms * attempt) / 1000.0)

    def build_page_text_inputs() -> List[Dict[str, Any]]:
        page_inputs = []
        for idx, page in enumerate(pages_result):
            clean_texts = []
            conf_texts = []
            for w in page.get("words", []):
                if "text" not in w:
                    continue
                t = str(w["text"]).strip()
                if not t:
                    continue
                try:
                    c = int(round(float(w.get("confidence", 0))))
                except Exception:
                    c = 0
                clean_texts.append(t)
                conf_texts.append(f"{t}|{c}")
            clean = " ".join(clean_texts)
            conf = " ".join(conf_texts)
            raw = str(page.get("raw_text") or clean)
            page_number = int(page.get("page_number") or idx + 1)
            page_inputs.append({
                "page_number": page_number,
                "clean_text": clean,
                "conf_text": conf,
                "raw_text": raw,
                "token_estimate": estimate_llm_tokens(clean) + estimate_llm_tokens(conf),
            })
        return page_inputs

    def build_text_inputs(page_inputs: List[Dict[str, Any]]) -> Tuple[str, str, bool]:
        clean_texts = []
        conf_texts = []
        clean_chars_used = 0
        conf_chars_used = 0
        input_limit_hit = False
        for page in page_inputs:
            page_clean_tokens = page.get("clean_text", "").split()
            page_conf_tokens = page.get("conf_text", "").split()
            for token_idx, t in enumerate(page_clean_tokens):
                conf_token = page_conf_tokens[token_idx] if token_idx < len(page_conf_tokens) else f"{t}|0"
                next_clean = clean_chars_used + (1 if clean_texts else 0) + len(t)
                next_conf = conf_chars_used + (1 if conf_texts else 0) + len(conf_token)
                if next_clean > llm_input_char_limit or next_conf > llm_input_char_limit:
                    input_limit_hit = True
                    break
                clean_texts.append(t)
                conf_texts.append(conf_token)
                clean_chars_used = next_clean
                conf_chars_used = next_conf
            if input_limit_hit:
                break
        return " ".join(clean_texts), " ".join(conf_texts), input_limit_hit

    def normalize_extracted_payload(payload: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        for key in schema.keys():
            if key not in payload:
                if key in SCHEMA_METADATA_FIELDS:
                    payload[key] = clone_json_value(schema.get(key, SCHEMA_METADATA_DEFAULTS[key]))
                else:
                    payload[key] = {
                        "value": None,
                        "ocr_confidence": None,
                        "extraction_confidence": 0.0,
                        "source_text": None,
                    }
            elif key in SCHEMA_METADATA_FIELDS and is_extraction_field_object(payload.get(key)):
                payload[key] = clone_json_value(schema.get(key, SCHEMA_METADATA_DEFAULTS[key]))

        if "DeathCategory" in payload:
            dc = payload["DeathCategory"]
            if isinstance(dc, dict):
                val = dc.get("value")
                if val not in ["NATURAL_OR_MEDICAL", "UNNATURAL", None]:
                    dc["value"] = None
                    dc["extraction_confidence"] = 0.0

        if "LowConfidenceFields" not in payload:
            payload["LowConfidenceFields"] = []
        if "UnmappedFields" not in payload:
            payload["UnmappedFields"] = []
        return apply_backend_schema_metadata(payload, schema_key, filename, request_id, pages_result)

    def clone_json(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value))
        except Exception:
            return value

    def build_page_chunks(
        page_inputs: List[Dict[str, Any]],
        schema_token_estimate: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        chunk_threshold = max(1000, parse_int_env("LLM_CHUNK_TOKEN_THRESHOLD", 3500))
        max_chunks = max(1, parse_int_env("LLM_MAX_CHUNKS_PER_DOC", 10))
        overlap_lines = max(0, min(parse_int_env("LLM_CHUNK_OVERLAP_LINES", 3), 8))
        prompt_overhead_tokens = 300
        page_budget = max(600, chunk_threshold - schema_token_estimate - prompt_overhead_tokens)
        chunks = []
        current_pages = []
        current_tokens = 0

        def emit_chunk(pages: List[Dict[str, Any]]) -> None:
            if not pages:
                return
            clean = "\n".join(page.get("clean_text", "") for page in pages if page.get("clean_text"))
            conf = "\n".join(page.get("conf_text", "") for page in pages if page.get("conf_text"))
            raw = "\n".join(page.get("raw_text", "") for page in pages if page.get("raw_text"))
            start_page = pages[0].get("page_number")
            end_page = pages[-1].get("page_number")
            chunks.append({
                "index": len(chunks) + 1,
                "page_start": start_page,
                "page_end": end_page,
                "page_range": f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}",
                "clean_text": clean,
                "conf_text": conf,
                "raw_text": raw,
                "token_estimate": (
                    estimate_llm_tokens(clean)
                    + estimate_llm_tokens(conf)
                    + schema_token_estimate
                    + prompt_overhead_tokens
                ),
                "previous_context": "",
            })

        for page in page_inputs:
            page_tokens = max(1, int(page.get("token_estimate") or 1))
            if current_pages and current_tokens + page_tokens > page_budget:
                emit_chunk(current_pages)
                current_pages = []
                current_tokens = 0
            current_pages.append(page)
            current_tokens += page_tokens
        emit_chunk(current_pages)

        for idx, chunk in enumerate(chunks):
            chunk["total"] = len(chunks)
            if idx > 0:
                chunk["previous_context"] = tail_lines(chunks[idx - 1].get("raw_text", ""), overlap_lines)

        if len(chunks) > max_chunks:
            return chunks, f"too_many_chunks:{len(chunks)}>{max_chunks}"
        return chunks, None

    def merge_chunk_payloads(
        payloads: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        schema: Dict[str, Any],
        full_raw_text: str,
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}

        def merge_node(path: str, current: Any, incoming: Any, chunk: Dict[str, Any]) -> Any:
            if is_extraction_field_object(incoming):
                if not field_has_value(incoming):
                    return current if current is not None else clone_json(incoming)
                incoming_value = incoming.get("value")
                if not field_has_value(current):
                    return clone_json(incoming)
                current_value = current.get("value") if is_extraction_field_object(current) else current
                if normalize_compare_value(current_value) == normalize_compare_value(incoming_value):
                    if extraction_confidence_of(incoming) > extraction_confidence_of(current):
                        return clone_json(incoming)
                    return current

                current_grounded = is_value_grounded(current_value, full_raw_text)
                incoming_grounded = is_value_grounded(incoming_value, chunk.get("raw_text", ""))
                choose_incoming = incoming_grounded and not current_grounded
                chunk_merge_conflicts.append({
                    "field": path,
                    "kept": incoming_value if choose_incoming else current_value,
                    "rejected": current_value if choose_incoming else incoming_value,
                    "incoming_page_range": chunk.get("page_range"),
                    "reason": "conflicting_values",
                })
                return clone_json(incoming) if choose_incoming else current

            if isinstance(incoming, dict):
                base = current if isinstance(current, dict) and not is_extraction_field_object(current) else {}
                result = clone_json(base) if isinstance(base, dict) else {}
                for key, value in incoming.items():
                    child_path = f"{path}.{key}" if path else str(key)
                    result[key] = merge_node(child_path, result.get(key), value, chunk)
                return result

            if isinstance(incoming, list):
                result = clone_json(current) if isinstance(current, list) else []
                seen = {stable_json_key(item) for item in result}
                for item in incoming:
                    if not field_has_value(item):
                        continue
                    key = stable_json_key(item)
                    if key not in seen:
                        result.append(clone_json(item))
                        seen.add(key)
                return result

            if not field_has_value(incoming):
                return current
            if not field_has_value(current):
                return clone_json(incoming)
            if isinstance(current, bool) and isinstance(incoming, bool):
                return current or incoming
            if normalize_compare_value(current) == normalize_compare_value(incoming):
                return current
            chunk_merge_conflicts.append({
                "field": path,
                "kept": current,
                "rejected": incoming,
                "incoming_page_range": chunk.get("page_range"),
                "reason": "conflicting_values",
            })
            return current

        for idx, payload in enumerate(payloads):
            chunk = chunks[idx] if idx < len(chunks) else {}
            merged = merge_node("", merged, payload, chunk)

        if chunk_merge_conflicts:
            merged["NeedsReview"] = True
            merged["Conflicts"] = chunk_merge_conflicts
        return normalize_extracted_payload(merged, schema)

    def apply_grounding_checks(payload: Dict[str, Any], full_raw_text: str) -> None:
        if not parse_bool_env("LLM_GROUNDING_CHECK_ENABLED", True):
            return

        def walk(node: Any, path: str) -> None:
            if is_extraction_field_object(node):
                value = node.get("value")
                if field_has_value(node) and should_ground_value(path, value) and not is_value_grounded(value, full_raw_text):
                    grounding_issues.append({
                        "field": path,
                        "value": value,
                        "reason": "value_not_found_in_ocr_text",
                    })
                return
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {"Conflicts", "GroundingIssues"}:
                        continue
                    child_path = f"{path}.{key}" if path else str(key)
                    walk(value, child_path)
            elif isinstance(node, list):
                for idx, item in enumerate(node):
                    walk(item, f"{path}[{idx}]")

        walk(payload, "")
        if grounding_issues:
            payload["NeedsReview"] = True
            payload["GroundingIssues"] = grounding_issues
            low_conf = payload.setdefault("LowConfidenceFields", [])
            for issue in grounding_issues:
                field = issue.get("field")
                if field and field not in low_conf:
                    low_conf.append(field)

    try:
        resolved_doc_type = doc_type
        schema_key = DOC_TYPE_MAPPING.get(doc_type)
        if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"]:
            schema_key = "GENERIC_IDENTITY"

        hf_enabled = parse_bool_env("ENABLE_HUGGINGFACE", False)
        hf_api_key = get_next_hf_key() if hf_enabled else None
        is_huggingface = bool(hf_api_key)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        api_key = hf_api_key or openai_api_key
        supports_response_format = False

        if not api_key:
            print(f"[LLM ROUTER] req={request_id} provider=none (no API key configured)")
            return {
                "success": True,
                "doc_type": resolved_doc_type,
                "extracted_data": None,
                "llm_provider": None,
                "llm_model": None,
                "llm_fallback_used": False,
                "llm_fallback_reason": None,
                "llm_metrics": {},
            }
        if is_huggingface:
            llm_provider = "huggingface"
            llm_model = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=api_key,
                timeout=llm_request_timeout_s,
                max_retries=0,
            )
        else:
            llm_provider = "openai"
            llm_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            client = OpenAI(api_key=api_key, timeout=llm_request_timeout_s, max_retries=0)
            supports_response_format = True

        print(
            f"[LLM DECISION] req={request_id} provider={llm_provider} "
            f"model={llm_model} schema_key={schema_key or 'none'}"
        )

        page_inputs = build_page_text_inputs()
        clean_text, conf_text, llm_input_truncated = build_text_inputs(page_inputs)
        clean_text_chars = len(clean_text)
        conf_text_chars = len(conf_text)
        if llm_input_truncated:
            print(
                f"[LLM INPUT] req={request_id} truncated=true limit={llm_input_char_limit} "
                f"clean_chars={clean_text_chars} conf_chars={conf_text_chars}"
            )

        strict_classifier_for_fixed = parse_bool_env("STEP1_CLASSIFIER_FOR_FIXED_DOCS", False)
        should_run_classifier = doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"] or strict_classifier_for_fixed
        if clean_text.strip() and should_run_classifier:
            classifier_data = fast_classify_identity_document(clean_text) if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"] else None
            if classifier_data:
                detected_type = classifier_data.get("detected_type", "Unknown")
                conf = float(classifier_data.get("confidence", 0.0))
                classification_ms = 0
                print(
                    f"[CLASSIFIER LOCAL] req={request_id} detected_type={detected_type} "
                    f"conf={conf} reason={classifier_data.get('reason')}"
                )
                if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"]:
                    schema_key = detected_type
                    resolved_doc_type = {v: k for k, v in DOC_TYPE_MAPPING.items()}.get(schema_key, doc_type)
                if conf < 0.5:
                    return {
                        "success": False,
                        "status": "REUPLOAD_REQUIRED",
                        "error_type": "UNRECOGNIZED_DOCUMENT",
                        "title": "Document Not Recognized",
                        "message": "We could not identify this document. It may be too blurry.",
                        "action": "Please retake the photo ensuring it is clear and well-lit.",
                        "missing_fields": [],
                    }
            else:
                try:
                    class_prompt = CLASSIFIER_PROMPT_TEMPLATE.format(OCR_TEXT=clean_text[:4000])
                    class_model = llm_model if is_huggingface else os.getenv("OPENAI_CLASSIFIER_MODEL", "gpt-4o-mini")
                    class_kwargs = {
                        "model": class_model,
                        "messages": [{"role": "user", "content": class_prompt}],
                        "temperature": 0.0,
                    }
                    if supports_response_format:
                        class_kwargs["response_format"] = {"type": "json_object"}
                    classify_start = time.time()
                    class_resp = call_llm_with_retry(client, class_kwargs, "classification")
                    classification_ms = int((time.time() - classify_start) * 1000)
                    raw_content = class_resp.choices[0].message.content
                    if is_huggingface:
                        raw_content = clean_json_response(raw_content)
                    classifier_data = json.loads(raw_content)
                    detected_type = classifier_data.get("detected_type", "Unknown")
                    conf = float(classifier_data.get("confidence", 0.0))
                    print(f"[CLASSIFIER API] req={request_id} detected_type={detected_type} conf={conf}")

                    if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"]:
                        schema_key = detected_type
                        resolved_doc_type = {v: k for k, v in DOC_TYPE_MAPPING.items()}.get(schema_key, doc_type)
                    if conf < 0.5:
                        return {
                            "success": False,
                            "status": "REUPLOAD_REQUIRED",
                            "error_type": "UNRECOGNIZED_DOCUMENT",
                            "title": "Document Not Recognized",
                            "message": "We could not identify this document. It may be too blurry.",
                            "action": "Please retake the photo ensuring it is clear and well-lit.",
                            "missing_fields": [],
                        }
                except Exception as ce:
                    print(f"[CLASSIFIER ERROR] req={request_id} error={ce}")

        raw_schema = get_combined_schemas().get(schema_key)
        if not raw_schema or not clean_text:
            return {
                "success": True,
                "extracted_data": None,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "llm_fallback_used": False,
                "llm_fallback_reason": "schema_or_text_missing",
                "llm_metrics": {
                    "classification_ms": classification_ms,
                    "extraction_ms": extraction_ms,
                    "fallback_extraction_ms": fallback_extraction_ms,
                    "input_clean_chars": clean_text_chars,
                    "input_conf_chars": conf_text_chars,
                    "input_truncated": llm_input_truncated,
                    "request_timeout_s": llm_request_timeout_s,
                    "retry_count": llm_retry_count,
                },
            }

        schema = prepare_schema_for_extraction(schema_key, raw_schema)
        schema_compact = json.dumps(schema, separators=(",", ":"))
        schema_token_estimate = estimate_llm_tokens(schema_compact)
        full_clean_text = "\n".join(page.get("clean_text", "") for page in page_inputs if page.get("clean_text"))
        full_conf_text = "\n".join(page.get("conf_text", "") for page in page_inputs if page.get("conf_text"))
        full_raw_text = "\n".join(page.get("raw_text", "") for page in page_inputs if page.get("raw_text"))
        total_token_estimate = (
            estimate_llm_tokens(full_clean_text)
            + estimate_llm_tokens(full_conf_text)
            + schema_token_estimate
        )
        chunk_threshold = max(1000, parse_int_env("LLM_CHUNK_TOKEN_THRESHOLD", 3500))
        timeout_base_s = max(5.0, parse_float_env("LLM_TIMEOUT_BASE_SECONDS", llm_request_timeout_s))
        timeout_per_1k_tokens = max(0.0, parse_float_env("LLM_TIMEOUT_PER_1K_TOKENS", 4.0))
        adaptive_timeout_s = min(
            120.0,
            max(llm_request_timeout_s, timeout_base_s + (min(total_token_estimate, chunk_threshold) / 1000.0) * timeout_per_1k_tokens),
        )
        if adaptive_timeout_s > llm_request_timeout_s + 0.1:
            llm_request_timeout_s = adaptive_timeout_s
            if is_huggingface:
                client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=api_key,
                    timeout=llm_request_timeout_s,
                    max_retries=0,
                )
            else:
                client = OpenAI(api_key=api_key, timeout=llm_request_timeout_s, max_retries=0)
            print(
                f"[LLM TIMEOUT] req={request_id} adaptive_timeout_s={llm_request_timeout_s:.1f} "
                f"estimated_tokens={total_token_estimate}"
            )

        prompt = PROMPT_TEMPLATE.format(
            DOCUMENT_TYPE=schema_key,
            SCHEMA=schema_compact,
            CLEAN_TEXT=clean_text,
            CONF_TEXT=conf_text,
        )

        fallback_model = os.getenv("FALLBACK_OPENAI_MODEL", "gpt-4o")
        fallback_enabled = parse_bool_env("CLOUD_FALLBACK_TO_OPENAI", True) and (
            llm_provider == "huggingface" or (llm_provider == "openai" and fallback_model != llm_model)
        )

        def build_completion_kwargs(prompt_text: str, model_name: str, response_format_enabled: bool) -> Dict[str, Any]:
            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a precise JSON data extraction engine."},
                    {"role": "user", "content": prompt_text},
                ],
                "temperature": 0.0,
            }
            if response_format_enabled:
                kwargs["response_format"] = {"type": "json_object"}
            return kwargs

        def parse_payload(raw_text: Optional[str], provider_name: str, run_validation: bool = True):
            if raw_text is None:
                return None, None, "empty_response"
            try:
                payload_text = clean_json_response(raw_text) if provider_name == "huggingface" else raw_text
                payload = json.loads(payload_text)
                if not isinstance(payload, dict):
                    return None, None, "invalid_json"
                payload = normalize_extracted_payload(payload, schema)
                if validate_output and run_validation:
                    validation_result = validate_document(doc_type, payload)
                    if validation_result.get("status") != "OK":
                        return None, validation_result, "validation_failed"
                return payload, None, None
            except json.JSONDecodeError:
                return None, None, "invalid_json"
            except Exception:
                return None, None, "parse_error"

        def try_openai_fallback(
            reason: str,
            fallback_prompt: Optional[str] = None,
            run_validation: bool = True,
            stage: str = "fallback_extraction",
            mutate_state: bool = True,
        ):
            nonlocal fallback_extraction_ms, llm_provider, llm_model, llm_fallback_used, llm_fallback_reason
            fallback_key = os.getenv("OPENAI_API_KEY")
            if not fallback_key:
                if mutate_state:
                    llm_fallback_reason = "cloud_fallback_no_api_key"
                return None, None
            fallback_client = OpenAI(api_key=fallback_key, timeout=llm_request_timeout_s, max_retries=0)
            fallback_kwargs = build_completion_kwargs(fallback_prompt or prompt, fallback_model, True)
            try:
                fallback_start = time.time()
                fallback_resp = call_llm_with_retry(
                    fallback_client,
                    fallback_kwargs,
                    stage,
                    provider_override="openai",
                )
                fallback_duration_ms = int((time.time() - fallback_start) * 1000)
                with fallback_state_lock:
                    fallback_extraction_ms = (fallback_extraction_ms or 0) + fallback_duration_ms
                fallback_data, fallback_validation, parse_error = parse_payload(
                    fallback_resp.choices[0].message.content,
                    "openai",
                    run_validation=run_validation,
                )
                if fallback_data is not None:
                    if mutate_state:
                        with fallback_state_lock:
                            llm_provider = "openai"
                            llm_model = fallback_model
                            llm_fallback_used = True
                            llm_fallback_reason = reason
                    print(f"[LLM FALLBACK] req={request_id} provider=openai model={fallback_model} reason={reason}")
                    return fallback_data, None
                if fallback_validation is not None:
                    if mutate_state:
                        llm_fallback_reason = "cloud_fallback_validation_failed"
                    return None, fallback_validation
                if mutate_state:
                    llm_fallback_reason = f"cloud_fallback_{parse_error or 'failed'}"
                return None, None
            except Exception as fallback_exc:
                print(f"[LLM FALLBACK ERROR] req={request_id} error={fallback_exc}")
                if mutate_state:
                    llm_fallback_reason = "cloud_fallback_error"
                return None, None

        def run_primary_prompt(prompt_text: str, stage: str) -> Tuple[str, int]:
            kwargs = build_completion_kwargs(prompt_text, llm_model, supports_response_format)
            started_at = time.time()
            response = call_llm_with_retry(client, kwargs, stage)
            duration_ms = int((time.time() - started_at) * 1000)
            return response.choices[0].message.content, duration_ms

        def build_chunk_prompt(chunk: Dict[str, Any]) -> str:
            return CHUNK_PROMPT_TEMPLATE.format(
                DOCUMENT_TYPE=schema_key,
                SCHEMA=schema_compact,
                CHUNK_INDEX=chunk.get("index"),
                CHUNK_TOTAL=chunk.get("total"),
                PAGE_RANGE=chunk.get("page_range"),
                PREVIOUS_CONTEXT=chunk.get("previous_context") or "",
                CLEAN_TEXT=chunk.get("clean_text") or "",
                CONF_TEXT=chunk.get("conf_text") or "",
            )

        def extract_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
            chunk_prompt = build_chunk_prompt(chunk)
            provider_name = "huggingface" if is_huggingface else "openai"
            stage = f"chunk_{chunk.get('index')}_extraction"
            fallback_attempted = False
            try:
                response_text, duration_ms = run_primary_prompt(chunk_prompt, stage)
                payload, validation_result, parse_error = parse_payload(
                    response_text,
                    provider_name,
                    run_validation=False,
                )
                if payload is not None:
                    return {
                        "index": chunk.get("index"),
                        "payload": payload,
                        "provider": provider_name,
                        "model": llm_model,
                        "duration_ms": duration_ms,
                        "fallback": False,
                    }
                if fallback_enabled:
                    fallback_data, fallback_validation = try_openai_fallback(
                        f"chunk_{chunk.get('index')}_{parse_error or 'invalid'}",
                        fallback_prompt=chunk_prompt,
                        run_validation=False,
                        stage=f"fallback_chunk_{chunk.get('index')}",
                        mutate_state=False,
                    )
                    fallback_attempted = True
                    if fallback_data is not None:
                        return {
                            "index": chunk.get("index"),
                            "payload": fallback_data,
                            "provider": "openai",
                            "model": fallback_model,
                            "duration_ms": fallback_extraction_ms,
                            "fallback": True,
                        }
                    if fallback_validation is not None:
                        raise ValueError(f"chunk validation failed: {fallback_validation}")
                raise ValueError(f"chunk parse failed: {parse_error or 'unknown'}")
            except Exception as primary_exc:
                if fallback_enabled and not fallback_attempted:
                    fallback_data, fallback_validation = try_openai_fallback(
                        f"chunk_{chunk.get('index')}_primary_error",
                        fallback_prompt=chunk_prompt,
                        run_validation=False,
                        stage=f"fallback_chunk_{chunk.get('index')}",
                        mutate_state=False,
                    )
                    if fallback_data is not None:
                        return {
                            "index": chunk.get("index"),
                            "payload": fallback_data,
                            "provider": "openai",
                            "model": fallback_model,
                            "duration_ms": fallback_extraction_ms,
                            "fallback": True,
                        }
                    if fallback_validation is not None:
                        raise ValueError(f"chunk validation failed: {fallback_validation}") from primary_exc
                raise

        if total_token_estimate > chunk_threshold:
            chunks, chunk_error = build_page_chunks(page_inputs, schema_token_estimate)
            chunk_count = len(chunks)
            chunk_token_estimates = [int(chunk.get("token_estimate") or 0) for chunk in chunks]
            if chunk_error:
                return {
                    "success": False,
                    "status": "PROCESSING_FAILED",
                    "error_type": "LLM_INPUT_TOO_LARGE",
                    "title": "Document Too Large",
                    "message": f"The document needs {chunk_count} extraction chunks, above the configured limit.",
                    "action": "Increase LLM_MAX_CHUNKS_PER_DOC or upload a shorter document.",
                    "missing_fields": [],
                }

            chunked_extraction_used = True
            max_doc_chunk_workers = max(1, min(parse_int_env("LLM_MAX_CONCURRENT_CHUNKS_PER_DOC", 2), 4, chunk_count))
            print(
                f"[LLM CHUNK ROUTE] req={request_id} doc={doc_type} chunks={chunk_count} "
                f"estimated_tokens={total_token_estimate} threshold={chunk_threshold} "
                f"workers={max_doc_chunk_workers}"
            )
            chunk_start = time.time()
            chunk_results: Dict[int, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=max_doc_chunk_workers) as executor:
                future_to_chunk = {executor.submit(extract_chunk, chunk): chunk for chunk in chunks}
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    result = future.result()
                    result_idx = int(result.get("index") or chunk.get("index"))
                    chunk_results[result_idx] = result
                    chunk_extraction_metrics.append({
                        "chunk_index": result_idx,
                        "page_range": chunk.get("page_range"),
                        "provider": result.get("provider"),
                        "model": result.get("model"),
                        "duration_ms": result.get("duration_ms"),
                        "fallback": result.get("fallback"),
                        "token_estimate": chunk.get("token_estimate"),
                    })
                    print(
                        f"[LLM CHUNK DONE] req={request_id} doc={doc_type} chunk={result_idx}/{chunk_count} "
                        f"provider={result.get('provider')} page_range={chunk.get('page_range')}"
                    )

            ordered_payloads = [chunk_results[idx]["payload"] for idx in sorted(chunk_results.keys())]
            ordered_chunks = [chunks[idx - 1] for idx in sorted(chunk_results.keys())]
            extracted_data = merge_chunk_payloads(ordered_payloads, ordered_chunks, schema, full_raw_text or full_clean_text)
            apply_grounding_checks(extracted_data, full_raw_text or full_clean_text)
            extraction_ms = int((time.time() - chunk_start) * 1000)
            providers_used = sorted({metric.get("provider") for metric in chunk_extraction_metrics if metric.get("provider")})
            models_used = sorted({metric.get("model") for metric in chunk_extraction_metrics if metric.get("model")})
            if any(metric.get("fallback") for metric in chunk_extraction_metrics):
                llm_fallback_used = True
                llm_fallback_reason = llm_fallback_reason or "chunk_fallback"
            if len(providers_used) > 1:
                llm_provider = "mixed"
                llm_model = "+".join(models_used)
            elif providers_used:
                llm_provider = providers_used[0]
                llm_model = models_used[0] if models_used else llm_model
            if validate_output:
                validation_result = validate_document(doc_type, extracted_data)
                if validation_result.get("status") != "OK":
                    return {"success": False, **validation_result}
        else:
            response_text = None
            try:
                response_text, extraction_ms = run_primary_prompt(prompt, "extraction")
            except Exception:
                extraction_ms = int((time.time() - extraction_started_at) * 1000)
                if fallback_enabled:
                    extracted_data, fallback_validation_result = try_openai_fallback("primary_extraction_error")
                    if extracted_data is None and fallback_validation_result:
                        return {"success": False, **fallback_validation_result}
                if extracted_data is None:
                    return {
                        "success": False,
                        "status": "PROCESSING_FAILED",
                        "error_type": "LLM_TIMEOUT",
                        "title": "Extraction Timed Out",
                        "message": f"The {(llm_provider or 'primary').title()} model did not finish extraction in time.",
                        "action": "Please retry. If this persists, use OpenAI fallback or reduce input size.",
                        "missing_fields": [],
                    }

            if extracted_data is None:
                provider_name = "huggingface" if is_huggingface else "openai"
                extracted_data, validation_result, parse_error = parse_payload(response_text, provider_name)
                if validation_result is not None and fallback_enabled:
                    fallback_data, fallback_validation_result = try_openai_fallback("primary_validation_failed")
                    if fallback_data is not None:
                        extracted_data = fallback_data
                    elif fallback_validation_result:
                        return {"success": False, **fallback_validation_result}
                elif parse_error and fallback_enabled:
                    fallback_data, fallback_validation_result = try_openai_fallback(f"primary_{parse_error}")
                    if fallback_data is not None:
                        extracted_data = fallback_data
                    elif fallback_validation_result:
                        return {"success": False, **fallback_validation_result}

                if extracted_data is None:
                    if validation_result is not None:
                        return {"success": False, **validation_result}
                    return {
                        "success": False,
                        "status": "PROCESSING_FAILED",
                        "error_type": "INVALID_LLM_JSON",
                        "title": "Extraction Format Error",
                        "message": "The model returned invalid JSON for this document.",
                        "action": "Please retry the upload. If this persists, switch to cloud extraction.",
                        "missing_fields": [],
                    }
            if extracted_data is not None and not chunked_extraction_used:
                apply_grounding_checks(extracted_data, full_raw_text or full_clean_text)

        return {
            "success": True,
            "doc_type": resolved_doc_type,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_fallback_used": llm_fallback_used,
            "llm_fallback_reason": llm_fallback_reason,
            "extracted_data": extracted_data,
            "llm_metrics": {
                "classification_ms": classification_ms,
                "extraction_ms": extraction_ms,
                "fallback_extraction_ms": fallback_extraction_ms,
                "input_clean_chars": clean_text_chars,
                "input_conf_chars": conf_text_chars,
                "input_truncated": llm_input_truncated,
                "estimated_input_tokens": total_token_estimate,
                "chunked_extraction_used": chunked_extraction_used,
                "chunk_count": chunk_count,
                "chunk_token_estimates": chunk_token_estimates,
                "chunk_extraction_metrics": chunk_extraction_metrics,
                "chunk_merge_conflicts": chunk_merge_conflicts,
                "grounding_issues": grounding_issues,
                "request_timeout_s": llm_request_timeout_s,
                "retry_count": llm_retry_count,
            },
            "pipeline_metrics": {
                "llm_stage_ms": int((time.time() - extraction_started_at) * 1000),
            },
        }
    finally:
        print(
            f"[LLM STAGE DONE] req={request_id} doc={doc_type} provider={llm_provider or 'none'} "
            f"model={llm_model or 'none'} llm_ms={int((time.time() - extraction_started_at) * 1000)}"
        )

INTEGRATION_EXCLUDED_KEYS = set(SCHEMA_METADATA_FIELDS) | {"LowConfidenceFields", "UnmappedFields"}

NAME_HONORIFICS = {
    "mr", "mrs", "ms", "miss", "mx", "mister", "shri", "sri", "shree",
    "kumar", "kumari", "dr", "doctor", "prof", "professor",
}

KEY_ALIAS_MAP = {
    "fullname": "name",
    "full_name": "name",
    "claimantname": "name",
    "claimant_name": "name",
    "lifeassuredname": "name",
    "life_assured_name": "name",
    "fatherorhusbandname": "father_or_husband_name",
    "father_or_husband_name": "father_or_husband_name",
    "fathername": "father_name",
    "mothername": "mother_name",
    "spousename": "spouse_name",
    "guardianname": "guardian_name",
    "aadhaarnumber": "aadhaar_number",
    "maskedaadhaarnumber": "aadhaar_number",
    "deceasedaadhaarnumber": "aadhaar_number",
    "fatheraadhaarnumber": "aadhaar_number",
    "uid": "aadhaar_number",
    "uidofdeceased": "aadhaar_number",
    "pan": "pan_number",
    "pannumber": "pan_number",
    "voteroidnumber": "voter_id_number",
    "epicnumber": "voter_id_number",
    "passportnumber": "passport_number",
    "drivinglicencenumber": "driving_licence_number",
    "driverlicencenumber": "driving_licence_number",
    "registrationnumber": "registration_number",
    "doctorregistrationnumber": "doctor_registration_number",
    "nmrnumber": "doctor_registration_number",
    "doctorregno": "doctor_registration_number",
    "hospitalid": "hospital_id",
    "hospitalrohiniid": "hospital_rohini_id",
    "rohiniid": "hospital_rohini_id",
    "firnumber": "fir_number",
    "dateofdeath": "date_of_death",
    "dateofdischargeordeath": "date_of_death",
    "causeofdeath": "cause_of_death",
    "primarycauseofdeath": "cause_of_death",
    "immediatecause": "cause_of_death",
    "finalopinion": "cause_of_death",
    "placeofdeath": "place_of_death",
    "bankname": "bank_name",
    "accountholdername": "account_holder_name",
    "accountnumber": "account_number",
    "ifsccode": "ifsc_code",
    "micrcode": "micr_code",
    "branchname": "branch_name",
}

NAME_FIELD_KEYS = {
    "name",
    "father_name",
    "mother_name",
    "spouse_name",
    "guardian_name",
    "father_or_husband_name",
    "account_holder_name",
}

ID_FIELD_KEYS = {
    "aadhaar_number",
    "pan_number",
    "voter_id_number",
    "passport_number",
    "driving_licence_number",
    "registration_number",
    "doctor_registration_number",
    "hospital_rohini_id",
    "hospital_id",
    "fir_number",
    "account_number",
    "ifsc_code",
    "micr_code",
}

BANK_FIELD_KEYS = {
    "bank_name",
    "account_holder_name",
    "account_number",
    "ifsc_code",
    "micr_code",
    "branch_name",
    "account_type",
}

DATE_FIELD_KEYS = {
    "date_of_death",
    "date_of_birth",
    "date_of_issue",
    "date_of_registration",
    "date_of_fir",
    "date_of_admission",
    "date_of_discharge",
    "date_of_discharge_or_death",
    "date",
}

CLAIMANT_DOC_TYPES_FOR_COMPARE = {
    "claimant_statement_form",
    "aadhaar_card",
    "passport",
    "driving_licence",
    "voter_id",
    "pan_card",
    "bank_proof",
    "identity_proof",
    "address_proof",
}

LIFE_ASSURED_DOC_TYPES_FOR_COMPARE = {
    "death_certificate",
    "medico_legal_cause_of_death_certificate",
    "discharge_summary",
    "past_medical_records_and_treatment_papers",
    "medical_attendant_hospital_certificate",
    "employer_certificate",
    "first_information_report",
    "inquest_panchanana_report",
    "final_police_investigation_report",
    "postmortem_report",
    "viscera_chemical_examination_report",
    "newspaper_cutting",
    "admission_form",
    "diagnostic_test_report",
    "indoor_case_papers",
}


def _snake_doc_type(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return "unknown_document"

    # Robust CamelCase/PascalCase -> snake_case conversion that preserves acronyms:
    # PANCard -> pan_card, VoterID -> voter_id, FIR -> fir.
    snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake)
    snake = re.sub(r"[^a-zA-Z0-9]+", "_", snake).strip("_").lower()
    return snake or "unknown_document"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _normalize_ocr_confidence_ratio(value: Any) -> float:
    numeric = _safe_float(value, 0.0)
    if numeric > 1.0:
        numeric = numeric / 100.0
    numeric = max(0.0, min(1.0, numeric))
    return round(numeric, 4)


def _extract_scalar_from_node(value: Any) -> Optional[str]:
    if value is None:
        return None
    if is_extraction_field_object(value):
        return _extract_scalar_from_node(value.get("value"))
    if isinstance(value, dict):
        if "value" in value:
            return _extract_scalar_from_node(value.get("value"))
        if "source_text" in value:
            return _extract_scalar_from_node(value.get("source_text"))
        return None
    if isinstance(value, list):
        items = [_extract_scalar_from_node(item) for item in value]
        items = [item for item in items if item]
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        return " | ".join(items)
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    return text or None


def _walk_key_match_integration(obj: Any, key_candidates: List[str]) -> Optional[str]:
    candidates = {key.lower() for key in key_candidates}

    def _walk(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in candidates:
                    scalar = _extract_scalar_from_node(value)
                    if scalar:
                        return scalar
                hit = _walk(value)
                if hit:
                    return hit
        elif isinstance(node, list):
            for item in node:
                hit = _walk(item)
                if hit:
                    return hit
        return None

    return _walk(obj)


def _walk_section_value(
    obj: Any,
    section_candidates: List[str],
    field_candidates: List[str],
) -> Optional[str]:
    section_set = {key.lower() for key in section_candidates}

    def _walk(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in section_set:
                    hit = _walk_key_match_integration(value, field_candidates)
                    if hit:
                        return hit
                nested_hit = _walk(value)
                if nested_hit:
                    return nested_hit
        elif isinstance(node, list):
            for item in node:
                nested_hit = _walk(item)
                if nested_hit:
                    return nested_hit
        return None

    return _walk(obj)


def _extract_party_name(entities: Any, party: str) -> Optional[str]:
    if party == "claimant":
        direct = _walk_key_match_integration(entities, ["ClaimantName"])
        if direct:
            return direct
        return _walk_section_value(entities, ["Claimant"], ["Name", "FullName"])
    direct = _walk_key_match_integration(entities, ["LifeAssuredName"])
    if direct:
        return direct
    return _walk_section_value(
        entities,
        ["LifeAssured", "Deceased", "Person", "LifeAssuredDetails"],
        ["Name", "FullName"],
    )


def _extract_entities_from_payload(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    entities: Dict[str, Any] = {}
    for key, value in extracted_data.items():
        if key in INTEGRATION_EXCLUDED_KEYS:
            continue
        entities[key] = value
    return entities


def _collect_ocr_text_from_entities(entities: Dict[str, Any], max_chars: int = 12000) -> str:
    chunks: List[str] = []

    def _walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            if is_extraction_field_object(node):
                scalar = _extract_scalar_from_node(node)
                if scalar:
                    chunks.append(f"{prefix}: {scalar}" if prefix else scalar)
                return
            for key, value in node.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(next_prefix, value)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                _walk(f"{prefix}[{index}]", item)

    _walk("", entities)
    return " | ".join(chunks)[:max_chars]


def _collect_ocr_text_from_pages(pages: Any, max_chars: int = 12000) -> str:
    if not isinstance(pages, list):
        return ""
    lines: List[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        raw_text = str(page.get("raw_text") or "").strip()
        if raw_text:
            lines.append(raw_text)
    return "\n".join(lines)[:max_chars]


def _normalize_for_cross_compare(value: str, kind: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if kind == "date":
        return re.sub(r"[^0-9]+", "", text)
    if kind == "id":
        return re.sub(r"[^a-z0-9]+", "", text)
    if kind == "name":
        tokens = re.findall(r"[a-z0-9]+", text)
        filtered = [token for token in tokens if token not in NAME_HONORIFICS]
        return " ".join(filtered).strip()
    return re.sub(r"\s+", " ", text).strip()


def _normalize_key_token(token: Any) -> str:
    raw = str(token or "").strip()
    if not raw:
        return ""
    camel_split = re.sub(r"(?<!^)(?=[A-Z])", "_", raw)
    normalized = re.sub(r"[^a-z0-9]+", "_", camel_split.lower()).strip("_")
    return normalized


def _canonical_leaf_key(raw_leaf: str) -> str:
    token = _normalize_key_token(raw_leaf)
    token_flat = token.replace("_", "")
    return KEY_ALIAS_MAP.get(token_flat, KEY_ALIAS_MAP.get(token, token))


def _infer_doc_subject(doc_type: str) -> str:
    if doc_type in CLAIMANT_DOC_TYPES_FOR_COMPARE:
        return "claimant"
    if doc_type in LIFE_ASSURED_DOC_TYPES_FOR_COMPARE:
        return "life_assured"
    return "unknown"


def _infer_subject_from_path(doc_type: str, path_tokens: List[str], canonical_leaf: str) -> str:
    section = path_tokens[0] if path_tokens else ""
    section_flat = section.replace("_", "")

    if section_flat in {"claimant", "witness", "declarant"}:
        return "claimant"
    if section_flat in {"lifeassured", "deceased"}:
        return "life_assured"
    if section_flat == "bank" or canonical_leaf in BANK_FIELD_KEYS:
        return "bank"
    if section_flat in {"hospital"}:
        return "hospital"
    if section_flat in {"police", "fir"}:
        return "police"
    if section_flat in {"treatingdoctor", "familydoctor", "lastattendingdoctor", "doctor"}:
        return "doctor"
    if canonical_leaf in {"date_of_death", "cause_of_death", "place_of_death"}:
        return "life_assured"
    if section_flat in {"person", "identifiers", "address", "family", "passport", "licence", "license"}:
        return _infer_doc_subject(doc_type)
    return _infer_doc_subject(doc_type)


def _infer_field_kind(canonical_leaf: str, path_tokens: List[str]) -> str:
    if canonical_leaf in NAME_FIELD_KEYS:
        return "name"
    if canonical_leaf in ID_FIELD_KEYS:
        return "id"
    if canonical_leaf in DATE_FIELD_KEYS or "date" in canonical_leaf:
        return "date"
    token_hint = ".".join(path_tokens)
    if "date" in token_hint:
        return "date"
    if any(part in canonical_leaf for part in ["number", "id", "code"]):
        return "id"
    return "text"


def _flatten_extracted_entities_fields(entities: Dict[str, Any]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []

    def _walk(node: Any, path_tokens: List[str]) -> None:
        if is_extraction_field_object(node):
            value = _extract_scalar_from_node(node.get("value"))
            if not value:
                return
            extraction_conf = _safe_float(node.get("extraction_confidence"), 0.0)
            ocr_conf = _safe_float(node.get("ocr_confidence"), 0.0)
            ocr_ratio = ocr_conf / 100.0 if ocr_conf > 1.0 else ocr_conf
            ocr_ratio = max(0.0, min(1.0, ocr_ratio))
            confidence_score = round((0.7 * extraction_conf) + (0.3 * ocr_ratio), 4)
            flattened.append({
                "path_tokens": list(path_tokens),
                "path": ".".join(path_tokens),
                "leaf": path_tokens[-1] if path_tokens else "",
                "value": value,
                "source_text": _extract_scalar_from_node(node.get("source_text")),
                "ocr_confidence": node.get("ocr_confidence"),
                "extraction_confidence": node.get("extraction_confidence"),
                "confidence_score": confidence_score,
            })
            return

        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"LowConfidenceFields", "UnmappedFields"}:
                    continue
                _walk(value, path_tokens + [str(key)])
            return

        if isinstance(node, list):
            for index, item in enumerate(node):
                _walk(item, path_tokens + [f"[{index}]"])

    _walk(entities, [])
    return flattened


def _cross_verify_doc_key(doc: Dict[str, Any]) -> str:
    raw = str(doc.get("doc_type") or doc.get("document_type") or "")
    key = _snake_doc_type(raw)
    aliases = {
        "p_a_n_card": "pan_card",
        "voter_i_d": "voter_id",
        "driving_license": "driving_licence",
    }
    return aliases.get(key, key)


def _split_cross_verify_path(path: str) -> List[Any]:
    parts: List[Any] = []
    for segment in str(path or "").split("."):
        token = segment.strip()
        if not token:
            continue
        chunk = ""
        idx_buffer = ""
        in_index = False
        for ch in token:
            if ch == "[":
                if chunk:
                    parts.append(chunk)
                    chunk = ""
                in_index = True
                idx_buffer = ""
            elif ch == "]":
                if in_index and idx_buffer.isdigit():
                    parts.append(int(idx_buffer))
                in_index = False
                idx_buffer = ""
            else:
                if in_index:
                    idx_buffer += ch
                else:
                    chunk += ch
        if chunk:
            if chunk.isdigit():
                parts.append(int(chunk))
            else:
                parts.append(chunk)
        elif token.isdigit():
            parts.append(int(token))
    return parts


def _walk_cross_verify_path(node: Any, path: str) -> Any:
    current = node
    for part in _split_cross_verify_path(path):
        if isinstance(part, int):
            if isinstance(current, list) and 0 <= part < len(current):
                current = current[part]
            else:
                return None
        else:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
    return current


def _cross_verify_path_variants(path: str) -> List[str]:
    variants = [path]
    if str(path).startswith("extracted_entities."):
        variants.append(str(path)[len("extracted_entities."):])
    else:
        variants.append(f"extracted_entities.{path}")
    deduped: List[str] = []
    for item in variants:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _extract_canonical_field_from_entities(entities: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
    for candidate_path in _cross_verify_path_variants(path):
        node = _walk_cross_verify_path(entities, candidate_path)
        if node is None:
            continue
        if is_extraction_field_object(node):
            value = _extract_scalar_from_node(node.get("value"))
            if not value:
                continue
            extraction_conf = _safe_float(node.get("extraction_confidence"), 0.0)
            ocr_conf = _safe_float(node.get("ocr_confidence"), 0.0)
            ocr_ratio = ocr_conf / 100.0 if ocr_conf > 1.0 else ocr_conf
            ocr_ratio = max(0.0, min(1.0, ocr_ratio))
            return {
                "path": candidate_path,
                "value": value,
                "source_text": _extract_scalar_from_node(node.get("source_text")),
                "ocr_confidence": node.get("ocr_confidence"),
                "extraction_confidence": node.get("extraction_confidence"),
                "confidence_score": round((0.7 * extraction_conf) + (0.3 * ocr_ratio), 4),
            }
        scalar = _extract_scalar_from_node(node)
        if scalar:
            return {
                "path": candidate_path,
                "value": scalar,
                "source_text": None,
                "ocr_confidence": None,
                "extraction_confidence": None,
                "confidence_score": 0.0,
            }
    return None


def _aadhaar_values_match(left: str, right: str) -> bool:
    left = str(left or "")
    right = str(right or "")
    if not left or not right:
        return False
    left_digits = "".join(ch for ch in left if ch.isdigit())
    right_digits = "".join(ch for ch in right if ch.isdigit())
    if not left_digits or not right_digits:
        return False
    left_masked = "x" in left.lower()
    right_masked = "x" in right.lower()
    if not left_masked and not right_masked:
        return left_digits == right_digits
    return len(left_digits) >= 4 and len(right_digits) >= 4 and left_digits[-4:] == right_digits[-4:]


def _canonical_field_values_match(field_id: str, kind: str, left: str, right: str) -> bool:
    if field_id.endswith("aadhaar_number"):
        return _aadhaar_values_match(left, right)
    return str(left or "") == str(right or "")


def _group_cross_verify_values(compare_pool: List[Dict[str, Any]], field_id: str, kind: str) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for item in compare_pool:
        placed = False
        for group in groups:
            if _canonical_field_values_match(
                field_id,
                kind,
                str(item.get("normalized_value") or ""),
                str(group.get("normalized_value") or ""),
            ):
                group["documents"].append(item)
                # Prefer a fuller representative token for readability.
                current_norm = str(group.get("normalized_value") or "")
                incoming_norm = str(item.get("normalized_value") or "")
                if len(incoming_norm) > len(current_norm):
                    group["normalized_value"] = incoming_norm
                placed = True
                break
        if not placed:
            groups.append({
                "normalized_value": str(item.get("normalized_value") or ""),
                "documents": [item],
            })
    return groups


def _build_semantic_key_for_field(doc: Dict[str, Any], field: Dict[str, Any]) -> Tuple[str, str, str]:
    path_tokens = [
        _normalize_key_token(token)
        for token in field.get("path_tokens", [])
        if token and not str(token).startswith("[")
    ]
    if not path_tokens:
        return "", "", "text"

    canonical_leaf = _canonical_leaf_key(field.get("leaf", ""))
    subject = _infer_subject_from_path(str(doc.get("doc_type") or ""), path_tokens, canonical_leaf)
    kind = _infer_field_kind(canonical_leaf, path_tokens)

    if not canonical_leaf:
        normalized_path = ".".join(path_tokens)
        semantic_key = f"path.{normalized_path}"
    else:
        semantic_key = f"{subject}.{canonical_leaf}"

    label = semantic_key.replace("_", " ").replace(".", " / ").title()
    return semantic_key, label, kind


def _build_pipeline_doc_payloads(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload_docs: List[Dict[str, Any]] = []
    for doc in docs:
        result = doc.get("result")
        if not isinstance(result, dict):
            continue
        extracted_data = result.get("extracted_data")
        if not isinstance(extracted_data, dict):
            continue

        entities = _extract_entities_from_payload(extracted_data)
        fallback_type = result.get("doc_type") or doc.get("doc_type") or ""
        raw_document_type = str(
            extracted_data.get("document_type")
            or DOC_TYPE_MAPPING.get(str(fallback_type), fallback_type)
            or fallback_type
        ).strip()
        if not raw_document_type:
            raw_document_type = "UnknownDocument"
        normalized_doc_type = _snake_doc_type(raw_document_type)

        pages = result.get("pages")
        ocr_text = _collect_ocr_text_from_pages(pages)
        if not ocr_text:
            ocr_text = _collect_ocr_text_from_entities(entities)

        validation_flags = extracted_data.get("validation_flags")
        if not isinstance(validation_flags, list):
            validation_flags = []
        validation_flags = [str(flag) for flag in validation_flags if str(flag).strip()]
        if doc.get("status") == "validation_error" and "VALIDATION_ERROR" not in validation_flags:
            validation_flags.append("VALIDATION_ERROR")

        global_metrics = result.get("global_metrics") if isinstance(result.get("global_metrics"), dict) else {}
        metadata = {
            "doc_id": doc.get("doc_id"),
            "request_id": doc.get("request_id"),
            "filename": doc.get("filename"),
            "status": doc.get("status"),
            "pages": _safe_int(extracted_data.get("pages"), _safe_int(global_metrics.get("total_pages"), 0)),
            "global_metrics": global_metrics,
        }

        payload_docs.append({
            "doc_id": doc.get("doc_id"),
            "request_id": doc.get("request_id"),
            "doc_type": normalized_doc_type,
            "document_type": raw_document_type,
            "ocr_confidence": _normalize_ocr_confidence_ratio(extracted_data.get("ocr_confidence")),
            "ocr_text": ocr_text,
            "metadata": metadata,
            "validation_flags": validation_flags,
            "extracted_entities": entities,
        })
    return payload_docs


def _collect_claim_core_from_docs(payload_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    joined_entities = {"docs": [doc.get("extracted_entities", {}) for doc in payload_docs]}

    claimant_name = _extract_party_name(joined_entities, "claimant")
    claimant_relationship = _walk_key_match_integration(
        joined_entities,
        ["RelationshipWithLifeAssured", "RelationshipWithDeceased", "Relationship"],
    )
    bank_account = _walk_key_match_integration(joined_entities, ["AccountNumber"])
    bank_name = _walk_key_match_integration(joined_entities, ["BankName"])

    life_assured_name = _extract_party_name(joined_entities, "life_assured")
    life_assured_age = _walk_key_match_integration(joined_entities, ["Age"])
    life_assured_occupation = _walk_key_match_integration(joined_entities, ["Occupation", "OccupationCategory", "Designation"])
    sum_assured = _walk_key_match_integration(joined_entities, ["SumAssured"])

    date_of_death = _walk_key_match_integration(joined_entities, ["DateOfDeath", "DateOfDischargeOrDeath"])
    cause_of_death = _walk_key_match_integration(
        joined_entities,
        ["CauseOfDeath", "PrimaryCauseOfDeath", "ImmediateCause", "FinalOpinion"],
    )
    place_of_death = _walk_key_match_integration(joined_entities, ["PlaceOfDeath"])
    hospital_name = _walk_key_match_integration(joined_entities, ["HospitalName", "InstitutionName"])
    hospital_address = _walk_key_match_integration(joined_entities, ["HospitalAddress", "InstitutionAddress"])
    doctor_name = _walk_key_match_integration(joined_entities, ["DoctorName", "AttendingDoctor", "DiagnosingDoctorName"])
    doctor_registration_number = _walk_key_match_integration(joined_entities, ["DoctorRegistrationNumber", "RegistrationNumber", "NMRNumber", "DoctorRegNo"])
    hospital_rohini_id = _walk_key_match_integration(joined_entities, ["HospitalRohiniId", "RohiniId", "ROHINIID"])
    fir_number = _walk_key_match_integration(joined_entities, ["FIRNumber"])

    return {
        "claimant": {
            "name": claimant_name or "",
            "relationship_to_life_assured": claimant_relationship or "",
            "bank_account_number": bank_account or "",
            "bank_name": bank_name or "",
        },
        "life_assured": {
            "name": life_assured_name or "",
            "age": _safe_int(life_assured_age, 0),
            "occupation": life_assured_occupation or "",
            "sum_assured": _safe_float(sum_assured, 0.0),
        },
        "death_information": {
            "date_of_death": date_of_death or "",
            "cause_of_death": cause_of_death or "",
            "place_of_death": place_of_death or "",
            "hospital_name": hospital_name or "",
            "hospital_address": hospital_address or "",
            "doctor_name": doctor_name or "",
            "doctor_registration_number": doctor_registration_number or "",
            "hospital_rohini_id": hospital_rohini_id or "",
            "fir_number": fir_number or "",
        },
    }


def _build_medical_records_from_docs(payload_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    medical_doc_types = {
        "medico_legal_cause_of_death_certificate",
        "discharge_summary",
        "past_medical_records_and_treatment_papers",
        "medical_attendant_hospital_certificate",
        "postmortem_report",
        "viscera_chemical_examination_report",
        "admission_form",
        "diagnostic_test_report",
        "indoor_case_papers",
    }
    for doc in payload_docs:
        doc_type = str(doc.get("doc_type") or "")
        if doc_type not in medical_doc_types:
            continue
        entities = doc.get("extracted_entities", {})
        diagnosis = _walk_key_match_integration(
            entities,
            ["FinalDiagnosis", "ProvisionalDiagnosis", "ImmediateCause", "CauseOfDeath", "Diagnosis"],
        ) or ""
        treatment = _walk_key_match_integration(entities, ["TreatmentGiven", "Treatment"]) or ""
        date_value = _walk_key_match_integration(
            entities,
            ["DateOfAdmission", "DateOfDischargeOrDeath", "Date", "DateOfReport"],
        ) or ""
        record = {
            "record_type": doc_type,
            "hospital_name": _walk_key_match_integration(entities, ["HospitalName", "InstitutionName"]) or "",
            "doctor_name": _walk_key_match_integration(entities, ["DoctorName", "AttendingDoctor", "DiagnosingDoctorName"]) or "",
            "diagnosis": diagnosis,
            "treatment": treatment,
            "content_summary": diagnosis or treatment,
            "admission_date": _walk_key_match_integration(entities, ["DateOfAdmission"]) or "",
            "discharge_date": _walk_key_match_integration(entities, ["DateOfDischargeOrDeath"]) or "",
            "date": date_value,
            "ocr_text": str(doc.get("ocr_text") or "")[:4000],
        }
        if any(value for key, value in record.items() if key != "record_type"):
            records.append(record)
    return records


def _build_fir_records_from_docs(payload_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for doc in payload_docs:
        doc_type = str(doc.get("doc_type") or "")
        if "fir" not in doc_type and "inquest" not in doc_type and "police" not in doc_type:
            continue
        entities = doc.get("extracted_entities", {})
        fir_number = _walk_key_match_integration(entities, ["FIRNumber", "RegistrationNumber"])
        if not fir_number:
            continue
        records.append({
            "fir_number": fir_number,
            "police_station": _walk_key_match_integration(entities, ["PoliceStationName", "PoliceStation"]) or "",
            "date_filed": _walk_key_match_integration(entities, ["DateOfFIR", "DateOfRegistration"]) or "",
            "date": _walk_key_match_integration(entities, ["DateOfFIR", "DateOfRegistration"]) or "",
            "description": _walk_key_match_integration(
                entities,
                ["Description", "NatureOfIncident", "IncidentDescription"],
            ) or "",
            "incident_description": _walk_key_match_integration(
                entities,
                ["Description", "NatureOfIncident", "IncidentDescription"],
            ) or "",
            "location": _walk_key_match_integration(entities, ["Location", "PlaceOfIncident"]) or "",
        })
    return records


def _build_cross_verify_skipped(reason: str, mapping_scope: str) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "decision": "NOT_APPLICABLE",
        "review_required": False,
        "overall_match_score": None,
        "comparable_fields": 0,
        "matched_fields_count": 0,
        "mismatched_fields_count": 0,
        "mismatched_fields": [],
        "field_results": [],
        "settings": {
            "min_confidence": max(0.0, min(1.0, parse_float_env("CROSS_VERIFY_MIN_CONFIDENCE", 0.35))),
            "name_honorifics_removed": sorted(NAME_HONORIFICS),
            "mapping_scope": mapping_scope,
        },
        "updated_at": time.time(),
        "reason": reason,
    }


def _detect_stage2_cross_verify_branch(payload_docs: List[Dict[str, Any]]) -> str:
    joined_entities = {"docs": [doc.get("extracted_entities", {}) for doc in payload_docs]}
    category = (_walk_key_match_integration(joined_entities, ["DeathCategory"]) or "").strip().upper()
    if category.startswith("UNNATURAL"):
        return "unnatural_death"
    if category.startswith("NATURAL"):
        return "natural_death"

    doc_types = {str(doc.get("doc_type") or "").strip() for doc in payload_docs}
    if doc_types & STAGE2_UNNATURAL_DOC_TYPES:
        return "unnatural_death"
    if doc_types & STAGE2_NATURAL_DOC_TYPES:
        return "natural_death"
    return "natural_death"


def _select_stage2_cross_verify_map(payload_docs: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    stage2_map = get_stage2_cross_verify_map()
    branch = _detect_stage2_cross_verify_branch(payload_docs)
    selected = stage2_map.get(branch, {})
    if not isinstance(selected, dict):
        selected = {}
    selected.setdefault("fields", {})
    selected.setdefault("documents", {})
    selected.setdefault("version", stage2_map.get("version", 0))
    selected.setdefault("mapping_scope", branch)
    return selected, branch


def _build_cross_document_verification(
    payload_docs: List[Dict[str, Any]],
    total_docs: int,
    terminal_docs: int,
    map_payload: Optional[Dict[str, Any]] = None,
    mapping_scope: str = "stage1",
) -> Dict[str, Any]:
    min_confidence = max(0.0, min(1.0, parse_float_env("CROSS_VERIFY_MIN_CONFIDENCE", 0.35)))
    map_payload = map_payload or get_canonical_cross_verify_map()
    map_fields = map_payload.get("fields", {}) if isinstance(map_payload.get("fields"), dict) else {}
    map_docs = map_payload.get("documents", {}) if isinstance(map_payload.get("documents"), dict) else {}

    grouped: Dict[str, Dict[str, Any]] = {}
    for field_id, cfg in map_fields.items():
        grouped[str(field_id)] = {
            "field_label": str(cfg.get("label") or field_id),
            "kind": str(cfg.get("kind") or "text"),
            "evidence": [],
            "missing": [],
            "expected_docs": [],
        }

    document_logs: List[Dict[str, Any]] = []
    for doc in payload_docs:
        doc_key = _cross_verify_doc_key(doc)
        doc_mapping = map_docs.get(doc_key)
        entities = doc.get("extracted_entities")
        if not isinstance(doc_mapping, dict) or not isinstance(entities, dict):
            continue

        doc_id = doc.get("doc_id")
        doc_ref = {
            "doc_id": doc_id,
            "request_id": doc.get("request_id"),
            "document_type": doc.get("document_type"),
            "doc_type": doc.get("doc_type"),
        }
        present_fields: List[str] = []
        missing_fields: List[str] = []

        for field_id, raw_paths in doc_mapping.items():
            entry = grouped.get(str(field_id))
            if not entry:
                continue
            paths: List[str] = []
            if isinstance(raw_paths, list):
                paths = [str(path) for path in raw_paths if str(path).strip()]
            elif isinstance(raw_paths, str) and raw_paths.strip():
                paths = [raw_paths.strip()]
            if not paths:
                continue

            entry["expected_docs"].append(doc_ref)

            selected = None
            for path in paths:
                candidate = _extract_canonical_field_from_entities(entities, path)
                if not candidate:
                    continue
                normalized_value = _normalize_for_cross_compare(candidate.get("value", ""), entry.get("kind", "text"))
                if not normalized_value:
                    continue
                selected = {
                    "doc_id": doc_id,
                    "request_id": doc.get("request_id"),
                    "document_type": doc.get("document_type"),
                    "doc_type": doc.get("doc_type"),
                    "path": candidate.get("path"),
                    "value": candidate.get("value"),
                    "normalized_value": normalized_value,
                    "source_text": candidate.get("source_text"),
                    "ocr_confidence": candidate.get("ocr_confidence"),
                    "extraction_confidence": candidate.get("extraction_confidence"),
                    "confidence_score": candidate.get("confidence_score"),
                }
                break

            if selected:
                entry["evidence"].append(selected)
                present_fields.append(str(field_id))
            else:
                entry["missing"].append({
                    **doc_ref,
                    "expected_paths": paths,
                })
                missing_fields.append(str(field_id))

        document_logs.append({
            **doc_ref,
            "mapped_fields_count": len(present_fields) + len(missing_fields),
            "present_fields": present_fields,
            "missing_fields": missing_fields,
        })

    field_results: List[Dict[str, Any]] = []
    mismatched_fields: List[Dict[str, Any]] = []
    comparable_count = 0
    matched_count = 0

    for semantic_key in sorted(grouped.keys()):
        entry = grouped[semantic_key]
        evidence = entry.get("evidence", [])
        expected_docs = entry.get("expected_docs", [])
        missing_docs = entry.get("missing", [])
        if not expected_docs:
            continue

        best_by_doc: Dict[str, Dict[str, Any]] = {}
        for item in evidence:
            doc_id = str(item.get("doc_id") or item.get("document_type") or item.get("doc_type") or "unknown")
            previous = best_by_doc.get(doc_id)
            if previous is None or _safe_float(item.get("confidence_score"), 0.0) > _safe_float(previous.get("confidence_score"), 0.0):
                best_by_doc[doc_id] = item

        selected = list(best_by_doc.values())
        high_conf_selected = [
            item for item in selected
            if _safe_float(item.get("confidence_score"), 0.0) >= min_confidence
        ]
        compare_pool = high_conf_selected if len(high_conf_selected) >= 2 else selected

        status = "insufficient_evidence"
        conflicting_groups: List[Dict[str, Any]] = []
        if len(compare_pool) >= 2:
            comparable_count += 1
            value_groups = _group_cross_verify_values(compare_pool, semantic_key, str(entry.get("kind") or "text"))
            if len(value_groups) == 1:
                status = "match"
                matched_count += 1
            else:
                status = "mismatch"
                for group in value_groups:
                    norm_value = group.get("normalized_value")
                    items = group.get("documents") or []
                    conflicting_groups.append({
                        "normalized_value": norm_value,
                        "values": sorted({str(sample.get("value") or "") for sample in items}),
                        "documents": items,
                    })
                mismatched_fields.append({
                    "field_id": semantic_key,
                    "field_label": entry.get("field_label"),
                    "comparison_mode": "high_confidence" if compare_pool is high_conf_selected else "all_evidence",
                    "values": sorted({str(item.get("value") or "") for item in compare_pool}),
                    "conflicting_groups": conflicting_groups,
                    "documents": compare_pool,
                })

        field_results.append({
            "field_id": semantic_key,
            "field_label": entry.get("field_label"),
            "kind": entry.get("kind"),
            "status": status,
            "comparison_mode": "high_confidence" if compare_pool is high_conf_selected else "all_evidence",
            "expected_documents_count": len(expected_docs),
            "documents_with_value_count": len(selected),
            "evidence_count": len(selected),
            "evidence": selected,
            "missing_in_documents": missing_docs,
        })

    overall_match_score = round((matched_count * 100.0) / comparable_count, 2) if comparable_count > 0 else None
    review_required = bool(mismatched_fields) or comparable_count == 0
    decision = "MATCH" if overall_match_score is not None and overall_match_score >= 80.0 and not mismatched_fields else "MANUAL_REVIEW"
    status = "completed" if total_docs > 0 and terminal_docs >= total_docs else "processing"

    return {
        "status": status,
        "decision": decision,
        "review_required": review_required,
        "overall_match_score": overall_match_score,
        "comparable_fields": comparable_count,
        "matched_fields_count": matched_count,
        "mismatched_fields_count": len(mismatched_fields),
        "mismatched_fields": mismatched_fields,
        "field_results": field_results,
        "document_field_log": document_logs,
        "settings": {
            "min_confidence": min_confidence,
            "name_honorifics_removed": sorted(NAME_HONORIFICS),
            "mapping_version": map_payload.get("version"),
            "mapping_scope": mapping_scope,
        },
        "updated_at": time.time(),
    }


def _build_next_step_claim_case_payload(
    batch_id: str,
    payload_docs: List[Dict[str, Any]],
    cross_verification: Dict[str, Any],
    stage2_cross_verification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    core = _collect_claim_core_from_docs(payload_docs)
    submitted_documents: List[Dict[str, Any]] = []
    ocr_confidence_scores: Dict[str, float] = {}
    aggregate_flags: List[str] = []

    for doc in payload_docs:
        doc_type = str(doc.get("doc_type") or "unknown_document")
        ocr_confidence = _normalize_ocr_confidence_ratio(doc.get("ocr_confidence"))
        current = ocr_confidence_scores.get(doc_type)
        if current is None or ocr_confidence > current:
            ocr_confidence_scores[doc_type] = ocr_confidence

        validation_flags = [str(flag) for flag in (doc.get("validation_flags") or []) if str(flag).strip()]
        aggregate_flags.extend(validation_flags)
        entities = doc.get("extracted_entities", {})
        submitted_documents.append({
            "doc_type": doc_type,
            "document_type": doc.get("document_type") or "",
            "ocr_confidence": ocr_confidence,
            "ocr_text": str(doc.get("ocr_text") or "")[:12000],
            "metadata": doc.get("metadata") or {},
            "validation_flags": validation_flags,
            "hospital_rohini_id": _walk_key_match_integration(entities, ["HospitalRohiniId", "RohiniId", "ROHINIID"]),
            "doctor_registration_number": _walk_key_match_integration(entities, ["DoctorRegistrationNumber", "RegistrationNumber", "NMRNumber", "DoctorRegNo"]),
            "extracted_entities": entities,
        })

    mismatch_ids = [item.get("field_id") for item in cross_verification.get("mismatched_fields", []) if item.get("field_id")]
    stage2_mismatch_ids: List[str] = []
    if isinstance(stage2_cross_verification, dict):
        stage2_mismatch_ids = [
            item.get("field_id")
            for item in stage2_cross_verification.get("mismatched_fields", [])
            if item.get("field_id")
        ]
    for field_id in mismatch_ids:
        aggregate_flags.append(f"CROSS_DOC_MISMATCH:{field_id}")
    for field_id in stage2_mismatch_ids:
        aggregate_flags.append(f"CROSS_DOC_MISMATCH_STAGE2:{field_id}")

    deduped_flags: List[str] = []
    for flag in aggregate_flags:
        if flag not in deduped_flags:
            deduped_flags.append(flag)

    return {
        "claim_case_id": batch_id,
        "claim_source": "stage_pipeline",
        "claimant": core["claimant"],
        "life_assured": core["life_assured"],
        "death_information": core["death_information"],
        "submitted_documents": submitted_documents,
        "medical_records": _build_medical_records_from_docs(payload_docs),
        "fir_records": _build_fir_records_from_docs(payload_docs),
        "ocr_confidence_scores": ocr_confidence_scores,
        "validation_flags": deduped_flags,
        "cross_document_verification": {
            "overall_match_score": cross_verification.get("overall_match_score"),
            "decision": cross_verification.get("decision"),
            "review_required": cross_verification.get("review_required"),
            "mismatched_fields": mismatch_ids,
            "stage1": {
                "overall_match_score": cross_verification.get("overall_match_score"),
                "decision": cross_verification.get("decision"),
                "review_required": cross_verification.get("review_required"),
                "mismatched_fields": mismatch_ids,
            },
            "stage2": {
                "overall_match_score": (
                    stage2_cross_verification.get("overall_match_score")
                    if isinstance(stage2_cross_verification, dict)
                    else None
                ),
                "decision": (
                    stage2_cross_verification.get("decision")
                    if isinstance(stage2_cross_verification, dict)
                    else "NOT_APPLICABLE"
                ),
                "review_required": (
                    stage2_cross_verification.get("review_required")
                    if isinstance(stage2_cross_verification, dict)
                    else False
                ),
                "mismatched_fields": stage2_mismatch_ids,
            },
        },
        "prepared_at": time.time(),
    }


def _batch_face_verification_ready(batch: Dict[str, Any]) -> bool:
    face_verification = batch.get("face_verification") or {}
    if not face_verification.get("enabled"):
        return True
    return is_face_status_terminal(face_verification.get("status"))


def recompute_pipeline_batch_locked(batch: Dict[str, Any]) -> None:
    docs = list(batch.get("docs", {}).values())
    total = len(docs)
    terminal = sum(1 for doc in docs if doc.get("status") in PIPELINE_TERMINAL_STATUSES)
    failed = sum(1 for doc in docs if doc.get("status") == "failed")
    validation_error = sum(1 for doc in docs if doc.get("status") == "validation_error")
    success = sum(1 for doc in docs if doc.get("status") == "success")
    batch["counts"] = {
        "total": total,
        "terminal": terminal,
        "success": success,
        "failed": failed,
        "validation_error": validation_error,
        "ocr_queue": OCR_QUEUE.qsize(),
        "llm_queue": LLM_QUEUE.qsize(),
        "validation_queue": VALIDATION_QUEUE.qsize(),
        "face_queue": FACE_QUEUE.qsize(),
    }
    batch["status"] = "completed" if total > 0 and terminal == total else "processing"
    batch["updated_at"] = time.time()

    stage1_docs = [doc for doc in docs if is_stage1_internal_doc_type(doc.get("doc_type"))]
    stage2_docs = [doc for doc in docs if is_stage2_internal_doc_type(doc.get("doc_type"))]

    stage1_total = len(stage1_docs)
    stage1_terminal = sum(1 for doc in stage1_docs if doc.get("status") in PIPELINE_TERMINAL_STATUSES)
    stage2_total = len(stage2_docs)
    stage2_terminal = sum(1 for doc in stage2_docs if doc.get("status") in PIPELINE_TERMINAL_STATUSES)

    stage1_payload_docs = _build_pipeline_doc_payloads(stage1_docs)
    stage2_payload_docs = _build_pipeline_doc_payloads(stage2_docs)
    payload_docs = list(stage1_payload_docs) + list(stage2_payload_docs)

    if stage1_total > 0:
        cross_verification = _build_cross_document_verification(
            stage1_payload_docs,
            stage1_total,
            stage1_terminal,
            map_payload=get_canonical_cross_verify_map(),
            mapping_scope="stage1",
        )
    else:
        cross_verification = _build_cross_verify_skipped(
            "No Stage 1 documents available in this batch.",
            mapping_scope="stage1",
        )

    if stage2_total > 0 and stage2_payload_docs:
        stage2_map_payload, stage2_scope = _select_stage2_cross_verify_map(stage2_payload_docs)
        cross_verification_stage2 = _build_cross_document_verification(
            stage2_payload_docs,
            stage2_total,
            stage2_terminal,
            map_payload=stage2_map_payload,
            mapping_scope=stage2_scope,
        )
    elif stage2_total > 0:
        cross_verification_stage2 = _build_cross_verify_skipped(
            "Stage 2 documents exist but none have extractable entities yet.",
            mapping_scope="stage2",
        )
    else:
        cross_verification_stage2 = _build_cross_verify_skipped(
            "No Stage 2 documents available in this batch.",
            mapping_scope="stage2",
        )

    batch["cross_document_verification"] = cross_verification
    batch["cross_document_verification_stage2"] = cross_verification_stage2
    batch["next_step_claim_case_payload"] = _build_next_step_claim_case_payload(
        batch.get("batch_id", ""),
        payload_docs,
        cross_verification,
        cross_verification_stage2,
    )

    auto_analysis_enabled = parse_bool_env("AUTO_FRAUD_ANALYSIS", True)
    if (
        auto_analysis_enabled
        and batch["status"] == "completed"
        and _batch_face_verification_ready(batch)
        and batch.get("analysis_result") is None
    ):
        payload = batch.get("next_step_claim_case_payload")
        if isinstance(payload, dict):
            batch["analysis_result"] = {
                "status": "running",
                "started_at": time.time(),
            }
            bid = batch.get("batch_id", "")
            threading.Thread(
                target=_run_fraud_and_policy_background,
                args=(bid, payload),
                daemon=True,
            ).start()
            print(f"[AUTO-ANALYSIS TRIGGERED] batch={bid} docs={total}")

def update_pipeline_doc(batch_id: str, doc_id: str, **updates) -> None:
    with PIPELINE_LOCK:
        batch = PIPELINE_BATCHES.get(batch_id)
        if not batch:
            return
        doc = batch["docs"].get(doc_id)
        if not doc:
            return
        doc.update(updates)
        doc["updated_at"] = time.time()
        recompute_pipeline_batch_locked(batch)

def update_pipeline_face_verification(batch_id: str, **updates) -> None:
    with PIPELINE_LOCK:
        batch = PIPELINE_BATCHES.get(batch_id)
        if not batch:
            return
        current = batch.get("face_verification") or {}
        current.update(updates)
        current["updated_at"] = time.time()
        batch["face_verification"] = current
        recompute_pipeline_batch_locked(batch)

def get_pipeline_batch_public(batch_id: str) -> Optional[Dict[str, Any]]:
    with PIPELINE_LOCK:
        batch = PIPELINE_BATCHES.get(batch_id)
        if not batch:
            return None
        recompute_pipeline_batch_locked(batch)
        return {
            "batch_id": batch["batch_id"],
            "status": batch["status"],
            "created_at": batch["created_at"],
            "updated_at": batch["updated_at"],
            "counts": batch["counts"],
            "workers": batch["workers"],
            "docs": list(batch["docs"].values()),
            "face_verification": batch.get("face_verification"),
            "cross_document_verification": batch.get("cross_document_verification"),
            "cross_document_verification_stage2": batch.get("cross_document_verification_stage2"),
            "next_step_claim_case_payload": batch.get("next_step_claim_case_payload"),
            "analysis_result": batch.get("analysis_result"),  # fraud + policy result (auto-populated)
        }

def build_face_only_document_result(doc_type: str, filename: str) -> Dict[str, Any]:
    return {
        "success": True,
        "doc_type": doc_type,
        "filename": filename,
        "face_only": True,
        "message": "Captured for face verification only. OCR and JSON extraction are not required.",
        "pages": [],
        "global_metrics": {},
        "pipeline_metrics": {},
        "extracted_data": None,
    }

def pipeline_ocr_worker(worker_id: int) -> None:
    while True:
        job = OCR_QUEUE.get()
        try:
            batch_id = job["batch_id"]
            doc_id = job["doc_id"]
            doc_type = job["doc_type"]
            filename = job["filename"]
            request_id = job["request_id"]
            update_pipeline_doc(batch_id, doc_id, status="ocr_running", stage="ocr")
            print(f"[PIPELINE OCR START] worker={worker_id} req={request_id} doc={doc_type} file={filename}")
            ocr_result = process_document_ocr_only(
                file_bytes=job["file_bytes"],
                filename=filename,
                doc_type=doc_type,
                request_id=request_id,
            )
            update_pipeline_doc(
                batch_id,
                doc_id,
                status="queued_llm",
                stage="llm",
                result=ocr_result,
                ocr_done_at=time.time(),
            )
            LLM_QUEUE.put({
                "batch_id": batch_id,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "filename": filename,
                "request_id": request_id,
                "ocr_result": ocr_result,
            })
            print(f"[PIPELINE OCR DONE] worker={worker_id} req={request_id} doc={doc_type} -> llm_queue")
        except Exception as exc:
            update_pipeline_doc(
                job.get("batch_id"),
                job.get("doc_id"),
                status="failed",
                stage="ocr",
                errorMsg=str(exc),
            )
            print(f"[PIPELINE OCR ERROR] worker={worker_id} error={exc}")
        finally:
            OCR_QUEUE.task_done()

def pipeline_llm_worker(worker_id: int) -> None:
    while True:
        job = LLM_QUEUE.get()
        try:
            batch_id = job["batch_id"]
            doc_id = job["doc_id"]
            doc_type = job["doc_type"]
            filename = job["filename"]
            request_id = job["request_id"]
            ocr_result = job["ocr_result"]
            update_pipeline_doc(batch_id, doc_id, status="llm_running", stage="llm")
            print(f"[PIPELINE LLM START] worker={worker_id} req={request_id} doc={doc_type} file={filename}")
            extraction_result = run_json_extraction_from_pages(
                doc_type=doc_type,
                pages_result=ocr_result.get("pages", []),
                filename=filename,
                request_id=request_id,
                validate_output=False,
            )
            update_pipeline_doc(
                batch_id,
                doc_id,
                status="queued_validation",
                stage="validation",
                llm_result=extraction_result,
            )
            VALIDATION_QUEUE.put({
                "batch_id": batch_id,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "filename": filename,
                "request_id": request_id,
                "ocr_result": ocr_result,
                "extraction_result": extraction_result,
            })
            print(f"[PIPELINE LLM DONE] worker={worker_id} req={request_id} doc={doc_type} -> validation_queue")
        except Exception as exc:
            update_pipeline_doc(
                job.get("batch_id"),
                job.get("doc_id"),
                status="failed",
                stage="llm",
                errorMsg=str(exc),
            )
            print(f"[PIPELINE LLM ERROR] worker={worker_id} error={exc}")
        finally:
            LLM_QUEUE.task_done()

def pipeline_validation_worker(worker_id: int) -> None:
    while True:
        job = VALIDATION_QUEUE.get()
        try:
            batch_id = job["batch_id"]
            doc_id = job["doc_id"]
            doc_type = job["doc_type"]
            request_id = job["request_id"]
            ocr_result = job["ocr_result"]
            extraction_result = job["extraction_result"]
            update_pipeline_doc(batch_id, doc_id, status="validation_running", stage="validation")
            print(f"[PIPELINE VALIDATION START] worker={worker_id} req={request_id} doc={doc_type}")

            merged_result = {
                **ocr_result,
                **extraction_result,
                "pages": ocr_result.get("pages", []),
                "global_metrics": ocr_result.get("global_metrics", {}),
                "pipeline_metrics": {
                    **(ocr_result.get("pipeline_metrics") or {}),
                    **(extraction_result.get("pipeline_metrics") or {}),
                },
            }

            if extraction_result.get("success") is False:
                status = "validation_error" if extraction_result.get("status") == "REUPLOAD_REQUIRED" else "failed"
                update_pipeline_doc(
                    batch_id,
                    doc_id,
                    status=status,
                    stage="done",
                    result=merged_result,
                    errorMsg=extraction_result.get("message"),
                    completed_at=time.time(),
                )
                continue

            extracted_data = extraction_result.get("extracted_data")
            resolved_doc_type = extraction_result.get("doc_type") or doc_type
            if extracted_data:
                validation_result = validate_document(resolved_doc_type, extracted_data)
            else:
                validation_result = {"status": "OK"}

            if validation_result.get("status") != "OK":
                merged_result = {
                    **merged_result,
                    "success": False,
                    **validation_result,
                }
                update_pipeline_doc(
                    batch_id,
                    doc_id,
                    status="validation_error",
                    stage="done",
                    result=merged_result,
                    completed_at=time.time(),
                )
            else:
                merged_result["success"] = True
                update_pipeline_doc(
                    batch_id,
                    doc_id,
                    status="success",
                    stage="done",
                    result=merged_result,
                    completed_at=time.time(),
                )
            print(f"[PIPELINE VALIDATION DONE] worker={worker_id} req={request_id} doc={doc_type}")
        except Exception as exc:
            update_pipeline_doc(
                job.get("batch_id"),
                job.get("doc_id"),
                status="failed",
                stage="validation",
                errorMsg=str(exc),
            )
            print(f"[PIPELINE VALIDATION ERROR] worker={worker_id} error={exc}")
        finally:
            VALIDATION_QUEUE.task_done()

def pipeline_face_worker(worker_id: int) -> None:
    while True:
        job = FACE_QUEUE.get()
        try:
            batch_id = job["batch_id"]
            request_id = job["request_id"]
            documents = job["documents"]
            update_pipeline_face_verification(
                batch_id,
                status="running",
                decision="MANUAL_REVIEW",
                started_at=time.time(),
                worker_id=worker_id,
            )
            print(f"[FACE VERIFY START] worker={worker_id} req={request_id} docs={len(documents)}")
            result = run_stage1_face_verification(documents)
            update_pipeline_face_verification(
                batch_id,
                **result,
                completed_at=time.time(),
            )
            print(
                f"[FACE VERIFY DONE] worker={worker_id} req={request_id} "
                f"status={result.get('status')} decision={result.get('decision')} "
                f"confidence={result.get('overall_confidence')}"
            )
        except Exception as exc:
            update_pipeline_face_verification(
                job.get("batch_id"),
                status="error",
                decision="MANUAL_REVIEW",
                review_required=True,
                review_flags=["FACE_VERIFY_ERROR"],
                error=str(exc),
                completed_at=time.time(),
            )
            print(f"[FACE VERIFY ERROR] worker={worker_id} error={exc}")
        finally:
            FACE_QUEUE.task_done()

def ensure_pipeline_workers_started() -> None:
    global PIPELINE_WORKERS_STARTED
    if PIPELINE_WORKERS_STARTED:
        return
    with PIPELINE_WORKERS_LOCK:
        if PIPELINE_WORKERS_STARTED:
            return
        ocr_workers = max(1, min(parse_int_env("OCR_WORKERS", 4), 8))
        llm_workers = max(1, min(parse_int_env("LLM_WORKERS", 4), 8))
        validation_workers = max(1, min(parse_int_env("VALIDATION_WORKERS", 2), 8))
        face_workers = max(1, min(parse_int_env("FACE_VERIFY_WORKERS", 1), 4))
        for idx in range(ocr_workers):
            threading.Thread(target=pipeline_ocr_worker, args=(idx + 1,), daemon=True).start()
        for idx in range(llm_workers):
            threading.Thread(target=pipeline_llm_worker, args=(idx + 1,), daemon=True).start()
        for idx in range(validation_workers):
            threading.Thread(target=pipeline_validation_worker, args=(idx + 1,), daemon=True).start()
        for idx in range(face_workers):
            threading.Thread(target=pipeline_face_worker, args=(idx + 1,), daemon=True).start()
        PIPELINE_WORKERS_STARTED = True
        print(
            f"[PIPELINE WORKERS] ocr={ocr_workers} llm={llm_workers} "
            f"validation={validation_workers} face={face_workers} "
            f"llm_max_concurrency={LLM_MAX_CONCURRENCY}"
        )

@router.post("/ocr")
def run_claims_ocr(
    file: UploadFile = File(...),
    doc_type: str = Form("DEATH_CERTIFICATE"),
    engine: str = Form("auto"),  # kept for backwards compatibility but not used
    extract_json: bool = Form(True)
):
    doc_type = canonicalize_doc_type(doc_type)
    request_id = uuid.uuid4().hex[:8]
    request_started_at = time.time()
    filename = file.filename or "uploaded_file"
    file_extension = os.path.splitext(filename)[1].lower()
    print(
        f"[DOC START] req={request_id} doc={doc_type} file={filename} "
        f"thread={threading.get_ident()}"
    )

    file_bytes = file.file.read()
    strict_upload_validation = parse_bool_env("STRICT_UPLOAD_VALIDATION", False)
    strict_mime_validation = parse_bool_env("STRICT_MIME_VALIDATION", False)
    max_upload_bytes = parse_int_env("MAX_UPLOAD_BYTES", 15 * 1024 * 1024)
    allowed_extensions = set(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in parse_csv_env("ALLOWED_UPLOAD_EXTENSIONS", ".png,.jpg,.jpeg,.pdf")
    )
    if not allowed_extensions:
        allowed_extensions = {".png", ".jpg", ".jpeg", ".pdf"}

    if strict_upload_validation:
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(file_bytes) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max allowed size is {max_upload_bytes} bytes."
            )
        is_pdf_bytes = file_bytes.startswith(b"%PDF")
        if file_extension not in allowed_extensions and not is_pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension or 'unknown'}."
            )
        if strict_mime_validation:
            allowed_mime_types = {
                ".png": {"image/png"},
                ".jpg": {"image/jpeg"},
                ".jpeg": {"image/jpeg"},
                ".pdf": {"application/pdf"},
            }
            expected_mimes = allowed_mime_types.get(file_extension, set())
            if expected_mimes and file.content_type and file.content_type not in expected_mimes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported MIME type '{file.content_type}' for extension '{file_extension}'.",
                )

    pages_result = []
    pdf_parallel_threshold_pages = max(1, parse_int_env("PDF_PARALLEL_THRESHOLD_PAGES", 1))
    default_pdf_workers = min(4, os.cpu_count() or 2)
    pdf_parallel_workers = min(max(1, parse_int_env("PDF_PARALLEL_PAGE_WORKERS", default_pdf_workers)), 8)
    
    try:
        if filename.lower().endswith(".pdf") or file_bytes.startswith(b"%PDF"):
            if not HAS_PYMUPDF:
                raise HTTPException(
                    status_code=500,
                    detail="PyMuPDF (fitz) is not installed on the server. Cannot process PDF files."
                )
            
            # Open PDF
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            num_pages = len(pdf_doc)
            if num_pages == 0:
                raise HTTPException(status_code=400, detail="The uploaded PDF file has no pages.")
                
            # Always use rapidocr as the default engine as requested
            auto_engine = "rapidocr"
            
            def process_page(page_idx):
                page_started_at = time.time()
                print(
                    f"[PDF PAGE OCR START] req={request_id} doc={doc_type} "
                    f"file={filename} page={page_idx + 1}/{num_pages}"
                )
                page = pdf_doc.load_page(page_idx)
                text = page.get_text()
                
                if not needs_ocr(text):
                    # use native text directly
                    words = []
                    for w in text.split():
                        cleaned = w.strip(".,;:?!()[]{}*\"'")
                        if cleaned:
                            words.append({
                                "text": cleaned,
                                "confidence": 100.0,
                                "is_numeric": is_numeric_word(cleaned)
                            })
                    total_count = len(words)
                    num_count = sum(1 for w in words if w["is_numeric"])
                    text_count = total_count - num_count
                    
                    # Convert page to image solely for QR code detection
                    pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    detected_qrs = detect_qr_codes(img)
                    
                    page_ocr = {
                        "raw_text": text.strip(),
                        "words": words,
                        "qr_codes": detected_qrs,
                        "metrics": {
                            "overall_confidence": 100.0 if total_count > 0 else 0.0,
                            "text_confidence": 100.0 if text_count > 0 else 0.0,
                            "number_confidence": 100.0 if num_count > 0 else 0.0,
                            "word_count": total_count,
                            "text_count": text_count,
                            "number_count": num_count,
                            "processing_time_ms": 1
                        }
                    }
                    page_ocr["ocr_engine_used"] = "native_pdf_text"
                else:
                    # PDF to Image
                    pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    detected_qrs = detect_qr_codes(img)
                    page_ocr, page_engine = run_ocr_with_fallback(img, auto_engine)
                    page_ocr["ocr_engine_used"] = page_engine
                        
                    page_ocr["qr_codes"] = detected_qrs
                        
                page_ocr["page_number"] = page_idx + 1
                print(
                    f"[PDF PAGE OCR DONE] req={request_id} doc={doc_type} "
                    f"file={filename} page={page_idx + 1}/{num_pages} "
                    f"words={page_ocr.get('metrics', {}).get('word_count', 0)} "
                    f"ms={int((time.time() - page_started_at) * 1000)}"
                )
                return page_ocr
                
            if num_pages > pdf_parallel_threshold_pages:
                # Process in parallel for large documents.
                with ThreadPoolExecutor(max_workers=pdf_parallel_workers) as executor:
                    future_to_idx = {executor.submit(process_page, idx): idx for idx in range(num_pages)}
                    results_map = {}
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            results_map[idx] = future.result()
                        except Exception as exc:
                            print(f"Page {idx} generated an exception: {exc}")
                            raise
                            
                # Reconstruct correctly sorted pages_result
                for idx in range(num_pages):
                    if idx in results_map:
                        pages_result.append(results_map[idx])
            else:
                # Sequential processing for documents with <= 5 pages
                for idx in range(num_pages):
                    pages_result.append(process_page(idx))

        else:
            # Process single image
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            
            # Images are 1 page, so num_pages = 1 (which is <= 3), use rapidocr
            auto_engine = "rapidocr"
            
            detected_qrs = detect_qr_codes(img)
            page_ocr, page_engine = run_ocr_with_fallback(img, auto_engine)
            page_ocr["ocr_engine_used"] = page_engine
            page_ocr["qr_codes"] = detected_qrs
            page_ocr["page_number"] = 1
            pages_result.append(page_ocr)
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )
        
    # Calculate global document statistics across all pages
    total_words = 0
    total_texts = 0
    total_nums = 0
    
    weighted_overall_conf = 0.0
    weighted_text_conf = 0.0
    weighted_num_conf = 0.0
    
    total_processing_time = 0
    all_qr_codes = []
    
    for page in pages_result:
        metrics = page["metrics"]
        total_words += metrics["word_count"]
        total_texts += metrics["text_count"]
        total_nums += metrics["number_count"]
        
        weighted_overall_conf += metrics["overall_confidence"] * metrics["word_count"]
        weighted_text_conf += metrics["text_confidence"] * metrics["text_count"]
        weighted_num_conf += metrics["number_confidence"] * metrics["number_count"]
        total_processing_time += metrics["processing_time_ms"]
        
        if "qr_codes" in page and page["qr_codes"]:
            all_qr_codes.extend(page["qr_codes"])
            
    # Deduplicate QR codes across pages
    all_qr_codes = list(set(all_qr_codes))
        
    global_overall = round(weighted_overall_conf / total_words, 2) if total_words > 0 else 0.0
    global_text = round(weighted_text_conf / total_texts, 2) if total_texts > 0 else 0.0
    global_num = round(weighted_num_conf / total_nums, 2) if total_nums > 0 else 0.0
    engines_used = sorted(set(page.get("ocr_engine_used", auto_engine) for page in pages_result))
    engine_used = "+".join(engines_used) if engines_used else auto_engine

    ocr_stage_ms = int((time.time() - request_started_at) * 1000)
    print(
        f"[OCR DONE] req={request_id} doc={doc_type} file={filename} "
        f"engine={engine_used} words={total_words} ocr_ms={ocr_stage_ms}"
    )

    if not extract_json:
        return JSONResponse({
            "success": True,
            "filename": filename,
            "doc_type": doc_type,
            "engine_used": engine_used,
            "llm_provider": None,
            "llm_model": None,
            "llm_fallback_used": False,
            "llm_fallback_reason": None,
            "llm_metrics": {},
            "pipeline_metrics": {
                "ocr_stage_ms": ocr_stage_ms,
                "llm_stage_ms": 0,
                "rapid_ocr_max_concurrency": RAPID_OCR_MAX_CONCURRENCY,
                "pdf_parallel_workers": pdf_parallel_workers,
                "pdf_parallel_threshold_pages": pdf_parallel_threshold_pages,
                "total_request_ms": int((time.time() - request_started_at) * 1000),
            },
            "qr_codes": all_qr_codes,
            "extracted_data": None,
            "global_metrics": {
                "overall_confidence": global_overall,
                "text_confidence": global_text,
                "number_confidence": global_num,
                "total_pages": len(pages_result),
                "total_words": total_words,
                "total_texts": total_texts,
                "total_numbers": total_nums,
                "total_processing_time_ms": total_processing_time
            },
            "pages": pages_result
        })
    
    # Process LLM Extraction
    extracted_data = None
    llm_provider = None
    llm_model = None
    classification_ms = None
    extraction_ms = None
    fallback_extraction_ms = None
    llm_fallback_used = False
    llm_fallback_reason = None
    llm_stage_started_at = time.time()
    llm_stage_ms = None
    llm_input_char_limit = max(2000, parse_int_env("LLM_INPUT_CHAR_LIMIT", 30000))
    llm_request_timeout_s = max(5.0, parse_float_env("LLM_REQUEST_TIMEOUT_S", 45.0))
    llm_retry_count = max(0, parse_int_env("LLM_RETRY_COUNT", 1))
    llm_retry_backoff_ms = max(0, parse_int_env("LLM_RETRY_BACKOFF_MS", 400))
    llm_input_truncated = False
    clean_text_chars = 0
    conf_text_chars = 0
    parallel_json_enabled = False
    parallel_json_used = False

    def call_llm_with_retry(
        client: OpenAI,
        kwargs: Dict[str, Any],
        stage: str,
        provider_override: Optional[str] = None,
    ):
        total_attempts = llm_retry_count + 1
        provider_label = provider_override or llm_provider or "unknown"
        for attempt in range(1, total_attempts + 1):
            try:
                queue_started_at = time.time()
                with llm_request_lock:
                    queue_ms = int((time.time() - queue_started_at) * 1000)
                    if queue_ms > 50:
                        print(
                            f"[LLM QUEUE] req={request_id} doc={doc_type} provider={provider_label} "
                            f"stage={stage} wait_ms={queue_ms} max_concurrency={LLM_MAX_CONCURRENCY}"
                        )
                    print(
                        f"[LLM CALL START] req={request_id} doc={doc_type} provider={provider_label} "
                        f"model={kwargs.get('model')} stage={stage} attempt={attempt}/{total_attempts}"
                    )
                    call_started_at = time.time()
                    response = client.chat.completions.create(**kwargs)
                    print(
                        f"[LLM CALL DONE] req={request_id} doc={doc_type} provider={provider_label} "
                        f"model={kwargs.get('model')} stage={stage} ms={int((time.time() - call_started_at) * 1000)}"
                    )
                    return response
            except Exception as call_exc:
                transient = is_transient_llm_error(call_exc)
                print(
                    f"[LLM ERROR] req={request_id} doc={doc_type} provider={provider_label} stage={stage} "
                    f"attempt={attempt}/{total_attempts} transient={transient} error={call_exc}"
                )
                if attempt >= total_attempts or not transient:
                    raise
                time.sleep((llm_retry_backoff_ms * attempt) / 1000.0)

    try:
        schema_key = DOC_TYPE_MAPPING.get(doc_type)
        if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"]:
            schema_key = "GENERIC_IDENTITY"

        is_huggingface = False
        supports_response_format = False
        api_key = None

        hf_enabled = parse_bool_env("ENABLE_HUGGINGFACE", False)
        hf_api_key = get_next_hf_key() if hf_enabled else None
        is_huggingface = bool(hf_api_key)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        api_key = hf_api_key or openai_api_key
        if not api_key:
            print("[LLM ROUTER] provider=none (no API key configured)")
            print("Skipping LLM extraction: HF_API_KEY or OPENAI_API_KEY is not set.")
            api_key = None
        elif is_huggingface:
            llm_provider = "huggingface"
            llm_model = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
            print(f"[LLM ROUTER] provider=huggingface timeout_s={llm_request_timeout_s} retries={llm_retry_count}")
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=api_key,
                timeout=llm_request_timeout_s,
                max_retries=0,
            )
        else:
            llm_provider = "openai"
            llm_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            print(f"[LLM ROUTER] provider=openai timeout_s={llm_request_timeout_s} retries={llm_retry_count}")
            client = OpenAI(api_key=api_key, timeout=llm_request_timeout_s, max_retries=0)
            supports_response_format = True
        print(f"[LLM DECISION] provider={llm_provider or 'none'} model={llm_model or 'none'} schema_key={schema_key or 'none'}")

        if schema_key and api_key:
            # Build clean_text and conf_text
            clean_texts = []
            conf_texts = []
            clean_chars_used = 0
            conf_chars_used = 0
            input_limit_hit = False
            for page in pages_result:
                for w in page.get("words", []):
                    if "text" in w:
                        t = w["text"].strip()
                        try:
                            c = int(round(float(w.get("confidence", 0))))
                        except Exception:
                            c = 0
                        if t:
                            conf_token = f"{t}|{c}"
                            next_clean = clean_chars_used + (1 if clean_texts else 0) + len(t)
                            next_conf = conf_chars_used + (1 if conf_texts else 0) + len(conf_token)
                            if next_clean > llm_input_char_limit or next_conf > llm_input_char_limit:
                                input_limit_hit = True
                                break
                            clean_texts.append(t)
                            conf_texts.append(conf_token) # Using | separator as requested for cleaner tokenization
                            clean_chars_used = next_clean
                            conf_chars_used = next_conf
                if input_limit_hit:
                    break

            clean_text = " ".join(clean_texts)
            conf_text = " ".join(conf_texts)
            clean_text_chars = len(clean_text)
            conf_text_chars = len(conf_text)
            llm_input_truncated = input_limit_hit
            if input_limit_hit:
                print(
                    f"[LLM INPUT] truncated=true limit={llm_input_char_limit} "
                    f"clean_chars={clean_text_chars} conf_chars={conf_text_chars}"
                )

            # --- STAGE 1: DOCUMENT CLASSIFICATION ---
            # For fixed document uploads (e.g. DEATH_CERTIFICATE), skip classifier by default.
            # Classifier is mandatory for generic doc buckets like IDENTITY_PROOF/ADDRESS_PROOF.
            strict_classifier_for_fixed = parse_bool_env("STEP1_CLASSIFIER_FOR_FIXED_DOCS", False)
            should_run_classifier = doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"] or strict_classifier_for_fixed

            if clean_text.strip() and should_run_classifier:
                classifier_data = fast_classify_identity_document(clean_text) if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"] else None
                if classifier_data:
                    detected_type = classifier_data.get("detected_type", "Unknown")
                    conf = float(classifier_data.get("confidence", 0.0))
                    classification_ms = 0
                    print(
                        f"[CLASSIFIER LOCAL] req={request_id} detected_type={detected_type} "
                        f"conf={conf} reason={classifier_data.get('reason')}"
                    )
                    if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"]:
                        expected_type = detected_type
                        schema_key = detected_type
                    else:
                        expected_type = schema_key

                    if conf < 0.5:
                        return JSONResponse({
                            "success": False,
                            "status": "REUPLOAD_REQUIRED",
                            "error_type": "UNRECOGNIZED_DOCUMENT",
                            "title": "Document Not Recognized",
                            "message": "We could not identify this document. It may be too blurry.",
                            "action": "Please retake the photo ensuring it is clear and well-lit.",
                            "missing_fields": []
                        })

                    if detected_type != "Unknown" and detected_type != expected_type:
                        return JSONResponse({
                            "success": False,
                            "status": "REUPLOAD_REQUIRED",
                            "error_type": "WRONG_DOCUMENT",
                            "title": "Wrong Document Uploaded",
                            "message": f"You uploaded a {detected_type.replace('Card', ' Card').replace('Certificate', ' Certificate')} but we expected a {expected_type.replace('Form', ' Form').replace('Card', ' Card')}.",
                            "action": f"Please upload the correct {expected_type.replace('Form', ' Form').replace('Card', ' Card')} document.",
                            "missing_fields": []
                        })

                    # After successful classification and routing, update doc_type for downstream tracking
                    if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"]:
                        doc_type = {v: k for k, v in DOC_TYPE_MAPPING.items()}.get(schema_key, detected_type)
                else:
                    try:
                        class_prompt = CLASSIFIER_PROMPT_TEMPLATE.format(OCR_TEXT=clean_text[:4000])
                        class_model = llm_model if is_huggingface else "gpt-4o-mini"
                        kwargs = {
                            "model": class_model,
                            "messages": [{"role": "user", "content": class_prompt}],
                            "temperature": 0.0
                        }
                        if supports_response_format:
                            kwargs["response_format"] = {"type": "json_object"}

                        classify_start = time.time()
                        class_resp = call_llm_with_retry(client, kwargs, "classification")
                        raw_content = class_resp.choices[0].message.content
                        classification_ms = int((time.time() - classify_start) * 1000)
                        if is_huggingface:
                            raw_content = clean_json_response(raw_content)
                        classifier_data = json.loads(raw_content)
                        detected_type = classifier_data.get("detected_type", "Unknown")
                        conf = float(classifier_data.get("confidence", 0.0))
                        print(f"[CLASSIFIER API] req={request_id} detected_type={detected_type} conf={conf}")

                        if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"]:
                            expected_type = detected_type
                            schema_key = detected_type
                        else:
                            expected_type = schema_key

                        if conf < 0.5:
                            return JSONResponse({
                                "success": False,
                                "status": "REUPLOAD_REQUIRED",
                                "error_type": "UNRECOGNIZED_DOCUMENT",
                                "title": "Document Not Recognized",
                                "message": "We could not identify this document. It may be too blurry.",
                                "action": "Please retake the photo ensuring it is clear and well-lit.",
                                "missing_fields": []
                            })

                        if detected_type != "Unknown" and detected_type != expected_type:
                            return JSONResponse({
                                "success": False,
                                "status": "REUPLOAD_REQUIRED",
                                "error_type": "WRONG_DOCUMENT",
                                "title": "Wrong Document Uploaded",
                                "message": f"You uploaded a {detected_type.replace('Card', ' Card').replace('Certificate', ' Certificate')} but we expected a {expected_type.replace('Form', ' Form').replace('Card', ' Card')}.",
                                "action": f"Please upload the correct {expected_type.replace('Form', ' Form').replace('Card', ' Card')} document.",
                                "missing_fields": []
                            })

                        # After successful classification and routing, update doc_type for downstream tracking
                        if doc_type in ["IDENTITY_PROOF", "ADDRESS_PROOF"]:
                            doc_type = {v: k for k, v in DOC_TYPE_MAPPING.items()}.get(schema_key, detected_type)
                    except Exception as ce:
                        print("Classification Error:", ce)

            # --- STAGE 2: LLM EXTRACTION ---
            raw_schema = get_combined_schemas().get(schema_key)

            if raw_schema and clean_text:
                schema = prepare_schema_for_extraction(schema_key, raw_schema)
                schema_compact = json.dumps(schema, separators=(",", ":"))
                prompt = PROMPT_TEMPLATE.format(
                    DOCUMENT_TYPE=schema_key,
                    SCHEMA=schema_compact,
                    CLEAN_TEXT=clean_text,
                    CONF_TEXT=conf_text
                )

                kwargs = {
                    "model": llm_model,
                    "messages": [
                        {"role": "system", "content": "You are a precise JSON data extraction engine."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }
                if supports_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                openai_fallback_key = os.getenv("OPENAI_API_KEY")
                fallback_model = os.getenv("FALLBACK_OPENAI_MODEL", "gpt-4o")
                cloud_fallback_enabled = parse_bool_env("CLOUD_FALLBACK_TO_OPENAI", True)
                fallback_enabled = cloud_fallback_enabled and (
                    llm_provider == "huggingface"
                    or (llm_provider == "openai" and fallback_model != llm_model)
                )
                parallel_json_extraction = parse_bool_env("PARALLEL_JSON_EXTRACTION", False)
                parallel_json_enabled = parallel_json_extraction

                def normalize_extracted_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
                    for key in schema.keys():
                        if key not in payload:
                            if key in SCHEMA_METADATA_FIELDS:
                                payload[key] = clone_json_value(schema.get(key, SCHEMA_METADATA_DEFAULTS[key]))
                            else:
                                payload[key] = {
                                    "value": None,
                                    "ocr_confidence": None,
                                    "extraction_confidence": 0.0,
                                    "source_text": None,
                                }
                        elif key in SCHEMA_METADATA_FIELDS and is_extraction_field_object(payload.get(key)):
                            payload[key] = clone_json_value(schema.get(key, SCHEMA_METADATA_DEFAULTS[key]))

                    if "DeathCategory" in payload:
                        dc = payload["DeathCategory"]
                        if isinstance(dc, dict):
                            val = dc.get("value")
                            if val not in ["NATURAL_OR_MEDICAL", "UNNATURAL", None]:
                                dc["value"] = None
                                dc["extraction_confidence"] = 0.0

                    if "LowConfidenceFields" not in payload:
                        payload["LowConfidenceFields"] = []
                    if "UnmappedFields" not in payload:
                        payload["UnmappedFields"] = []
                    return apply_backend_schema_metadata(payload, schema_key, filename, request_id, pages_result)

                def try_cloud_fallback(reason: str):
                    nonlocal fallback_extraction_ms, llm_provider, llm_model, llm_fallback_used, llm_fallback_reason
                    if not openai_fallback_key:
                        llm_fallback_reason = "cloud_fallback_no_api_key"
                        return None, None

                    try:
                        fallback_client = OpenAI(
                            api_key=openai_fallback_key,
                            timeout=llm_request_timeout_s,
                            max_retries=0,
                        )
                        fallback_kwargs = {
                            "model": fallback_model,
                            "messages": [
                                {"role": "system", "content": "You are a precise JSON data extraction engine."},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.0,
                            "response_format": {"type": "json_object"},
                        }

                        fallback_start = time.time()
                        fallback_response = call_llm_with_retry(
                            fallback_client,
                            fallback_kwargs,
                            "fallback_extraction",
                            provider_override="openai",
                        )
                        fallback_extraction_ms = int((time.time() - fallback_start) * 1000)
                        fallback_text = fallback_response.choices[0].message.content
                        fallback_data = json.loads(fallback_text)

                        if not isinstance(fallback_data, dict):
                            llm_fallback_reason = "cloud_fallback_invalid_json"
                            return None, None

                        fallback_data = normalize_extracted_payload(fallback_data)
                        fallback_validation_result = validate_document(doc_type, fallback_data)
                        if fallback_validation_result.get("status") != "OK":
                            llm_fallback_reason = "cloud_fallback_validation_failed"
                            return None, fallback_validation_result

                        llm_provider = "openai"
                        llm_model = fallback_model
                        llm_fallback_used = True
                        llm_fallback_reason = reason
                        print(f"[LLM FALLBACK] provider=openai model={fallback_model} reason={reason}")
                        return fallback_data, None
                    except Exception as fallback_exc:
                        print(f"OpenAI fallback extraction failed: {fallback_exc}")
                        llm_fallback_reason = "cloud_fallback_error"
                        return None, None

                def parse_candidate_payload(raw_text: Optional[str], provider_name: str):
                    if raw_text is None:
                        return None, None, "empty_response"
                    try:
                        payload_text = clean_json_response(raw_text) if provider_name == "huggingface" else raw_text
                        candidate_data = json.loads(payload_text)
                        if not isinstance(candidate_data, dict):
                            return None, None, "invalid_json"
                        candidate_data = normalize_extracted_payload(candidate_data)
                        validation_result = validate_document(doc_type, candidate_data)
                        if validation_result.get("status") != "OK":
                            return None, validation_result, "validation_failed"
                        return candidate_data, None, None
                    except json.JSONDecodeError:
                        return None, None, "invalid_json"
                    except Exception:
                        return None, None, "parse_error"

                def run_parallel_hedged_extraction():
                    nonlocal extraction_ms, fallback_extraction_ms, llm_provider, llm_model, llm_fallback_used, llm_fallback_reason, parallel_json_used
                    if not (parallel_json_extraction and fallback_enabled and openai_fallback_key):
                        return None, None, None

                    parallel_json_used = True
                    print("[LLM PARALLEL] providers=huggingface+openai mode=hedged")
                    fallback_client = OpenAI(
                        api_key=openai_fallback_key,
                        timeout=llm_request_timeout_s,
                        max_retries=0,
                    )
                    fallback_kwargs = {
                        "model": fallback_model,
                        "messages": [
                            {"role": "system", "content": "You are a precise JSON data extraction engine."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    }

                    executor = ThreadPoolExecutor(max_workers=2)
                    start_hf = time.time()
                    start_openai = time.time()
                    futures = {
                        executor.submit(call_llm_with_retry, client, kwargs, "extraction_parallel", "huggingface"): ("huggingface", llm_model, start_hf),
                        executor.submit(call_llm_with_retry, fallback_client, fallback_kwargs, "extraction_parallel", "openai"): ("openai", fallback_model, start_openai),
                    }
                    first_validation_failure = None
                    saw_invalid_json = False
                    saw_error = False

                    try:
                        for future in concurrent.futures.as_completed(futures):
                            provider_name, model_name, started_at = futures[future]
                            try:
                                response = future.result()
                                duration_ms = int((time.time() - started_at) * 1000)
                                response_text = response.choices[0].message.content
                            except Exception as candidate_exc:
                                saw_error = True
                                print(f"[LLM HEDGE ERROR] provider={provider_name} error={candidate_exc}")
                                continue

                            if provider_name == "huggingface":
                                extraction_ms = duration_ms
                            else:
                                fallback_extraction_ms = duration_ms

                            candidate_data, validation_failure, parse_error = parse_candidate_payload(response_text, provider_name)
                            if candidate_data is not None:
                                if provider_name == "openai":
                                    llm_provider = "openai"
                                    llm_model = model_name
                                    llm_fallback_used = True
                                    llm_fallback_reason = "parallel_hedged_winner"
                                    print(f"[LLM FALLBACK] provider=openai model={model_name} reason=parallel_hedged_winner")
                                return candidate_data, None, None

                            if validation_failure is not None and first_validation_failure is None:
                                first_validation_failure = validation_failure
                            if parse_error == "invalid_json":
                                saw_invalid_json = True
                    finally:
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)

                    if first_validation_failure:
                        return None, first_validation_failure, "validation_failed"
                    if saw_invalid_json and not saw_error:
                        return None, None, "invalid_json"
                    return None, None, "timeout_or_error"

                response_text = None
                fallback_validation_result = None
                parallel_failure_reason = None
                hedged_parallel_active = parallel_json_extraction and fallback_enabled and bool(openai_fallback_key)

                if hedged_parallel_active:
                    extracted_data, fallback_validation_result, parallel_failure_reason = run_parallel_hedged_extraction()
                    if extracted_data is None:
                        if fallback_validation_result:
                            return JSONResponse({
                                "success": False,
                                **fallback_validation_result
                            })
                        if parallel_failure_reason == "invalid_json":
                            return JSONResponse({
                                "success": False,
                                "status": "PROCESSING_FAILED",
                                "error_type": "INVALID_LLM_JSON",
                                "title": "Extraction Format Error",
                                "message": "Parallel extraction returned invalid JSON from all providers.",
                                "action": "Please retry the upload. If this persists, reduce input size.",
                                "missing_fields": []
                            })
                        return JSONResponse({
                            "success": False,
                            "status": "PROCESSING_FAILED",
                            "error_type": "LLM_TIMEOUT",
                            "title": "Extraction Timed Out",
                            "message": "Both primary and fallback providers timed out or failed in parallel extraction.",
                            "action": "Please retry. If this persists, reduce input size.",
                            "missing_fields": []
                        })

                if extracted_data is None:
                    extract_start = time.time()
                    try:
                        response = call_llm_with_retry(client, kwargs, "extraction")
                        response_text = response.choices[0].message.content
                        extraction_ms = int((time.time() - extract_start) * 1000)
                    except Exception:
                        extraction_ms = int((time.time() - extract_start) * 1000)
                        if fallback_enabled:
                            extracted_data, fallback_validation_result = try_cloud_fallback("primary_extraction_error")
                        if extracted_data is None:
                            if fallback_validation_result:
                                return JSONResponse({
                                    "success": False,
                                    **fallback_validation_result
                                })
                            primary_label = (llm_provider or "primary").title()
                            return JSONResponse({
                                "success": False,
                                "status": "PROCESSING_FAILED",
                                "error_type": "LLM_TIMEOUT",
                                "title": "Extraction Timed Out",
                                "message": f"The {primary_label} model did not finish extraction in time.",
                                "action": "Please retry. If this persists, use OpenAI fallback or reduce input size.",
                                "missing_fields": []
                            })

                if extracted_data is None and response_text is not None:
                    provider_name = "huggingface" if is_huggingface else "openai"
                    candidate_data, validation_result, parse_error = parse_candidate_payload(response_text, provider_name)
                    if candidate_data is not None:
                        extracted_data = candidate_data
                    elif validation_result is not None:
                        if fallback_enabled:
                            fallback_data, _ = try_cloud_fallback("primary_validation_failed")
                            if fallback_data is not None:
                                extracted_data = fallback_data
                        if extracted_data is None:
                            return JSONResponse({
                                "success": False,
                                **validation_result
                            })
                    elif parse_error == "invalid_json":
                        if fallback_enabled:
                            extracted_data, fallback_validation_result = try_cloud_fallback("primary_json_parse_failed")
                        if extracted_data is None:
                            if fallback_validation_result:
                                return JSONResponse({
                                    "success": False,
                                    **fallback_validation_result
                                })
                            return JSONResponse({
                                "success": False,
                                "status": "PROCESSING_FAILED",
                                "error_type": "INVALID_LLM_JSON",
                                "title": "Extraction Format Error",
                                "message": "The model returned invalid JSON for this document.",
                                "action": "Please retry the upload. If this persists, switch to cloud extraction.",
                                "missing_fields": []
                            })
    except Exception as e:
        print(f"LLM Extraction Error: {e}")
    finally:
        llm_stage_ms = int((time.time() - llm_stage_started_at) * 1000)
        print(
            f"[LLM STAGE DONE] req={request_id} doc={doc_type} provider={llm_provider or 'none'} "
            f"model={llm_model or 'none'} llm_ms={llm_stage_ms} total_ms={int((time.time() - request_started_at) * 1000)}"
        )
        
    # --- CPU & RAM USAGE MONITORING FOR UNNATURAL DOCUMENTS ---
    unnatural_docs = [
        "FIR", "INQUEST_REPORT", "FINAL_POLICE_REPORT", 
        "POSTMORTEM_REPORT", "VISCERA_REPORT", "NEWSPAPER_CUTTING", 
        "DRIVING_LICENCE_STEP2"
    ]
    if doc_type in unnatural_docs:
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_info = psutil.virtual_memory()
            ram_usage_mb = (ram_info.total - ram_info.available) / (1024 ** 2)
            print(f"\n{'='*50}\n[SYSTEM MONITOR] After Parsing {doc_type}\nCPU Usage: {cpu_usage}%\nRAM In-Use: {ram_usage_mb:.0f} MB ({ram_info.percent}%)\n{'='*50}\n")
        except ImportError:
            print("[SYSTEM MONITOR] psutil not installed. Cannot log RAM/CPU usage.")
            
    # --- BLUR / LENGTH CHECK FOR STEP 2 (ALL DOCS) ---
    step_2_docs = [
        "MEDICO_LEGAL_CERT", 
        "HOSPITALIZATION_RECORDS", 
        "TREATING_DOCTOR_CERT", 
        "HOSPITAL_ATTENDANT_CERT", 
        "EMPLOYER_CERT",
        "FIR",
        "INQUEST_REPORT",
        "FINAL_POLICE_REPORT",
        "POSTMORTEM_REPORT",
        "VISCERA_REPORT",
        "NEWSPAPER_CUTTING",
        "DRIVING_LICENCE_STEP2"
    ]
    if doc_type in step_2_docs:
        total_extracted_text = " ".join([p.get("raw_text", "") for p in pages_result])
        if len(total_extracted_text.strip()) < 50:
            return JSONResponse({
                "success": False,
                "status": "REUPLOAD_REQUIRED",
                "error_type": "POOR_SCAN",
                "title": "Document Too Blurry or Blank",
                "message": "We could not extract enough text from this document. It appears to be extremely blurry, poorly lit, or blank.",
                "action": "Please take a clearer photo and re-upload.",
                "missing_fields": []
            })
    
    return JSONResponse({
        "success": True,
        "filename": filename,
        "doc_type": doc_type,
        "engine_used": engine_used,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_fallback_used": llm_fallback_used,
        "llm_fallback_reason": llm_fallback_reason,
        "llm_metrics": {
            "classification_ms": classification_ms,
            "extraction_ms": extraction_ms,
            "fallback_extraction_ms": fallback_extraction_ms,
            "input_clean_chars": clean_text_chars,
            "input_conf_chars": conf_text_chars,
            "input_truncated": llm_input_truncated,
            "request_timeout_s": llm_request_timeout_s,
            "retry_count": llm_retry_count,
            "parallel_json_enabled": parallel_json_enabled,
            "parallel_json_used": parallel_json_used,
        },
        "pipeline_metrics": {
            "ocr_stage_ms": ocr_stage_ms,
            "llm_stage_ms": llm_stage_ms,
            "rapid_ocr_max_concurrency": RAPID_OCR_MAX_CONCURRENCY,
            "pdf_parallel_workers": pdf_parallel_workers,
            "pdf_parallel_threshold_pages": pdf_parallel_threshold_pages,
            "total_request_ms": int((time.time() - request_started_at) * 1000),
        },
        "qr_codes": all_qr_codes,
        "extracted_data": extracted_data,
        "global_metrics": {
            "overall_confidence": global_overall,
            "text_confidence": global_text,
            "number_confidence": global_num,
            "total_pages": len(pages_result),
            "total_words": total_words,
            "total_texts": total_texts,
            "total_numbers": total_nums,
            "total_processing_time_ms": total_processing_time
        },
        "pages": pages_result
    })

@router.post("/extract-json")
async def extract_claims_json(request: Request):
    payload = await request.json()
    doc_type = canonicalize_doc_type(payload.get("doc_type", "DEATH_CERTIFICATE"))
    pages_result = payload.get("pages") or []
    filename = payload.get("filename") or "uploaded_file"
    request_id = payload.get("request_id") or uuid.uuid4().hex[:8]

    if not isinstance(pages_result, list) or not pages_result:
        return JSONResponse({
            "success": False,
            "status": "PROCESSING_FAILED",
            "error_type": "OCR_TEXT_MISSING",
            "title": "OCR Text Missing",
            "message": "No OCR page data was provided for JSON extraction.",
            "action": "Please retry the document upload.",
            "missing_fields": [],
        })

    print(f"[JSON START] req={request_id} doc={doc_type} file={filename}")
    extraction_result = run_json_extraction_from_pages(
        doc_type=doc_type,
        pages_result=pages_result,
        filename=filename,
        request_id=request_id,
    )
    return JSONResponse(extraction_result)

@router.post("/pipeline/start")
async def start_claims_pipeline(
    files: List[UploadFile] = File(...),
    doc_types: List[str] = Form(...),
):
    ensure_pipeline_workers_started()
    raw_doc_types = list(doc_types or [])
    doc_types = normalize_doc_type_list(doc_types or [])
    print(f"[PIPELINE INPUT] raw_doc_types={raw_doc_types} normalized_doc_types={doc_types}")
    if len(files) != len(doc_types):
        raise HTTPException(
            status_code=400,
            detail=(
                f"files and doc_types must have the same length. "
                f"Received files={len(files)} doc_types={len(doc_types)}. "
                "If Swagger sent doc_types as CSV, use separate items or keep comma values; server now splits them."
            )
        )

    batch_id = uuid.uuid4().hex
    created_at = time.time()
    docs: Dict[str, Any] = {}
    queued_jobs = []
    face_documents = []
    incoming_doc_types = list(doc_types)
    run_face_verification = should_run_stage1_face_verification(incoming_doc_types)

    for idx, upload in enumerate(files):
        doc_type = doc_types[idx]
        filename = upload.filename or f"uploaded_file_{idx + 1}"
        file_bytes = await upload.read()
        is_face_only_doc = doc_type in FACE_ONLY_DOC_TYPES
        doc_id = uuid.uuid4().hex
        request_id = doc_id[:8]
        docs[doc_id] = {
            "doc_id": doc_id,
            "request_id": request_id,
            "doc_type": doc_type,
            "filename": filename,
            "status": "success" if is_face_only_doc else "queued_ocr",
            "stage": "done" if is_face_only_doc else "ocr",
            "result": build_face_only_document_result(doc_type, filename) if is_face_only_doc else None,
            "errorMsg": None,
            "created_at": created_at,
            "updated_at": created_at,
            "completed_at": created_at if is_face_only_doc else None,
        }
        if not is_face_only_doc:
            queued_jobs.append({
                "batch_id": batch_id,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "filename": filename,
                "file_bytes": file_bytes,
                "request_id": request_id,
            })
        if run_face_verification:
            face_documents.append({
                "doc_id": doc_id,
                "request_id": request_id,
                "doc_type": doc_type,
                "filename": filename,
                "file_bytes": file_bytes,
            })

    face_verification_status = {
        "enabled": run_face_verification,
        "status": "queued" if run_face_verification else "skipped",
        "decision": "MANUAL_REVIEW" if run_face_verification else "NOT_APPLICABLE",
        "review_required": bool(run_face_verification),
        "review_flags": [],
        "created_at": created_at,
        "updated_at": created_at,
    }
    with PIPELINE_LOCK:
        PIPELINE_BATCHES[batch_id] = {
            "batch_id": batch_id,
            "status": "processing",
            "created_at": created_at,
            "updated_at": created_at,
            "workers": {
                "ocr_workers": max(1, min(parse_int_env("OCR_WORKERS", 4), 8)),
                "llm_workers": max(1, min(parse_int_env("LLM_WORKERS", 4), 8)),
                "validation_workers": max(1, min(parse_int_env("VALIDATION_WORKERS", 2), 8)),
                "face_workers": max(1, min(parse_int_env("FACE_VERIFY_WORKERS", 1), 4)),
                "llm_max_concurrency": LLM_MAX_CONCURRENCY,
            },
            "counts": {},
            "docs": docs,
            "face_verification": face_verification_status,
        }
        recompute_pipeline_batch_locked(PIPELINE_BATCHES[batch_id])

    for job in queued_jobs:
        OCR_QUEUE.put(job)
    if run_face_verification:
        FACE_QUEUE.put({
            "batch_id": batch_id,
            "request_id": batch_id[:8],
            "documents": face_documents,
        })

    print(
        f"[PIPELINE START] batch={batch_id} docs={len(docs)} "
        f"ocr_jobs={len(queued_jobs)} face={run_face_verification}"
    )
    batch = get_pipeline_batch_public(batch_id)
    return JSONResponse(batch)

@router.get("/pipeline/{batch_id}")
async def get_claims_pipeline_status(batch_id: str):
    batch = get_pipeline_batch_public(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Pipeline batch not found.")
    return JSONResponse(batch)

# Add placeholder endpoints for external claimant router
@router.post("/")
async def create_claim(
    policy_id: str = Form(...),
    claimant_name: str = Form(...),
    date_of_death: str = Form(...),
    cause_of_death: str = Form(...),
    contact_email: str = Form(...),
    contact_phone: str = Form(...)
):
    import uuid
    from datetime import datetime, timedelta
    claim_id = str(uuid.uuid4())
    return {
        "claim_id": claim_id,
        "status": "RECEIVED",
        "acknowledgement_sla": (datetime.now() + timedelta(hours=24)).isoformat()
    }

@router.get("/{claim_id}")
async def get_claim_status(claim_id: str):
    from datetime import datetime, timedelta
    return {
        "claim_id": claim_id,
        "current_stage": "DOC_INTEL",
        "submitted_at": datetime.now().isoformat(),
        "sla_deadline": (datetime.now() + timedelta(days=30)).isoformat(),
        "missing_documents": [],
        "last_updated": datetime.now().isoformat()
    }
