import base64
import io
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


CLAIMANT_RECENT_PHOTOGRAPH_DOC_TYPE = "CLAIMANT_RECENT_PHOTOGRAPH"

REFERENCE_DOC_TYPES = {CLAIMANT_RECENT_PHOTOGRAPH_DOC_TYPE}
PHOTO_ID_DOC_TYPES = {
    "CLAIMANT_STATEMENT_FORM",
    "IDENTITY_PROOF",
    "ADDRESS_PROOF",
    "AADHAAR_CARD",
    "PASSPORT",
    "DRIVING_LICENCE",
    "VOTER_ID",
    "PAN_CARD",
}
STAGE1_DOC_TYPES = {
    CLAIMANT_RECENT_PHOTOGRAPH_DOC_TYPE,
    "CLAIMANT_STATEMENT_FORM",
    "DEATH_CERTIFICATE",
    "IDENTITY_PROOF",
    "ADDRESS_PROOF",
    "AADHAAR_CARD",
    "PASSPORT",
    "DRIVING_LICENCE",
    "VOTER_ID",
    "PAN_CARD",
    "BANK_PROOF",
}
FACE_TERMINAL_STATUSES = {"completed", "skipped", "dependency_missing", "error", "disabled"}


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


def is_stage1_batch(doc_types: List[str]) -> bool:
    return bool(doc_types) and all(doc_type in STAGE1_DOC_TYPES for doc_type in doc_types)


def should_run_stage1_face_verification(doc_types: List[str]) -> bool:
    if not parse_bool_env("FACE_VERIFY_STAGE1_ENABLED", True):
        return False
    if not is_stage1_batch(doc_types):
        return False
    return any(doc_type in REFERENCE_DOC_TYPES or doc_type in PHOTO_ID_DOC_TYPES for doc_type in doc_types)


def is_face_status_terminal(status: Optional[str]) -> bool:
    return (status or "").lower() in FACE_TERMINAL_STATUSES


def doc_label(doc_type: str) -> str:
    return (doc_type or "").replace("_", " ").title()


def canonical_photo_doc_type(doc_type: str, extracted_doc_type: Optional[str] = None) -> str:
    value = extracted_doc_type or doc_type
    return value or doc_type


def doc_weight(doc_type: str, extracted_doc_type: Optional[str] = None) -> float:
    canonical = canonical_photo_doc_type(doc_type, extracted_doc_type)
    weights = {
        "PASSPORT": 1.0,
        "Passport": 1.0,
        "CLAIMANT_STATEMENT_FORM": 1.0,
        "AADHAAR_CARD": 1.0,
        "AadhaarCard": 1.0,
        "IDENTITY_PROOF": 0.9,
        "ADDRESS_PROOF": 0.85,
        "DRIVING_LICENCE": 0.8,
        "DrivingLicence": 0.8,
        "VOTER_ID": 0.75,
        "VoterID": 0.75,
        "PAN_CARD": 0.4,
        "PANCard": 0.4,
    }
    return weights.get(canonical, weights.get(doc_type, 0.7))


def safe_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    return re.sub(r"\s+", " ", message)[:500]


def classify_deepface_import_error(error: Optional[str]) -> str:
    message = (error or "").lower()
    if "tf-keras" in message or "tf_keras" in message:
        return "TF_KERAS_NOT_INSTALLED"
    if "tensorflow" in message:
        return "TENSORFLOW_RUNTIME_ERROR"
    return "DEEPFACE_NOT_INSTALLED"


def load_deepface():
    try:
        from deepface import DeepFace  # type: ignore
        return DeepFace, None
    except Exception as exc:
        return None, safe_error(exc)


def build_face_verification_config(
    anti_spoofing: Optional[bool] = None,
    return_face_previews: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "model_name": os.getenv("FACE_VERIFY_MODEL", "Facenet512"),
        "detector_backend": os.getenv("FACE_VERIFY_DETECTOR_BACKEND", "retinaface"),
        "distance_metric": os.getenv("FACE_VERIFY_DISTANCE_METRIC", "cosine"),
        "anti_spoofing": (
            parse_bool_env("FACE_VERIFY_ANTI_SPOOFING", True)
            if anti_spoofing is None else bool(anti_spoofing)
        ),
        "match_threshold": parse_float_env("FACE_VERIFY_MATCH_THRESHOLD", 80.0),
        "no_match_threshold": parse_float_env("FACE_VERIFY_NO_MATCH_THRESHOLD", 25.0),
        "review_low": parse_float_env("FACE_VERIFY_REVIEW_BAND_LOW", 35.0),
        "review_high": parse_float_env("FACE_VERIFY_REVIEW_BAND_HIGH", 65.0),
        "face_crop_padding_px": max(0, parse_int_env("FACE_VERIFY_FACE_CROP_PADDING_PX", 24)),
        "face_crop_padding_ratio": max(0.0, parse_float_env("FACE_VERIFY_FACE_CROP_PADDING_RATIO", 0.18)),
        "return_face_previews": (
            parse_bool_env("FACE_VERIFY_RETURN_FACE_PREVIEWS", True)
            if return_face_previews is None else bool(return_face_previews)
        ),
    }


def decide_face_confidence(confidence: Optional[float], config: Dict[str, Any]) -> str:
    if confidence is None:
        return "MANUAL_REVIEW"
    if config["review_low"] <= confidence <= config["review_high"]:
        return "MANUAL_REVIEW"
    if confidence >= config["match_threshold"]:
        return "MATCH"
    if confidence <= config["no_match_threshold"]:
        return "NO_MATCH"
    return "MANUAL_REVIEW"


def pil_to_bgr_array(image: Image.Image) -> np.ndarray:
    rgb = ImageOps.exif_transpose(image.convert("RGB"))
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def prepare_image_for_face_detection(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image.convert("RGB"))


def render_pdf_pages(file_bytes: bytes, max_pages: int, dpi: int) -> List[Image.Image]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF is required for PDF face verification: {safe_error(exc)}") from exc

    pages: List[Image.Image] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_idx in range(min(len(doc), max_pages)):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return pages


def upload_to_images(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    max_pdf_pages = max(1, min(parse_int_env("FACE_VERIFY_MAX_PDF_PAGES", 3), 10))
    render_dpi = max(100, min(parse_int_env("FACE_VERIFY_RENDER_DPI", 180), 300))
    lower_name = (filename or "").lower()
    if lower_name.endswith(".pdf"):
        images = render_pdf_pages(file_bytes, max_pdf_pages, render_dpi)
    else:
        image = Image.open(io.BytesIO(file_bytes))
        images = [image]

    frames = []
    for idx, image in enumerate(images):
        prepared = prepare_image_for_face_detection(image)
        frames.append({
            "frame_index": idx,
            "page_number": idx + 1,
            "image": prepared,
            "width": prepared.width,
            "height": prepared.height,
        })
    return frames


def call_with_optional_antispoof(func, anti_spoofing: bool, **kwargs):
    if anti_spoofing:
        try:
            return func(**kwargs, anti_spoofing=True), True
        except TypeError as exc:
            if "anti_spoofing" not in str(exc):
                raise
        except Exception:
            raise
    return func(**kwargs), False


def face_area_score(face_obj: Dict[str, Any], width: int, height: int) -> float:
    area = face_obj.get("facial_area") or {}
    try:
        face_w = float(area.get("w") or area.get("width") or 0)
        face_h = float(area.get("h") or area.get("height") or 0)
        detector_conf = float(face_obj.get("confidence") or face_obj.get("face_confidence") or 0.0)
    except Exception:
        return 0.0
    image_area = max(float(width * height), 1.0)
    return (face_w * face_h / image_area) * max(detector_conf, 0.01)


def preview_face(face_obj: Dict[str, Any], return_preview: Optional[bool] = None) -> Optional[str]:
    if return_preview is None:
        return_preview = parse_bool_env("FACE_VERIFY_RETURN_FACE_PREVIEWS", True)
    if not return_preview:
        return None
    face = face_obj.get("face")
    if face is None:
        return None
    try:
        arr = np.asarray(face)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


def image_to_data_url(image: Image.Image) -> Optional[str]:
    try:
        buffer = io.BytesIO()
        ImageOps.exif_transpose(image.convert("RGB")).save(buffer, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


def expanded_face_crop(
    image: Image.Image,
    area: Dict[str, Any],
    padding_px: int,
    padding_ratio: float,
) -> Tuple[Optional[Image.Image], Optional[Dict[str, int]]]:
    try:
        source = ImageOps.exif_transpose(image.convert("RGB"))
        x = int(float(area.get("x") or 0))
        y = int(float(area.get("y") or 0))
        w = int(float(area.get("w") or area.get("width") or 0))
        h = int(float(area.get("h") or area.get("height") or 0))
        if w <= 0 or h <= 0:
            return None, None
        pad = int(max(float(padding_px), max(w, h) * float(padding_ratio)))
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(source.width, x + w + pad)
        bottom = min(source.height, y + h + pad)
        if right <= left or bottom <= top:
            return None, None
        bounds = {
            "x": left,
            "y": top,
            "w": right - left,
            "h": bottom - top,
            "padding_px": pad,
        }
        return source.crop((left, top, right, bottom)), bounds
    except Exception:
        return None, None


def face_obj_to_bgr_array(face_obj: Dict[str, Any]) -> Optional[np.ndarray]:
    face = face_obj.get("face")
    if face is None:
        return None
    try:
        arr = np.asarray(face)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def extract_best_face_for_document(DeepFace, document: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.time()
    doc_id = document.get("doc_id")
    doc_type = document.get("doc_type")
    filename = document.get("filename")
    base_result: Dict[str, Any] = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "filename": filename,
        "label": doc_label(doc_type),
        "face_found": False,
        "status": "NO_FACE_FOUND",
        "review_required": True,
        "review_flags": ["NO_FACE_FOUND"],
        "face_count": 0,
        "page_number": None,
        "quality_score": 0.0,
        "detector_backend": config["detector_backend"],
        "anti_spoofing_checked": False,
        "anti_spoofing_passed": None,
        "anti_spoofing_retry_used": False,
        "processing_ms": 0,
    }

    try:
        frames = upload_to_images(document["file_bytes"], filename or "")
    except Exception as exc:
        base_result.update({
            "status": "IMAGE_LOAD_ERROR",
            "error": safe_error(exc),
            "review_flags": ["IMAGE_LOAD_ERROR"],
            "processing_ms": int((time.time() - started_at) * 1000),
        })
        return base_result

    best: Optional[Tuple[float, Dict[str, Any], Dict[str, Any], bool, bool]] = None
    total_faces = 0
    frame_errors = []
    for frame in frames:
        img_array = pil_to_bgr_array(frame["image"])
        anti_retry_used = False
        try:
            face_objs, anti_applied = call_with_optional_antispoof(
                DeepFace.extract_faces,
                config["anti_spoofing"],
                img_path=img_array,
                detector_backend=config["detector_backend"],
                align=True,
                enforce_detection=True,
            )
        except Exception as exc:
            if not config["anti_spoofing"]:
                frame_errors.append({
                    "page_number": frame["page_number"],
                    "error": safe_error(exc),
                })
                continue
            try:
                face_objs, anti_applied = call_with_optional_antispoof(
                    DeepFace.extract_faces,
                    False,
                    img_path=img_array,
                    detector_backend=config["detector_backend"],
                    align=True,
                    enforce_detection=True,
                )
                anti_retry_used = True
            except Exception as retry_exc:
                frame_errors.append({
                    "page_number": frame["page_number"],
                    "error": safe_error(retry_exc),
                    "anti_spoofing_error": safe_error(exc),
                })
                continue

        if not isinstance(face_objs, list):
            face_objs = [face_objs]
        total_faces += len(face_objs)
        for face_obj in face_objs:
            score = face_area_score(face_obj, frame["width"], frame["height"])
            if best is None or score > best[0]:
                best = (score, face_obj, frame, anti_applied, anti_retry_used)

    base_result["face_count"] = total_faces
    if best is None:
        if frame_errors:
            base_result["detector_errors"] = frame_errors[:3]
        base_result["processing_ms"] = int((time.time() - started_at) * 1000)
        return base_result

    score, face_obj, frame, anti_applied, anti_retry_used = best
    review_flags = []
    anti_passed = None
    if anti_applied:
        anti_passed = bool(face_obj.get("is_real", True))
        if not anti_passed:
            review_flags.append("ANTI_SPOOFING_REVIEW")
    elif anti_retry_used:
        review_flags.append("ANTI_SPOOFING_REVIEW")
    elif config["anti_spoofing"]:
        review_flags.append("ANTI_SPOOFING_UNAVAILABLE")
    if total_faces > 1:
        review_flags.append("MULTIPLE_FACES_DETECTED")

    area = face_obj.get("facial_area") or {}
    preview_crop, preview_bounds = expanded_face_crop(
        frame["image"],
        area,
        config["face_crop_padding_px"],
        config["face_crop_padding_ratio"],
    )
    face_preview = None
    if config.get("return_face_previews"):
        face_preview = image_to_data_url(preview_crop) if preview_crop is not None else None
        if face_preview is None:
            face_preview = preview_face(face_obj, True)
    base_result.update({
        "face_found": True,
        "status": "FACE_FOUND",
        "review_required": bool(review_flags),
        "review_flags": review_flags,
        "page_number": frame["page_number"],
        "quality_score": round(float(score), 6),
        "facial_area": {
            "x": area.get("x"),
            "y": area.get("y"),
            "w": area.get("w") or area.get("width"),
            "h": area.get("h") or area.get("height"),
        },
        "face_preview_bounds": preview_bounds,
        "anti_spoofing_checked": anti_applied,
        "anti_spoofing_passed": anti_passed,
        "anti_spoofing_retry_used": anti_retry_used,
        "face_preview": face_preview,
        "processing_ms": int((time.time() - started_at) * 1000),
        "_face_array": face_obj_to_bgr_array(face_obj),
        "_image_array": pil_to_bgr_array(frame["image"]),
    })
    return base_result


def strip_private_image(result: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(result)
    clean.pop("_face_array", None)
    clean.pop("_image_array", None)
    return clean


def compare_face_pair(DeepFace, reference: Dict[str, Any], candidate: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.time()
    compare_input = "face_crop"
    result: Dict[str, Any] = {
        "reference_doc_id": reference.get("doc_id"),
        "candidate_doc_id": candidate.get("doc_id"),
        "candidate_doc_type": candidate.get("doc_type"),
        "candidate_filename": candidate.get("filename"),
        "confidence": None,
        "verified": None,
        "decision": "MANUAL_REVIEW",
        "weight": doc_weight(candidate.get("doc_type", "")),
        "review_required": True,
        "review_flags": [],
        "processing_flags": [],
        "compare_input": compare_input,
        "processing_ms": 0,
    }

    if not reference.get("face_found"):
        result["review_flags"].append("REFERENCE_FACE_MISSING")
        return result
    if not candidate.get("face_found"):
        result["review_flags"].append("CANDIDATE_FACE_MISSING")
        return result

    verify_kwargs = {
        "img1_path": reference.get("_face_array"),
        "img2_path": candidate.get("_face_array"),
        "model_name": config["model_name"],
        "detector_backend": "skip",
        "distance_metric": config["distance_metric"],
        "align": False,
        "enforce_detection": False,
    }
    if verify_kwargs["img1_path"] is None or verify_kwargs["img2_path"] is None:
        compare_input = "full_document"
        verify_kwargs.update({
            "img1_path": reference["_image_array"],
            "img2_path": candidate["_image_array"],
            "detector_backend": config["detector_backend"],
            "align": True,
            "enforce_detection": True,
        })

    try:
        verify_result, anti_applied = call_with_optional_antispoof(
            DeepFace.verify,
            config["anti_spoofing"] if compare_input == "full_document" else False,
            **verify_kwargs,
        )
    except Exception as exc:
        if compare_input == "face_crop":
            try:
                compare_input = "full_document"
                verify_result, anti_applied = call_with_optional_antispoof(
                    DeepFace.verify,
                    config["anti_spoofing"],
                    img1_path=reference["_image_array"],
                    img2_path=candidate["_image_array"],
                    model_name=config["model_name"],
                    detector_backend=config["detector_backend"],
                    distance_metric=config["distance_metric"],
                    align=True,
                    enforce_detection=True,
                )
                result["processing_flags"].append("FACE_CROP_COMPARE_FALLBACK")
            except Exception as fallback_exc:
                result.update({
                    "compare_input": compare_input,
                    "review_flags": ["VERIFY_ERROR"],
                    "error": safe_error(fallback_exc),
                    "crop_compare_error": safe_error(exc),
                    "processing_ms": int((time.time() - started_at) * 1000),
                })
                return result
        else:
            result.update({
                "compare_input": compare_input,
                "review_flags": ["VERIFY_ERROR"],
                "error": safe_error(exc),
                "processing_ms": int((time.time() - started_at) * 1000),
            })
            return result

    confidence = verify_result.get("confidence")
    if confidence is None:
        result["review_flags"].append("MODEL_CONFIDENCE_UNAVAILABLE")
    else:
        try:
            confidence = round(float(confidence), 2)
        except Exception:
            confidence = None

    result.update({
        "confidence": confidence,
        "verified": verify_result.get("verified"),
        "distance": verify_result.get("distance"),
        "threshold": verify_result.get("threshold"),
        "model": verify_result.get("model") or config["model_name"],
        "detector_backend": verify_result.get("detector_backend") or config["detector_backend"],
        "compare_input": compare_input,
        "anti_spoofing_checked": anti_applied,
        "processing_ms": int((time.time() - started_at) * 1000),
    })

    if confidence is not None:
        if config["review_low"] <= confidence <= config["review_high"]:
            result["review_flags"].append("AMBIGUOUS_DOC_SCORE")
        result["decision"] = decide_face_confidence(confidence, config)
        result["review_required"] = result["decision"] == "MANUAL_REVIEW" or bool(result["review_flags"])
    return result


def aggregate_comparisons(comparisons: List[Dict[str, Any]], review_flags: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    scored = [c for c in comparisons if c.get("confidence") is not None]
    if not comparisons:
        return {
            "decision": "MANUAL_REVIEW",
            "overall_confidence": None,
            "review_required": True,
            "review_flags": sorted(set(review_flags + ["NO_COMPARABLE_ID_FACE"])),
        }
    if not scored:
        return {
            "decision": "MANUAL_REVIEW",
            "overall_confidence": None,
            "review_required": True,
            "review_flags": sorted(set(review_flags + ["NO_CONFIDENCE_AVAILABLE"])),
        }

    total_weight = sum(float(c.get("weight") or 0.0) for c in scored)
    if total_weight <= 0:
        total_weight = float(len(scored))
        for comparison in scored:
            comparison["weight"] = 1.0
    overall = sum(float(c["confidence"]) * float(c.get("weight") or 1.0) for c in scored) / total_weight
    overall = round(overall, 2)
    any_ambiguous = any("AMBIGUOUS_DOC_SCORE" in c.get("review_flags", []) for c in comparisons)

    decision = decide_face_confidence(overall, config)
    if any_ambiguous:
        decision = "MANUAL_REVIEW"
        review_flags.append("AMBIGUOUS_DOC_SCORE")

    return {
        "decision": decision,
        "overall_confidence": overall,
        "review_required": decision == "MANUAL_REVIEW" or bool(review_flags),
        "review_flags": sorted(set(review_flags)),
    }


def run_stage1_face_verification(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    started_at = time.time()
    config = build_face_verification_config(return_face_previews=True)
    result: Dict[str, Any] = {
        "enabled": parse_bool_env("FACE_VERIFY_STAGE1_ENABLED", True),
        "status": "running",
        "decision": "MANUAL_REVIEW",
        "score": None,
        "overall_confidence": None,
        "model": config["model_name"],
        "detector_backend": config["detector_backend"],
        "distance_metric": config["distance_metric"],
        "anti_spoofing_requested": config["anti_spoofing"],
        "config": config,
        "reference_doc": None,
        "document_results": [],
        "comparisons": [],
        "review_required": True,
        "review_flags": [],
        "processing_ms": 0,
    }

    if not result["enabled"]:
        result.update({
            "status": "disabled",
            "decision": "NOT_APPLICABLE",
            "review_required": False,
            "processing_ms": int((time.time() - started_at) * 1000),
        })
        return result

    relevant_docs = [
        doc for doc in documents
        if doc.get("doc_type") in REFERENCE_DOC_TYPES or doc.get("doc_type") in PHOTO_ID_DOC_TYPES
    ]
    if not relevant_docs:
        result.update({
            "status": "skipped",
            "decision": "NOT_APPLICABLE",
            "review_required": False,
            "review_flags": [],
            "processing_ms": int((time.time() - started_at) * 1000),
        })
        return result

    DeepFace, import_error = load_deepface()
    if DeepFace is None:
        dependency_flag = classify_deepface_import_error(import_error)
        print(f"[FACE VERIFY IMPORT ERROR] {import_error}", flush=True)
        result.update({
            "status": "dependency_missing",
            "decision": "MANUAL_REVIEW",
            "review_required": True,
            "review_flags": [dependency_flag],
            "error": import_error,
            "processing_ms": int((time.time() - started_at) * 1000),
        })
        return result

    extracted = [extract_best_face_for_document(DeepFace, doc, config) for doc in relevant_docs]
    for item in extracted:
        if item.get("detector_errors"):
            print(
                f"[FACE VERIFY DETECTOR ERRORS] doc={item.get('doc_type')} "
                f"file={item.get('filename')} errors={item.get('detector_errors')}",
                flush=True,
            )
    reference_candidates = [doc for doc in extracted if doc.get("doc_type") in REFERENCE_DOC_TYPES]
    compare_candidates = [doc for doc in extracted if doc.get("doc_type") in PHOTO_ID_DOC_TYPES]
    review_flags: List[str] = []

    if not reference_candidates:
        review_flags.append("CLAIMANT_RECENT_PHOTOGRAPH_MISSING")
        reference = None
    else:
        reference_candidates.sort(key=lambda item: float(item.get("quality_score") or 0.0), reverse=True)
        reference = reference_candidates[0]
        if not reference.get("face_found"):
            review_flags.append("CLAIMANT_RECENT_PHOTOGRAPH_FACE_MISSING")
        review_flags.extend(reference.get("review_flags") or [])

    comparisons = []
    for candidate in compare_candidates:
        if not candidate.get("face_found"):
            review_flags.append(f"{candidate.get('doc_type')}_FACE_MISSING")
        review_flags.extend(candidate.get("review_flags") or [])
        if reference is not None:
            comparison = compare_face_pair(DeepFace, reference, candidate, config)
            if comparison.get("error"):
                print(
                    f"[FACE VERIFY COMPARE ERROR] reference={reference.get('doc_type')} "
                    f"candidate={candidate.get('doc_type')} error={comparison.get('error')}",
                    flush=True,
                )
            comparisons.append(comparison)

    for comparison in comparisons:
        review_flags.extend(comparison.get("review_flags") or [])

    aggregate = aggregate_comparisons(comparisons, review_flags, config)
    result.update({
        "status": "completed",
        "decision": aggregate["decision"],
        "score": aggregate["overall_confidence"],
        "overall_confidence": aggregate["overall_confidence"],
        "reference_doc": strip_private_image(reference) if reference else None,
        "document_results": [strip_private_image(item) for item in extracted],
        "comparisons": comparisons,
        "review_required": aggregate["review_required"],
        "review_flags": aggregate["review_flags"],
        "processing_ms": int((time.time() - started_at) * 1000),
    })
    return result
