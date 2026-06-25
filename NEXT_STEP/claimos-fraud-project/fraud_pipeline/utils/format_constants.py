"""
fraud_pipeline/utils/format_constants.py
==========================================
Static regulatory format constants for India.
These rules change only when legislation changes — NOT tuned by the business team.

All business thresholds (how many claims = suspicious) are in .env / config.
"""

import re

# ── FIR Number Formats — CCTNS Standard ───────────────────────────────────────
# Rule: 2 to 6 digits, a forward slash (/), a 4-digit year, and an optional alphanumeric suffix.
FIR_RE = re.compile(r"^\d{2,6}/\d{4}(?:/[A-Z0-9]{2,6})?$")

# ── ROHINI ID Format (Hospital) ───────────────────────────────────────────────
# Format: 13-digit GLN
ROHINI_ID_RE = re.compile(r"^\d{13}$")

VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB",
}

# ── NMR (National Medical Register) Registration Numbers ─────────────────────
# Rule: Optional 2 to 4 letter state/council code, an optional "NMR" tag, 
# optional separators (hyphens, spaces, or slashes), and 4 to 7 digits.
NMR_RE = re.compile(r"^([A-Z]{2,4}[-\s/]?)?(NMR[-\s/]?)?\d{4,7}$")

# ── IFSC Code Format ──────────────────────────────────────────────────────────
# RBI standard: 4 letters + 0 + 6 alphanumeric
IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

# ── PIN Code Format ───────────────────────────────────────────────────────────
# 6-digit Indian PIN code
PIN_CODE_RE = re.compile(r"\b(\d{6})\b")

# PIN region: first 3 digits identify the postal circle + sorting district
def pin_region(pin: str) -> str:
    """Return first 3 digits as the PIN region code."""
    return pin[:3] if len(pin) == 6 and pin.isdigit() else ""


# ── PDF Forensics: Known Editing Software Keywords ────────────────────────────
# These appear in PDF /Creator or /Producer metadata fields when a PDF
# was produced by an image editor rather than a legitimate source.
PDF_EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "paint.net", "inkscape", "illustrator",
    "pixlr", "affinity", "canva", "lightroom", "capture one",
    "snapseed", "fotor", "picmonkey", "corel", "paintshop",
    "phixr", "befunky", "photopea", "polarr", "luminar",
]

# ── EXIF Forensics: Known Editing Software Keywords ──────────────────────────
# These appear in EXIF Software tag when an image was post-processed.
EXIF_EDITING_SOFTWARE_KEYWORDS = PDF_EDITING_SOFTWARE_KEYWORDS + [
    "snagit", "greenshot", "sharex", "faststone", "irfanview",
]

# ── Helper: validate ROHINI ID ────────────────────────────────────────────────
def validate_rohini_id(rohini_id: str) -> tuple[bool, str]:
    """
    Returns (is_valid: bool, reason: str).
    Valid formats: Exactly a 13-digit GLN string.
    """
    if not rohini_id or not rohini_id.strip():
        return False, "empty"
    
    if ROHINI_ID_RE.match(rohini_id.strip()):
        return True, "valid"
    
    return False, f"format_invalid: expected 13 digits, got '{rohini_id}'"


# ── Helper: validate NMR registration number ─────────────────────────────────
def validate_nmr_number(reg_number: str) -> tuple[bool, str]:
    """
    Returns (is_valid: bool, reason: str).
    Valid formats: NMR and legacy SMC formats.
    """
    if not reg_number or not reg_number.strip():
        return False, "empty"
    upper = reg_number.strip().upper()
    
    if NMR_RE.match(upper):
        return True, "valid"
    
    return False, f"invalid_nmr_format for '{upper}'"


# ── Helper: validate FIR number ──────────────────────────────────────────────
def validate_fir_number(fir_number: str, state_hint: str = "") -> tuple[bool, str, str]:
    """
    Returns (is_valid: bool, matched_state: str, reason: str).
    Valid formats: CCTNS Standard.
    """
    if not fir_number or not fir_number.strip():
        return False, "", "empty"
    upper = fir_number.strip().upper()

    if FIR_RE.match(upper):
        return True, "CCTNS", "valid"

    return False, "", f"invalid_cctns_format for '{upper}'"

