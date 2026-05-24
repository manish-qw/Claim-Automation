"""
agents/verification/aadhaar_verifier.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Verifies the deceased person's Aadhaar card status via the UIDAI API.
    An active Aadhaar on a claimed-deceased person is a CRITICAL fraud signal.

WHAT GOES HERE:

    INPUT:
        deceased_aadhaar_number — extracted from AADHAAR_CARD document
                                  by field_extractor

    API CALL SEQUENCE:
        Primary:  UIDAI deceased deactivation status endpoint
                  (Available since June 2025 — families can report deaths
                   and request Aadhaar deactivation.)
                  Endpoint: settings.UIDAI_API_BASE_URL
        Fallback: Surepass Aadhaar verification API
                  Endpoint: settings.SUREPASS_API_BASE_URL

    AadhaarVerificationResult (dataclass):
        aadhaar_number_masked  — last 4 digits visible, rest masked
        status                 — enum: INACTIVE | ACTIVE | UNVERIFIED
        verified_at            — UTC timestamp
        api_used               — "UIDAI" | "SUREPASS" | "NONE"
        failure_reason         — None | "API_TIMEOUT" | "API_UNAVAILABLE" etc.

    OUTPUT STATES:

    INACTIVE:
        Aadhaar deactivated → consistent with death claim.
        No flag raised. Adds positive signal to verification report.

    ACTIVE:
        Aadhaar still active on claimed-deceased person.
        Raise CRITICAL FlagObject:
            flag_type   = "AADHAAR_ACTIVE_ON_DECEASED"
            severity    = CRITICAL
            explanation = "Deceased's Aadhaar [masked] remains active as of
                          [date]. Aadhaar is typically deactivated within
                          weeks of death registration."
        This CRITICAL flag triggers automatic escalation via
        escalation_evaluator criterion #4.

    UNVERIFIED:
        Both APIs unavailable.
        Mark field UNVERIFIED with reason code.
        Add to human manual verification checklist with HIGH priority.
        For claims above ₹5 lakhs: this UNVERIFIED status automatically
        increases the claim's uncertainty_score significantly.

    PII HANDLING:
        Full Aadhaar number is tokenised by llm_client before any API call.
        Only the masked version is stored in logs and audit events.

DEPENDENCIES:
    httpx, shared.config.settings, shared.schemas.agent_output (FlagObject),
    shared.audit.audit_service
"""
