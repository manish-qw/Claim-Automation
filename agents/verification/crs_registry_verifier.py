"""
agents/verification/crs_registry_verifier.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Verifies the death certificate's CRS (Civil Registration System) registration
    number against the government database. THE most important verification in
    the system — forged death certificates (Sambhal mafia pattern) have fake
    CRS numbers that do not exist in any registry.

WHAT GOES HERE:

    INPUT:
        crs_registration_number — extracted from DEATH_CERTIFICATE by field_extractor
        place_of_death          — used for state detection and routing

    STATE DETECTION:
        Extract the state from place_of_death field.
        Map state to its specific municipal portal API endpoint.
        State → endpoint mapping defined in this file as a dict constant.
        Supported states with dedicated endpoints:
            Maharashtra, Karnataka, Delhi, Uttar Pradesh, Tamil Nadu,
            Gujarat, Rajasthan, West Bengal, Telangana, Kerala, + others.

    API CASCADE (try in order until one succeeds):
        1. DigiLocker CRS API (primary)
           Direct government database lookup.
        2. Surepass Death Certificate Verification API (fallback)
        3. State-specific municipal portal API (second fallback)
           Routed based on detected state from place_of_death.

    CRSVerificationResult (dataclass):
        crs_number             — the registration number checked
        status                 — enum: CRS_VERIFIED | CRS_NOT_FOUND | CRS_UNVERIFIED
        verified_via           — "DIGILOCKER" | "SUREPASS" | "STATE_PORTAL" | "NONE"
        certificate_data       — dict of fields returned by registry (if found)
        verified_at            — UTC timestamp
        failure_reason         — None | reason code

    OUTPUT STATES:

    CRS_VERIFIED:
        Registration number found and matches. Certificate is authentic.
        No flag. Strong positive signal.

    CRS_NOT_FOUND:
        Number not found in DigiLocker, Surepass, AND the state portal.
        Raise CRITICAL FlagObject:
            flag_type   = "CRS_CERTIFICATE_NOT_FOUND"
            severity    = CRITICAL
            explanation = "Death certificate registration number [X] was not
                          found in the CRS government database via DigiLocker,
                          Surepass, and the [State] municipal portal. This may
                          indicate a forged death certificate."
        Triggers automatic escalation via escalation_evaluator criterion #3.

    CRS_UNVERIFIED:
        All APIs returned errors or timed out.
        Mark as UNVERIFIED. Flag for mandatory human manual verification.
        No approval decision can be made without CRS verification.

DEPENDENCIES:
    httpx, shared.config.settings, shared.schemas.agent_output (FlagObject),
    shared.audit.audit_service
"""
