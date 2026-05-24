"""
api/middleware/auth.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    JWT authentication middleware for all /internal/* routes.
    External /v1/* routes are unauthenticated (rate-limited separately).

WHAT GOES HERE:

    JWT VALIDATION:
        Algorithm: HS256 (settings.JWT_ALGORITHM)
        Secret key: settings.JWT_SECRET_KEY
        Library: python-jose

    MIDDLEWARE BEHAVIOUR:
        On each request to /internal/*:
            1. Extract Bearer token from Authorization header
            2. Decode and validate JWT signature + expiry
            3. Extract claims: user_id, role, email
            4. Attach user context to request.state for downstream use
        On invalid/expired token: return 401 Unauthorized
        On missing token: return 401 Unauthorized
        On /v1/* or /health/*: pass through without auth check

    ROLES:
        REVIEWER — can access escalation queue, submit human review decisions
        ADMIN    — all REVIEWER permissions + can update settings thresholds
        AGENT    — for agent-to-API communication (service account token)

    TOKEN GENERATION:
        generate_token(user_id: str, role: str, email: str) → str
        Used by: login endpoint (implement separately in users router)
        Token expiry: settings.JWT_EXPIRY_MINUTES (default: 480 = 8 hours)

DEPENDENCIES:
    fastapi, python-jose, shared.config.settings
"""
