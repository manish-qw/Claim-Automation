"""
api/middleware/logging_middleware.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Request/response logging middleware with claim_id correlation.
    Every API request is traceable back to a specific claim in the audit trail.

WHAT GOES HERE:

    LOGGING MIDDLEWARE:
        For every incoming request:
            1. Extract claim_id from URL path parameters (if present)
            2. Generate a request_id (UUID) for this specific request
            3. Log request: method, path, claim_id, request_id, timestamp
            4. Pass through to handler
            5. Log response: status_code, latency_ms, claim_id, request_id

    LOG FORMAT:
        Structured JSON logging (not plain text) for log aggregation:
        {
            "timestamp": "2025-01-15T10:23:45.123Z",
            "level": "INFO",
            "request_id": "uuid",
            "claim_id": "uuid | null",
            "method": "POST",
            "path": "/v1/claims",
            "status_code": 200,
            "latency_ms": 142,
            "user_id": "uuid | null"
        }

    PII SCRUBBING:
        Before logging request bodies: scrub Aadhaar numbers, PAN numbers,
        bank account numbers from request body logs.
        Replace with: "[REDACTED_PII]"
        Log bodies are safe for storage in CloudWatch/ELK.

    CORRELATION WITH AUDIT TRAIL:
        For /internal/* requests that trigger audit events:
        The request_id is passed into the AuditEvent.
        This links HTTP requests to their resulting audit events.

DEPENDENCIES:
    fastapi, starlette.middleware, re (for PII pattern matching),
    shared.config.settings
"""
