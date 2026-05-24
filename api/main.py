"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    FastAPI application entry point. Mounts all routers, middleware,
    and startup/shutdown lifecycle hooks. Serves both external
    (claimant-facing) and internal (agent + human reviewer) endpoints.

WHAT GOES HERE:

    FASTAPI APP SETUP:
        FastAPI instance with:
            title = "CLAIMOS AI — Claim Automation API"
            version = "1.0.0"
            docs_url = "/docs" (disabled in production)
            OpenAPI tags for grouping endpoints

    MIDDLEWARE (in order, outer to inner):
        1. CORSMiddleware — allow frontend origin (settings.FRONTEND_URL)
        2. LoggingMiddleware — logs every request/response with claim_id
           correlation (see api/middleware/logging_middleware.py)
        3. AuthMiddleware — JWT validation for /internal/* routes
           (see api/middleware/auth.py)

    ROUTERS MOUNTED:
        router_claims   → prefix="/v1/claims"    (external claimant endpoints)
        router_internal → prefix="/internal/v1"  (internal + reviewer endpoints)
        router_health   → prefix="/health"        (health + readiness probes)

    STARTUP EVENTS:
        Initialize asyncpg connection pool (from claim_repository)
        Initialize Kafka producer (from kafka_client)
        Load Isolation Forest model from S3 (from anomaly_detector)
        Log startup as AuditEvent type APP_STARTUP

    SHUTDOWN EVENTS:
        Flush Kafka producer buffer
        Close asyncpg pool gracefully
        Log shutdown as AuditEvent type APP_SHUTDOWN

DEPENDENCIES:
    fastapi, uvicorn, api.routers.claims, api.routers.internal,
    api.routers.health, api.middleware.auth, api.middleware.logging_middleware,
    shared.db.claim_repository, shared.events.kafka_client,
    shared.config.settings
"""
