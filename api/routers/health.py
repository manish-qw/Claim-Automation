"""
api/routers/health.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Health and readiness check endpoints for Kubernetes probes and
    load balancer monitoring.

ENDPOINTS TO IMPLEMENT:

    GET /health/live
        Liveness probe — is the process alive?
        Returns: {status: "ok"} always (if process is up, it responds).
        Used by Kubernetes to know if the pod needs restart.

    GET /health/ready
        Readiness probe — is the service ready to accept traffic?
        Checks all dependencies:
            PostgreSQL: run a lightweight SELECT 1 query
            Kafka: check producer is connected
            Pinecone: ping the index
        Returns: {status: "ready" | "degraded", checks: {db, kafka, pinecone}}
        Returns HTTP 200 if all checks pass, HTTP 503 if any fail.
        Used by Kubernetes to stop routing traffic during startup or
        when dependencies are down.

    GET /health/metrics
        Basic application metrics for monitoring.
        Returns: {
            claims_processed_today: int,
            claims_in_queue: int,
            avg_processing_time_ms: float,
            escalation_rate: float,
            kafka_consumer_lag: int
        }
        Feeds into Grafana dashboard.

DEPENDENCIES:
    fastapi, shared.db.claim_repository, shared.events.kafka_client,
    pinecone (ping)
"""
