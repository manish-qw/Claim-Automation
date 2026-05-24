"""
api/routers/internal.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Internal endpoints for agents, human reviewers, and the admin dashboard.
    All routes require JWT authentication (enforced by AuthMiddleware).

ENDPOINTS TO IMPLEMENT:

    POST /internal/v1/agent/output
        Agent posts its AgentOutput envelope after completion.
        Used by agents that run as separate services (not in-process).
        Body: AgentOutput (validated against schema).
        Action: writes to claim context via claim_repository.update_agent_output().
        Response: {accepted: true, next_stage: str}

    GET /internal/v1/escalation/queue
        Returns the list of claims currently in ESCALATED stage,
        ordered by sla_deadline ASC (most urgent first).
        Response: list[ClaimSummary] with key facts for each escalated claim.
        Used by the human reviewer dashboard to populate the queue.

    POST /internal/v1/human-review/{claim_id}
        Human reviewer submits their decision.
        Body: {decision, reasoning_note (100-300 words for DENY),
               second_reviewer_required: bool, reviewer_id}
        Action: resumes LangGraph execution from interrupt_before point.
        Response: {claim_id, resumed: true}
        Validates: reasoning_note required for DENY, decision is valid enum value.

    GET /internal/v1/claims/dashboard
        Claims queue for the internal dashboard.
        Query params: stage (filter), limit, offset, sort_by.
        Response: paginated list of ClaimSummary with stage counts.

    GET /internal/v1/audit/{claim_id}
        Full audit trail for a claim.
        Response: list[AuditEvent] in chronological order.
        Also returns: chain_integrity_valid (bool from audit_service.verify_chain_integrity).

    GET /internal/v1/audit/{claim_id}/export
        Download IRDAI-format PDF audit report.
        Response: PDF bytes as file download.
        Calls: audit_service.export_regulatory_report(claim_id).

    PATCH /internal/v1/settings/{parameter_name}
        Update a configurable threshold at runtime.
        Body: {new_value, changed_by (reviewer_id)}
        Action: updates settings, logs configuration change AuditEvent.
        Restricted to: ADMIN role JWT claims only.

DEPENDENCIES:
    fastapi, shared.db.claim_repository, shared.audit.audit_service,
    orchestration.langgraph_workflow (resume), api.middleware.auth,
    shared.config.settings
"""
