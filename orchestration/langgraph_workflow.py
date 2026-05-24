"""
orchestration/langgraph_workflow.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    The central LangGraph StateGraph. Defines every execution path through
    the CLAIMOS AI system. Every agent runs as a node in this graph.

WHAT GOES HERE:

    STATE OBJECT:
        ClaimContextObject fetched from PostgreSQL at graph entry.
        All nodes read from and write back to this state object.
        LangGraph persists state between node executions.

    GRAPH NODES (one per agent — 11 total):
        intake_node, doc_intel_node, ext_verify_node, fraud_node,
        policy_rag_node, synthesis_node, escalation_eval_node,
        decision_node, human_escalation_node, settlement_node,
        comms_node

    EDGE TYPES:

    Sequential edges:
        START → intake_node → doc_intel_node (starts immediately after intake)
        intake_node → ext_verify_node (starts simultaneously with doc_intel)

    Parallel edges (fan-out):
        intake_node → {doc_intel_node, ext_verify_node} simultaneously
        After BOTH complete → fraud_node + policy_rag_node simultaneously
        After BOTH complete → synthesis_node

    Conditional edges (routing):
        synthesis_node → escalation_eval_node
        escalation_eval_node → human_escalation_node (if ESCALATE)
        escalation_eval_node → decision_node (if PROCEED)
        decision_node → {settlement_node, comms_node} (on APPROVE/PARTIAL)
        decision_node → comms_node only (on DENY)

    INTERRUPT BEFORE escalation_node:
        LangGraph pauses execution BEFORE running human_escalation_node.
        Execution only resumes when the human reviewer submits their
        decision form via the internal API endpoint.
        See: api/routers/internal.py → POST /internal/v1/human-review/{claim_id}

    MAX ITERATION GUARD:
        Each node is allowed maximum 2 retries on failure.
        After 2 failed attempts: mark that node as FAILED, escalate claim,
        log as PIPELINE_NODE_FAILED in audit trail.

    TEMPORAL.IO WRAPPER (stub — implement when Temporal worker is deployed):
        The entire graph is intended to be wrapped in a Temporal.io workflow
        for crash recovery, checkpoint persistence, and durable execution.
        On crash: Temporal resumes from the last completed LangGraph checkpoint.
        All side-effectful operations (API calls, emails sent) carry
        idempotency keys to prevent duplication on resume.
        # TODO: wrap graph execution in Temporal workflow activity

DEPENDENCIES:
    langgraph, all agent modules, shared.db.claim_repository,
    shared.schemas.claim_context
"""
