"""
agents/output/human_escalation_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Builds and delivers the EscalationPackage to the human review queue.
    Implements LangGraph interrupt_before — execution pauses here until
    the human reviewer submits their decision.

WHAT GOES HERE:

    ESCALATION PACKAGE CONSTRUCTION (6 sections):

    Section 1 — Claim Snapshot:
        Key claim facts in a scannable format:
        policy_id, claimant_name, date_of_death, cause_of_death,
        sum_assured, date_intimated, days_until_sla_deadline.

    Section 2 — Completed Verifications:
        List what AI has already confirmed — human does NOT need to redo these.
        "✓ CRS registration number verified against DigiLocker"
        "✓ Premium active on date of death"
        "✓ No prior death claims on this policy"

    Section 3 — Open Questions:
        SPECIFIC questions the human must answer — NOT vague.
        Each question states:
            - What is unknown / unresolved
            - What the options are
            - What evidence supports each option
        BAD: "Please review the documents."
        GOOD: "The DEATH_CERTIFICATE lists cause of death as 'cardiac arrest'
               but the POSTMORTEM_REPORT states 'poisoning'. Are these
               medically consistent or contradictory? Options: (A) Consistent —
               cardiac arrest can result from poisoning. (B) Contradictory —
               requires further investigation."

    Section 4 — AI Recommendation:
        Labelled explicitly: "NON-BINDING AI RECOMMENDATION"
        suggested_action + specific reasoning.
        Human is not required to follow this recommendation.

    Section 5 — Document Links:
        Each uploaded document linked with AI-highlighted fields visible.
        Format: {doc_type, s3_url, highlighted_fields: list[field_name]}

    Section 6 — Structured Decision Form:
        Human selects from enum options (no free-text denial reasons).
        Required fields:
            decision         — APPROVE | PARTIAL_APPROVE | DENY
            reasoning_note   — 100–300 words (MANDATORY for DENY decisions)
            second_reviewer_required — bool checkbox

    SECOND REVIEWER TRIGGER (auto-check the checkbox if):
        net_payout > SECOND_REVIEWER_THRESHOLD (₹50 lakhs)
        OR cause_of_death = any contestability denial
        OR decision involves fraud-based denial

    LANGGRAPH INTERRUPT:
        Uses LangGraph's interrupt_before mechanism.
        Graph execution pauses before this node.
        API endpoint POST /internal/v1/human-review/{claim_id} resumes it.

    SLA REMINDERS:
        If human reviewer does not respond within 4 hours (business hours):
        Publish reminder notification to reviewer's notification queue.
        Log wait time as HUMAN_REVIEW_WAIT in audit trail.
        Wait time contributes to IRDAI SLA tracking.

DEPENDENCIES:
    langgraph (interrupt_before), shared.schemas.agent_output (EscalationPackage),
    shared.db.claim_repository, shared.events.kafka_client,
    shared.config.settings, shared.audit.audit_service
"""
