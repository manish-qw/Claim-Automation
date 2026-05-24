"""
agents/output/communications_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Generates and queues all outbound communications to claimants.
    Template-constrained generation — LLM fills reason/explanation sections
    only. Factual fields are injected deterministically — LLM never generates
    numbers, dates, or policy identifiers.

WHAT GOES HERE:

    COMMUNICATION TYPES:
        1. Acknowledgement Letter    — sent within 24h of intake (IRDAI mandate)
        2. Document Follow-up        — targeted request for specific missing doc
        3. Approval Letter           — with itemised payout breakdown
        4. Denial Letter             — with mandatory IRDAI content
        5. Partial Approval Letter   — with what is covered and what is excluded
        6. SLA Update                — if processing exceeds expected timeline
        7. Settlement Confirmation   — after payment is credited

    FACTUAL FIELD INJECTION (deterministic — never via LLM):
        policy_number, payout_amount, decision_date, tds_amount, claim_id,
        sla_deadline, document_request_url, claimant_name, bank_account_last4.
        These are inserted into templates after LLM generation.
        LLM NEVER generates these values — prevents hallucinated amounts.

    DENIAL LETTER — MANDATORY IRDAI 2024 CONTENT:
        Every denial letter MUST include:
        a) Specific policy clause(s) cited for each denial reason
        b) Claimant's right to appeal the decision
        c) Insurance Ombudsman contact for claimant's state
           (lookup: shared/schemas/document_types.py → OMBUDSMAN_BY_STATE)
        d) 30-day appeal window start date
        e) How to file a grievance with IRDAI IGMS portal

    LANGUAGE DETECTION AND TRANSLATION:
        Read preferred_language from intake_channel submission.
        Supported: English (default), Hindi, Marathi, Tamil, Bengali, Telugu.
        If non-English: use Claude Sonnet with translation instruction.
        Factual fields are translated separately using lookup tables —
        NOT via LLM (to prevent translation errors in amounts/dates).
        e.g. ₹5,00,000 is inserted as-is regardless of language.

    OUTPUT FORMAT:
        Generate BOTH:
            email_body  — HTML formatted email
            pdf_letter  — PDF document via reportlab
        Queue both for delivery via verified delivery service.
        Do NOT send directly — push to delivery queue.
        Delivery confirmation is tracked separately.

    CommunicationRecord (dataclass):
        communication_id  — UUID
        claim_id          — UUID
        communication_type — string
        language          — string
        email_body_hash   — SHA-256 of generated email body
        pdf_generated     — bool
        queued_at         — UTC timestamp
        delivery_status   — "QUEUED" | "DELIVERED" | "FAILED"

DEPENDENCIES:
    shared.llm.llm_client (LLMModel.SONNET), reportlab,
    shared.schemas.document_types (OMBUDSMAN_BY_STATE),
    shared.db.claim_repository, shared.audit.audit_service,
    shared.config.settings
"""
