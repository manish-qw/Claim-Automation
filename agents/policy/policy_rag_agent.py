"""
agents/policy/policy_rag_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    RAG (Retrieval-Augmented Generation) agent for policy interpretation.
    The most regulated agent in the system — every output MUST be grounded
    in actual policy text. Runs after FRAUD_COMPLETE.

WHAT GOES HERE:

    STEP 1 — Frozen Policy Version Fetch:
        Read policy_snapshot_version from claim context.
        This was frozen at claim intake (date of death — not today).
        NEVER use the current policy version if the policy was amended
        after the date of death. Always use the version active on date_of_death.

    STEP 2 — Multi-Angle Pinecone Query:
        Construct retrieval query combining:
            cause_of_death_type + specific items claimed + claim_location
            + contestability_status
        Query Pinecone with filters:
            policy_id = claim.policy_id
            policy_version = claim.policy_snapshot_version
            effective_date ≤ claim.date_of_death
        Retrieve top 20 most relevant chunks.

    STEP 3 — Always Append Exclusion Clauses:
        Regardless of retrieval score, always append ALL exclusion clauses
        for the claim's cause_of_death_type.
        Exclusions must always be considered — even if they score low
        in relevance, they could be decisive.

    STEP 4 — Cohere Rerank:
        Re-rank the retrieved + appended chunks by relevance using
        Cohere Rerank API before sending to Claude.
        This ensures the most relevant chunks appear first in the prompt.

    STEP 5 — Claude Opus with Strict Grounding Prompt:
        System prompt (invariant):
            "You may only use the policy text provided in this context.
             Never use your general knowledge about insurance.
             If a situation is not addressed in the provided text,
             write exactly: NOT_ADDRESSED_IN_POLICY
             Every coverage determination must cite the exact clause ID
             from the text provided."
        Output: PolicyAssessment (see shared/schemas/agent_output.py)

    STEP 6 — Citation Validator (post-generation, before writing):
        Extract all clause IDs cited in the PolicyAssessment output.
        Look up each clause ID in the retrieved chunks.
        If a cited clause does NOT exist in the retrieved context:
            → REJECT the output
            → Increment retry counter
            → Retry Claude with additional instruction:
              "Your previous response cited clause [X] which was not in
               the provided context. Do not cite clauses not present."
        If second attempt also has unverified citations:
            → Return PolicyAssessment with all_citations_verified = False
            → Log as GROUNDING_VALIDATION_FAILED in audit trail

    OUTPUT:
        Write PolicyAssessment to ClaimContextObject via claim_repository.
        Publish POLICY_COMPLETE to Kafka.

DEPENDENCIES:
    pinecone, cohere, shared.llm.llm_client (LLMModel.OPUS),
    shared.events.kafka_client, shared.db.claim_repository,
    shared.schemas.agent_output (PolicyAssessment)
"""
