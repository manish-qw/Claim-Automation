"""
agents/extraction/document_intelligence_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Orchestrates the full document processing pipeline for every document
    uploaded to a claim. Uses asyncio.gather to process all documents
    in parallel (one pipeline per document simultaneously).

WHAT GOES HERE:

    MAIN ORCHESTRATION FLOW:

    Step 1 — Parallel Document Processing:
        For EACH uploaded document, launch this sequential sub-pipeline:
            ocr_engine       → extract raw text + bounding boxes
            document_classifier → identify document type (DocumentType enum)
            field_extractor  → extract typed fields specific to doc type
            document_forensics → tampering detection + trust signals
        All documents run their sub-pipelines concurrently via asyncio.gather.

    Step 2 — Cross-Document Analysis (after all documents complete):
        cross_document_checker → produce ContradictionReport
        Checks name consistency, date consistency, cause-of-death consistency
        across all documents in the claim.

    Step 3 — Trust Scoring:
        trust_score_engine → compute per-document and aggregate trust scores.
        If aggregate trust < 0.65: inject warning into orchestrator context
        for all downstream agents.

    Step 4 — Missing Document Check:
        missing_doc_tracker → compare received docs vs mandatory matrix.
        Publish follow-up requests for any mandatory missing documents.
        Pipeline continues with available documents — does NOT fully block.

    Step 5 — Write & Publish:
        Write all document outputs to ClaimContextObject.documents field
        via claim_repository.
        Publish DOC_INTEL_COMPLETE to Kafka.

    CHAIN-OF-THOUGHT STRATEGY:
        Before processing begins, the agent reasons about:
        "What type of claim is this? What should each document contain?
        What inconsistencies am I expecting to look for?"
        This context is injected into the field_extractor prompts.

    ERROR HANDLING:
        If a single document fails its sub-pipeline at any step:
        mark that document as PROCESSING_FAILED, flag for manual review,
        continue processing remaining documents. Never block the full claim
        on a single document failure.

DEPENDENCIES:
    asyncio, agents.extraction.ocr_engine, agents.extraction.document_classifier,
    agents.extraction.field_extractor, agents.extraction.document_forensics,
    agents.extraction.cross_document_checker, agents.extraction.trust_score_engine,
    agents.extraction.missing_doc_tracker, shared.db.claim_repository,
    shared.events.kafka_client, shared.schemas.agent_output
"""
