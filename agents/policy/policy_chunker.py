"""
agents/policy/policy_chunker.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    One-time pipeline run whenever a new policy document is uploaded into the
    system. Processes the policy PDF and loads it into Pinecone for RAG.
    Not called per-claim — called per policy document upload event.

WHAT GOES HERE:

    INPUT:
        policy_pdf_bytes   — raw bytes of the policy document PDF
        policy_id          — UUID
        policy_version     — version string (e.g. "UIN-512N346V03")
        effective_date     — date from which this version is active
        expiry_date        — date until which this version is active
        document_type      — "BASE_POLICY" | "ENDORSEMENT" | "RIDER"
        endorsement_id     — None for base policy, UUID for endorsements

    CHUNKING STRATEGY:
        Split by LOGICAL CLAUSE STRUCTURE using PDF heading detection —
        NOT by fixed token length (token-length chunking breaks clause
        context and produces partial clauses).
        Each chunk = one complete sub-clause OR one definition entry.
        Use PyMuPDF to detect heading levels (H1, H2, H3) from font size.
        Split at each heading boundary, keeping the heading with the content.

    CHUNK METADATA (stored as Pinecone filterable fields per chunk):
        policy_id         — UUID
        policy_version    — version string
        effective_date    — date
        expiry_date       — date
        clause_id         — extracted from the heading (e.g. "4.2.1(b)")
        section_number    — parent section number
        section_title     — e.g. "Exclusions", "Death Benefit", "Definitions"
        clause_type       — enum: INCLUSION | EXCLUSION | LIMIT |
                                  DEFINITION | CONDITION | PROCEDURE | RIDER_TERM
        requires_external_statute — bool (True if clause references an external law)
        external_statute_name     — e.g. "Motor Vehicles Act 1988"

    EMBEDDING GENERATION:
        Model: BAAI/bge-large-en-v1.5 (Hugging Face)
        Run locally — do not send policy text to external embedding APIs
        for data security reasons.
        Each chunk is embedded separately.

    PINECONE UPSERT:
        Upsert all chunks with their embeddings and metadata.
        Use namespace = policy_id for isolation between policies.

    ENDORSEMENT HANDLING:
        Endorsement chunks are tagged with endorsement_id and effective_date.
        Precedence rule stored in metadata: endorsement clause OVERRIDES
        base policy clause with the same clause_id.
        Policy RAG agent must respect this precedence.

    EXTERNAL STATUTE FLAGGING:
        If a clause contains language like "subject to [Act Name] [Year]":
        Set requires_external_statute = True.
        Policy RAG agent will output NOT_ADDRESSED_IN_POLICY for those
        clauses and mark them as REQUIRES_LEGAL_INTERPRETATION.

DEPENDENCIES:
    PyMuPDF (fitz), sentence-transformers (BAAI/bge-large),
    pinecone, shared.config.settings
"""
