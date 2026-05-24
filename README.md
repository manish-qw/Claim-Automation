# CLAIMOS AI — Insurance Claim Automation System

> **ICICI Life Insurance** | Multi-Agent AI Pipeline for Death Claim Processing

---

## Architecture Overview

```
Claimant Submission
        │
   [Intake Agent]
        │
   [Pre-Intake Router] ──── contestability + cause-of-death routing
        │
   ┌────┴────┐
   │         │  (parallel)
[Doc Intel] [Ext Verify]
   │         │
   └────┬────┘
        │
   [Fraud Intelligence]  ─── rules + anomaly + LLM debate
   [Policy RAG Agent]    ─── frozen policy version + Pinecone
        │
   [Synthesis Agent]
        │
   [Escalation Evaluator] ── 9 hard criteria
        │
   ┌────┴──────────────┐
[Decision Agent]   [Human Escalation]
        │
[Settlement] + [Communications]
```

---

## Project Structure

```
claimos-ai/
├── shared/          # Schemas, DB, Kafka, LLM, Audit — shared by all agents
├── agents/
│   ├── extraction/  # Intake, OCR, classification, field extraction, forensics
│   ├── verification/# Aadhaar, CRS, fraud rules, anomaly detection, debate
│   ├── policy/      # RAG retrieval, benefit calculator
│   ├── decision/    # Synthesis, escalation evaluator, final decision
│   └── output/      # Human escalation, settlement, communications
├── orchestration/   # LangGraph workflow, orchestrator, conflict resolution
├── api/             # FastAPI — claimant-facing + internal endpoints
├── frontend/        # React + Vite — reviewer dashboard + claimant portal
├── infrastructure/  # Docker, Kubernetes, Terraform
└── tests/           # Unit + integration tests
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph + LangChain |
| LLM | Anthropic Claude Opus / Sonnet |
| Vector DB | Pinecone |
| Database | PostgreSQL (asyncpg) |
| Messaging | Apache Kafka (aiokafka) |
| Graph DB | Neo4j (fraud network) |
| OCR | Azure Document Intelligence → AWS Textract → PaddleOCR |
| Audit | AWS QLDB + DynamoDB Streams |
| API | FastAPI |
| Frontend | React + Vite + Tailwind CSS + shadcn/ui |
| Infra | Docker + Kubernetes + Terraform (AWS) |

---

## Setup

```bash
# 1. Clone and install Python deps
poetry install

# 2. Copy env vars
cp .env.example .env
# Fill in all API keys and DB credentials

# 3. Start local infra
docker-compose up -d

# 4. Run DB migrations
psql $DATABASE_URL -f shared/db/migrations/001_initial_schema.sql

# 5. Start API
uvicorn api.main:app --reload

# 6. Start frontend
cd frontend && npm install && npm run dev
```

---

## Environment Variables

See `.env.example` for the full list of required variables including:
- Anthropic / OpenAI API keys
- PostgreSQL connection string
- Kafka broker URLs
- Pinecone API key + index name
- AWS credentials (QLDB, S3, DynamoDB)
- Azure Document Intelligence endpoint
- UIDAI / DigiLocker / Surepass API keys
- Neo4j connection details

---

## Compliance

- **IRDAI 2024** — 24-hour acknowledgement SLA, 30-day decision SLA, mandatory denial letter content
- **DPDP Act 2023** — PII tokenisation before every LLM API call
- **Immutable Audit Trail** — SHA-256 hash chain on AWS QLDB

---

## Team

| Person | Owns |
|---|---|
| Person 1 | `shared/` + `agents/extraction/` |
| Person 2 | `agents/verification/` |
| Person 3 | `orchestration/` + `agents/policy/` + `agents/decision/` + `agents/output/` + `api/` + `frontend/` |
