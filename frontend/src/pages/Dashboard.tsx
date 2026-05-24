/*
 * frontend/src/pages/Dashboard.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE:
 *     Main dashboard page for human reviewers. Shows the live claim queue,
 *     SLA countdowns, stage distribution, and key metrics.
 *
 * WHAT GOES HERE:
 *
 *     DATA FETCHING (TanStack Query):
 *         useQuery(['escalation-queue']) → GET /internal/v1/escalation/queue
 *         useQuery(['dashboard-stats'])  → GET /internal/v1/claims/dashboard
 *         Auto-refetch every 30 seconds to keep queue live.
 *
 *     SECTIONS TO RENDER:
 *
 *     1. Header Stats Bar:
 *         Total claims today | Escalated (urgent) | Avg processing time |
 *         SLA compliance rate (%)
 *
 *     2. Escalated Claims Queue (primary panel):
 *         List of claims requiring human review, sorted by sla_deadline ASC.
 *         Each row shows: claim_id, claimant_name, cause_of_death,
 *         sum_assured, escalation_reason, sla_deadline countdown timer.
 *         Clicking a row → navigate to /review/{claim_id}
 *
 *     3. Pipeline Stage Visualization:
 *         Horizontal bar or donut chart showing claim counts per stage.
 *         Stages: INTAKE → DOC_INTEL → VERIFICATION → FRAUD → POLICY →
 *                 SYNTHESIS → DECISION → ESCALATED → SETTLED | REJECTED
 *         Data from dashboard-stats endpoint.
 *
 *     4. SLA Breach Risk Panel:
 *         Claims with sla_deadline < 48h highlighted in red.
 *         Claims with sla_deadline < 7d highlighted in amber.
 *
 *     COMPONENTS USED:
 *         <ClaimCard /> — one card per escalated claim
 *         <SLATimer />  — real-time countdown to sla_deadline
 *         <StageChart /> — pipeline stage distribution (recharts)
 *         <MetricCard /> — header stat boxes
 */
