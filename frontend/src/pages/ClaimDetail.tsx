/*
 * frontend/src/pages/ClaimDetail.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE:
 *     Detailed claim view page. Shows the full claim pipeline state,
 *     all agent outputs, flags, and document processing results.
 *     Used by internal reviewers to deep-dive into any claim.
 *
 * WHAT GOES HERE:
 *
 *     ROUTE: /claims/:claimId
 *
 *     DATA FETCHING:
 *         useQuery(['claim', claimId]) → GET /internal/v1/claims/{claimId}
 *         Polls every 10s if claim is still in progress.
 *
 *     SECTIONS:
 *
 *     1. Claim Header:
 *         claim_id (copyable), policy_id, current_stage (with pipeline progress bar),
 *         sla_deadline countdown, final_decision (if complete).
 *
 *     2. Agent Output Timeline:
 *         Vertical timeline of all agent outputs in chronological order.
 *         Each agent shows: status (COMPLETE/DEGRADED/FAILED),
 *         confidence score, duration_ms, model_used, produced_at.
 *         Expandable to see full output dict.
 *
 *     3. Flags Panel:
 *         All FlagObjects grouped by severity.
 *         CRITICAL flags shown at top in red banner.
 *         Each flag shows: flag_type, evidence_doc, evidence_field, explanation.
 *
 *     4. Documents Panel:
 *         Grid of all uploaded documents with:
 *             document_type badge, trust_score gauge, tamper_score indicator.
 *         Click to expand extracted fields with per-field confidence scores.
 *
 *     5. Fraud Report Panel:
 *         Overall risk level badge, risk_score gauge.
 *         Driving signals list with weights.
 *         Legitimate explanations accordion.
 *
 *     6. Policy Assessment Panel:
 *         Coverage determination badge.
 *         Covered/excluded items table with clause citations.
 *         Benefit calculation breakdown.
 *
 *     7. Uncertainty Score:
 *         Composite score gauge with component breakdown.
 *         CONFLICTED and UNVERIFIED fields listed.
 *
 *     COMPONENTS USED:
 *         <AgentOutputPanel />, <FlagBadge />, <DocumentViewer />,
 *         <UncertaintyGauge />, <SLATimer />
 */
