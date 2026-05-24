/*
 * frontend/src/pages/ReviewForm.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE:
 *     Human reviewer decision form page. Shows the full EscalationPackage
 *     and collects the reviewer's decision.
 *
 * WHAT GOES HERE:
 *
 *     ROUTE: /review/:claimId
 *
 *     DATA FETCHING:
 *         useQuery(['escalation-package', claimId])
 *         → GET /internal/v1/escalation/queue (filtered by claimId)
 *         Loads the full EscalationPackage for this claim.
 *
 *     6 SECTIONS (mirror EscalationPackage structure):
 *
 *     Section 1 — Claim Snapshot:
 *         Summary table: policy_id, claimant_name, date_of_death,
 *         cause_of_death, sum_assured, days_until_sla.
 *         SLA countdown timer (red if < 24h).
 *
 *     Section 2 — Completed Verifications (read-only):
 *         Checkmark list of what AI has confirmed.
 *         Reviewer does NOT redo these.
 *         e.g. "✓ CRS registration verified", "✓ No prior death claims"
 *
 *     Section 3 — Open Questions (primary focus area):
 *         Each question in an expandable card.
 *         Shows: what is unknown, what options exist, evidence for each option.
 *         Reviewer must address each question in their reasoning note.
 *
 *     Section 4 — AI Recommendation (clearly labelled NON-BINDING):
 *         AI's suggested_action + reasoning.
 *         Visual badge: "NON-BINDING AI RECOMMENDATION"
 *
 *     Section 5 — Documents:
 *         Thumbnail list of all uploaded documents with links to S3 viewer.
 *         AI-highlighted fields shown on hover.
 *
 *     Section 6 — Decision Form (sticky bottom bar or modal):
 *         decision         — radio: APPROVE | PARTIAL_APPROVE | DENY
 *         reasoning_note   — textarea (required for DENY, min 100 words counter shown)
 *         second_reviewer  — checkbox (auto-checked for high-value claims)
 *         Submit button → POST /internal/v1/human-review/{claimId}
 *         On success: navigate to Dashboard with success toast.
 *
 *     COMPONENTS USED:
 *         <FlagBadge />        — CRITICAL/HIGH/MEDIUM/LOW severity indicators
 *         <DocumentViewer />   — document thumbnail + field highlights
 *         <WordCounter />      — live word count for reasoning_note
 *         <SLATimer />         — countdown with urgency colour change
 */
