/*
 * frontend/src/pages/AuditTrail.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE:
 *     Audit trail viewer page. Shows the full immutable audit event chain
 *     for a claim. Used for regulatory review and dispute resolution.
 *
 * WHAT GOES HERE:
 *
 *     ROUTE: /audit/:claimId
 *
 *     DATA FETCHING:
 *         useQuery(['audit', claimId]) → GET /internal/v1/audit/{claimId}
 *         Returns: list[AuditEvent] + chain_integrity_valid (bool)
 *
 *     CHAIN INTEGRITY INDICATOR:
 *         Green banner: "✓ Hash chain verified — audit trail is intact"
 *         Red banner:   "⚠ Chain integrity check FAILED — possible tampering"
 *
 *     AUDIT EVENT LIST (chronological, oldest first):
 *         For each AuditEvent:
 *             timestamp, agent_id, action_type badge,
 *             confidence, duration_ms, model_version.
 *         Expandable to show:
 *             input_hash, output_hash, previous_entry_hash,
 *             reasoning_trace (full chain-of-thought text),
 *             tool_calls list with param/response hashes.
 *
 *     FILTER BAR:
 *         Filter by: agent_id, action_type, date range, confidence range.
 *         Search by: flag_type, audit_id.
 *
 *     EXPORT BUTTON:
 *         "Download IRDAI Report" → GET /internal/v1/audit/{claimId}/export
 *         Downloads PDF audit report.
 *
 *     HASH COPY:
 *         Each hash value has a copy-to-clipboard button.
 *         Useful for manual hash chain verification.
 *
 *     COMPONENTS USED:
 *         <AuditEventRow />, <HashDisplay />, <ReasoningTraceAccordion />
 */
