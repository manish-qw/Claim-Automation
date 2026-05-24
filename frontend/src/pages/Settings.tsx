/*
 * frontend/src/pages/Settings.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE:
 *     Admin settings page. Allows ADMIN role users to view and update
 *     configurable thresholds at runtime without code deployment.
 *
 * WHAT GOES HERE:
 *
 *     ROUTE: /settings
 *     ACCESS: ADMIN role only (redirect to dashboard if REVIEWER role)
 *
 *     DATA FETCHING:
 *         GET /internal/v1/settings → current values of all configurable params
 *
 *     SETTINGS GROUPS:
 *
 *     Group 1 — SLA Thresholds:
 *         IRDAI_ACKNOWLEDGEMENT_SLA_HOURS
 *         IRDAI_DECISION_SLA_DAYS
 *         HUMAN_REVIEW_RESPONSE_SLA_HOURS
 *         MISSING_DOCUMENT_FOLLOWUP_SLA_DAYS
 *
 *     Group 2 — Auto-Approve Thresholds:
 *         AUTO_APPROVE_MAX_SUM_ASSURED (shown in ₹ lakhs, stored in paise)
 *         SECOND_REVIEWER_THRESHOLD (shown in ₹ lakhs)
 *         CONTESTABILITY_WINDOW_DAYS
 *         SUICIDE_EXCLUSION_WINDOW_DAYS
 *
 *     Group 3 — Agent Confidence:
 *         AGENT_CONFIDENCE_MINIMUM
 *         AGENT_CONFIDENCE_WARNING
 *         UNCERTAINTY_SCORE_ESCALATION_THRESHOLD
 *         OCR_CONFIDENCE_MINIMUM
 *
 *     EDIT BEHAVIOUR:
 *         Each setting shown as read-only text with an "Edit" button.
 *         On Edit: inline input field with current value.
 *         On Save: PATCH /internal/v1/settings/{parameter_name}
 *         Show success toast + reload the page values.
 *         Show the audit trail link: "View change history →"
 *
 *     CHANGE HISTORY:
 *         Small table below each setting showing last 3 changes:
 *         (timestamp, old_value, new_value, changed_by)
 *         Data from audit trail (filter by action_type = CONFIG_CHANGE)
 */
