"""
Agent 3 — Early Claim Intelligence Agent

Responsibilities:
  - Detect recently issued policies (early claims)
  - Detect policy revivals after lapse
  - Detect premium payment irregularities
  - Increase investigation depth based on findings
"""

import logging
from datetime import datetime, timezone
from ..schemas import ClaimState, EarlyClaimOutput

logger = logging.getLogger(__name__)


class EarlyClaimIntelligenceAgent:
    """
    Analyses policy age, revival events, and premium irregularities.
    Enriches ClaimState.early_claim_analysis.
    """

    VERY_HIGH_RISK_DAYS = 90
    HIGH_RISK_DAYS = 180
    MEDIUM_RISK_DAYS = 365

    def run(self, state: ClaimState) -> ClaimState:
        logger.info("[Agent 3] Early Claim Intelligence → claim %s", state.claim_case_id)

        policy_age = state.policy_age_days
        if policy_age == 0 and state.policy_issue_date and state.death_information.date_of_death:
            policy_age = self._compute_policy_age(
                state.policy_issue_date, state.death_information.date_of_death
            )
            state.policy_age_days = policy_age

        risk_factors = []
        revival_detected = self._detect_revival(state, risk_factors)
        premium_irregularities = self._detect_premium_irregularities(state, risk_factors)

        # Risk classification
        if policy_age > 0 and policy_age < self.VERY_HIGH_RISK_DAYS:
            early_claim_risk = "VERY_HIGH"
            risk_factors.append(f"Policy only {policy_age} days old — very early claim")
            confidence = 0.95
        elif policy_age > 0 and policy_age < self.HIGH_RISK_DAYS:
            early_claim_risk = "HIGH"
            risk_factors.append(f"Policy {policy_age} days old — early claim threshold")
            confidence = 0.85
        elif policy_age > 0 and policy_age < self.MEDIUM_RISK_DAYS:
            early_claim_risk = "MEDIUM"
            risk_factors.append(f"Policy {policy_age} days old — within 1 year")
            confidence = 0.65
        else:
            early_claim_risk = "LOW"
            confidence = 0.40

        if revival_detected:
            # Upgrade risk if revival detected
            if early_claim_risk == "LOW":
                early_claim_risk = "MEDIUM"
            elif early_claim_risk == "MEDIUM":
                early_claim_risk = "HIGH"
            confidence = min(confidence + 0.10, 0.99)

        trust = round(1.0 - (confidence * 0.6), 4) if early_claim_risk != "LOW" else 0.85

        output = EarlyClaimOutput(
            policy_age_days=policy_age,
            policy_revival_detected=revival_detected,
            premium_irregularities=premium_irregularities,
            early_claim_risk=early_claim_risk,
            risk_factors=risk_factors,
            confidence_score=confidence,
            trust_score=trust,
            validation_flags=[f"EARLY_CLAIM:{early_claim_risk}"] if early_claim_risk != "LOW" else [],
        )

        state.early_claim_analysis = output.model_dump()
        if early_claim_risk in ("HIGH", "VERY_HIGH"):
            state.validation_flags.append(f"EARLY_CLAIM:{early_claim_risk}")
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_policy_age(self, issue_date: str, death_date: str) -> int:
        try:
            issue = datetime.fromisoformat(issue_date)
            death = datetime.fromisoformat(death_date)
            return max(0, (death - issue).days)
        except Exception:
            return 0

    def _detect_revival(self, state: ClaimState, risk_factors: list) -> bool:
        """
        A revival is detected when payment history shows a lapse gap of >90 days
        followed by resumption within 180 days before the death date.
        """
        payments = state.premium_payment_history
        if len(payments) < 2:
            return False

        death_date = state.death_information.date_of_death
        if not death_date:
            return False

        try:
            death_dt = datetime.fromisoformat(death_date)
        except ValueError:
            return False

        sorted_payments = sorted(payments, key=lambda p: p.get("date", ""))
        gaps = []
        for i in range(1, len(sorted_payments)):
            try:
                prev = datetime.fromisoformat(sorted_payments[i - 1]["date"])
                curr = datetime.fromisoformat(sorted_payments[i]["date"])
                gap = (curr - prev).days
                gaps.append((gap, curr))
            except (ValueError, KeyError):
                continue

        for gap_days, resume_date in gaps:
            if gap_days > 90:
                days_to_death = (death_dt - resume_date).days
                if days_to_death < 0:
                    # Zombie premium: policy revived AFTER the person died!
                    risk_factors.append(
                        f"CRITICAL: Policy revived {abs(days_to_death)} days AFTER death date (Zombie Premium)"
                    )
                    return True
                elif 0 <= days_to_death < 180:
                    risk_factors.append(
                        f"Policy lapsed {gap_days} days then revived {days_to_death} days before death"
                    )
                    return True
        return False

    def _detect_premium_irregularities(self, state: ClaimState, risk_factors: list) -> bool:
        """
        Flags if premiums were paid in irregular bulk (e.g., 3+ back-payments at once).
        """
        payments = state.premium_payment_history
        for p in payments:
            if p.get("months_covered", 1) >= 3 and p.get("bulk_payment", False):
                risk_factors.append(
                    f"Bulk premium payment detected: {p.get('months_covered')} months paid at once"
                )
                return True
        return False
