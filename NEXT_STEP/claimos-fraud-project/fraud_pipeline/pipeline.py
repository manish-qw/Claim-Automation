"""Pipeline Orchestrator — runs all 7 agents in sequence."""
import logging
from dataclasses import asdict
from .schemas import ClaimState, FraudIntelligencePackage
from .agents import (
    ExternalVerificationEngine, FraudIntelligenceAgent,
    EarlyClaimIntelligenceAgent, NonDisclosureDetectionAgent,
    ConflictResolutionAgent, TrustGovernanceAgent, ClaimEvidenceGraphEngine,
)

logger = logging.getLogger(__name__)


class FraudPipeline:
    def __init__(self):
        self.agent1 = ExternalVerificationEngine()
        self.agent2 = FraudIntelligenceAgent()
        self.agent3 = EarlyClaimIntelligenceAgent()
        self.agent4 = NonDisclosureDetectionAgent()
        self.agent5 = ConflictResolutionAgent()
        self.agent6 = TrustGovernanceAgent()
        self.agent7 = ClaimEvidenceGraphEngine()

    def run(self, state: ClaimState) -> FraudIntelligencePackage:
        logger.info("=== Fraud Pipeline START: %s ===", state.claim_case_id)
        steps = [
            ("External Verification", self.agent1.run),
            ("Fraud Intelligence",    self.agent2.run),
            ("Early Claim",           self.agent3.run),
            ("Non-Disclosure",        self.agent4.run),
            ("Conflict Resolution",   self.agent5.run),
            ("Trust Governance",      self.agent6.run),
            ("Graph Engine",          self.agent7.run),
        ]
        for name, fn in steps:
            try:
                state = fn(state)
                logger.info("  ✓ %s", name)
            except Exception as exc:
                logger.exception("  ✗ %s failed: %s", name, exc)
                state.validation_flags.append(f"AGENT_ERROR:{name}:{exc}")

        state.final_recommendation = self._derive_recommendation(state)
        state.escalation_required = self._derive_escalation(state)
        logger.info("=== Pipeline END: %s | %s | escalate=%s ===",
                    state.claim_case_id, state.final_recommendation, state.escalation_required)

        return FraudIntelligencePackage(
            claim_case_id=state.claim_case_id,
            fraud_analysis=state.fraud_analysis or {},
            trust_analysis=state.trust_analysis or {},
            external_verification=state.external_verification or {},
            early_claim_analysis=state.early_claim_analysis or {},
            non_disclosure_analysis=state.non_disclosure_analysis or {},
            conflict_resolution=state.conflict_resolution or {},
            graph_analysis=state.graph_analysis or {},
            final_recommendation=state.final_recommendation,
            escalation_required=state.escalation_required,
        )

    def _derive_recommendation(self, state):
        fraud_action  = (state.fraud_analysis or {}).get("recommended_action", "APPROVE")
        trust         = (state.trust_analysis or {}).get("overall_trust_score", 1.0)
        fraud_score   = (state.fraud_analysis or {}).get("fraud_risk_score", 0.0)
        fraud_ring    = (state.graph_analysis or {}).get("fraud_ring_detected", False)
        conflict_act  = (state.conflict_resolution or {}).get("resolved_action", "ACCEPT")
        non_disc      = (state.non_disclosure_analysis or {}).get("contradiction_detected", False)
        if fraud_ring or fraud_score >= 0.85:
            return "REJECT"
        if fraud_action == "ESCALATE" or conflict_act == "ESCALATE" or non_disc:
            return "ESCALATE"
        if trust < 0.65 or fraud_action == "REVIEW":
            return "INVESTIGATE"
        return "APPROVE"

    def _derive_escalation(self, state):
        # APPROVE always means no escalation — override any stale agent flags
        if state.final_recommendation == "APPROVE":
            return False
        return state.final_recommendation in ("ESCALATE", "REJECT") or state.escalation_required
