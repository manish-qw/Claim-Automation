"""
schemas/claim_state.py — All output dataclasses for the fraud pipeline.
Fields match exactly what each agent produces, plus model_dump() for serialization.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _uid() -> str:
    return f"AUD-{uuid.uuid4().hex[:12].upper()}"


@dataclass
class BaseAgentOutput:
    confidence_score: float = 0.0
    trust_score: float = 0.0
    validation_flags: List[str] = field(default_factory=list)
    audit_reference: str = field(default_factory=_uid)
    created_at: str = field(default_factory=_now)

    def model_dump(self) -> dict:
        return asdict(self)


# ── Re-export input dataclasses from base for convenience ─────────────────────
from .base import (

    Claimant, LifeAssured, DeathInformation,
    SubmittedDocument, MedicalRecord, FIRRecord,
    ClaimState,
)

# Aliases used in older agent files


# ── Agent Output Schemas ───────────────────────────────────────────────────────

@dataclass
class ExternalVerificationOutput(BaseAgentOutput):
    # ── Per-area partial float scores (0.0–1.0) ─────────────────────────────
    # These are the primary outputs. A component not applicable to this claim
    # type returns 1.0 (neutral) so it does not unfairly reduce overall score.
    hospital_score: float = 1.0          # ROHINI ID validity + frequency check
    doctor_score: float = 1.0            # NMR registration validity + frequency
    fir_score: float = 1.0               # FIR format + date arithmetic + jurisdiction
    metadata_score: float = 1.0          # OCR confidence + PDF/EXIF metadata
    image_score: float = 1.0             # SHA-256 hash + ELA forensics + EXIF editing
    geo_score: float = 1.0               # PIN code region + LLM entity extraction

    # ── Backward-compatible bool fields (derived from float scores) ──────────
    # True if score >= 0.70 (passes), False if below threshold (fails)
    hospital_verified: bool = True
    doctor_verified: bool = True
    fir_verified: bool = True
    geo_location_match: bool = True
    image_manipulation_detected: bool = False   # True if image_score < 0.70
    metadata_tampered: bool = False             # True if metadata_score < 0.80

    # ── Aggregate ────────────────────────────────────────────────────────────
    verification_confidence: float = 0.0        # weighted combination of all 6 scores
    hospital_verification_detail: Optional[str] = None
    fir_verification_detail: Optional[str] = None
    metadata_issues: List[str] = field(default_factory=list)

    # ── Evidence references (required for Explainability PDF) ────────────────
    # Maps each flag string to the document/field that triggered it.
    # e.g. {"ROHINI_ID_MISSING": "death_information.hospital_rohini_id",
    #        "ELA_ANOMALY_DETECTED:death_certificate": "submitted_documents[0].file_path"}
    flag_evidence: Dict[str, str] = field(default_factory=dict)


@dataclass
class FraudAnalysisOutput(BaseAgentOutput):
    fraud_risk_score: float = 0.0
    fraud_risk_level: str = "LOW"
    fraud_reasons: List[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    suspicious_nominee: bool = False
    recommended_action: str = "APPROVE"


# Alias used by agent2
FraudIntelligenceOutput = FraudAnalysisOutput


@dataclass
class EarlyClaimOutput(BaseAgentOutput):
    policy_age_days: int = 0
    policy_revival_detected: bool = False
    days_since_revival: Optional[int] = None
    premium_irregularities: bool = False
    lapsed_periods: List[str] = field(default_factory=list)
    early_claim_risk: str = "LOW"
    risk_factors: List[str] = field(default_factory=list)          # agent3


@dataclass
class NonDisclosureOutput(BaseAgentOutput):
    contradiction_detected: bool = False
    non_disclosure_findings: List[str] = field(default_factory=list)
    non_disclosure_score: float = 0.0
    proposal_disclosures: Dict[str, Any] = field(default_factory=dict)
    medical_findings: Dict[str, Any] = field(default_factory=dict)
    proposal_claims: Dict[str, Any] = field(default_factory=dict)  # agent4
    medical_facts: Dict[str, Any] = field(default_factory=dict)    # agent4
    api_failed: bool = False
    inference_reasoning: str = ""


@dataclass
class ConflictResolutionOutput(BaseAgentOutput):
    conflict_detected: bool = False
    conflicts: List[Any] = field(default_factory=list)
    conflicts_found: List[Any] = field(default_factory=list)       # agent5
    resolved_action: str = "ACCEPT"
    winning_source: Optional[str] = None
    resolution_rationale: Optional[str] = None
    resolution_reasoning: str = ""                                  # agent5
    # ── Medical Relationship Service fields ──────────────────────────────────
    medical_relationship_found: bool = False
    relationship_type: str = ""          # identical|synonym|is_a_parent|disease_family|direct_cause|complication|unresolved
    ontology_reasoning: str = ""         # human-readable explanation from OLS/DO
    resolution_layer: int = 0            # 1-4: which cascade layer resolved the relationship


@dataclass
class TrustGovernanceOutput(BaseAgentOutput):
    ocr_trust_score: float = 0.0
    external_trust_score: float = 0.0
    fraud_confidence_score: float = 0.0
    overall_trust_score: float = 0.0
    human_review_required: bool = False
    trust_breakdown: Dict[str, Any] = field(default_factory=dict)
    automation_decision: str = "HOLD"
    automation_eligible: bool = False                               # agent6
    trust_reduction_reasons: List[str] = field(default_factory=list)  # agent6
    executive_summary: str = ""                                       # LLM generated

    def __post_init__(self):
        # Derive automation_decision from automation_eligible if not set
        if not self.automation_decision or self.automation_decision == "HOLD":
            if self.automation_eligible and not self.human_review_required:
                self.automation_decision = "APPROVE"
            elif self.human_review_required:
                self.automation_decision = "ESCALATE"


@dataclass
class GraphAnalysisOutput(BaseAgentOutput):
    connected_claims: int = 0
    shared_nominees: int = 0
    shared_bank_accounts: int = 0
    shared_hospitals: int = 0
    shared_entities: Dict[str, int] = field(default_factory=dict)  # agent7
    high_risk_relationships: List[str] = field(default_factory=list)
    fraud_ring_detected: bool = False
    network_risk_score: float = 0.0
    graph_risk_score: float = 0.0                                  # agent7

    def __post_init__(self):
        # Keep both fields in sync
        if self.graph_risk_score and not self.network_risk_score:
            self.network_risk_score = self.graph_risk_score
        elif self.network_risk_score and not self.graph_risk_score:
            self.graph_risk_score = self.network_risk_score


@dataclass
class FraudIntelligencePackage:
    claim_case_id: str = ""
    fraud_analysis: Optional[Any] = None
    trust_analysis: Optional[Any] = None
    external_verification: Optional[Any] = None
    early_claim_analysis: Optional[Any] = None
    non_disclosure_analysis: Optional[Any] = None
    conflict_resolution: Optional[Any] = None
    graph_analysis: Optional[Any] = None
    final_recommendation: str = "PENDING"
    escalation_required: bool = False
    pipeline_duration_ms: Optional[float] = None
    created_at: str = field(default_factory=_now)

    def model_dump(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
