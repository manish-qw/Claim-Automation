"""
Schemas package — exports all Pydantic models for use as fraud_pipeline.schemas.*
"""
from .base import (
    ClaimState,
    Claimant,
    LifeAssured,
    DeathInformation,
    SubmittedDocument,
    MedicalRecord,
    FIRRecord,
    # New nested models
    DocumentQuality,
    Contradiction,
    HistoricalClaimLink,
    GraphRelationship,
    ProposalForm,
)
from .claim_state import (
    ExternalVerificationOutput,
    FraudAnalysisOutput,
    EarlyClaimOutput,
    NonDisclosureOutput,
    ConflictResolutionOutput,
    TrustGovernanceOutput,
    GraphAnalysisOutput,
    FraudIntelligencePackage,
    FraudIntelligenceOutput,  # alias
)

__all__ = [
    # Input models
    "ClaimState", "Claimant", "LifeAssured", "DeathInformation",
    "SubmittedDocument", "MedicalRecord", "FIRRecord",
    "DocumentQuality", "Contradiction", "HistoricalClaimLink",
    "GraphRelationship", "ProposalForm",
    # Output models
    "ExternalVerificationOutput", "FraudAnalysisOutput", "EarlyClaimOutput",
    "NonDisclosureOutput", "ConflictResolutionOutput",
    "TrustGovernanceOutput", "GraphAnalysisOutput",
    "FraudIntelligencePackage", "FraudIntelligenceOutput",
]
