"""
Base schemas — Pydantic v2 models.
All nested objects use typed models instead of raw dicts.
Agents access fields via attribute access (state.claimant.name etc.)
"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, AliasChoices
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return f"CLM-{uuid.uuid4().hex[:8].upper()}"


# ── Nested Input Models ────────────────────────────────────────────────────────

class Claimant(BaseModel):
    name: str = ""
    relationship: str = ""
    relationship_to_life_assured: str = ""  # alias
    contact: str = ""
    contact_number: str = ""                # alias
    bank_account: str = ""
    bank_account_number: str = ""           # alias
    bank_name: str = ""
    bank_ifsc: str = ""
    nominee_id: str = ""
    address: str = ""

    model_config = {"extra": "allow"}

    def model_post_init(self, __context: Any) -> None:
        if self.relationship_to_life_assured and not self.relationship:
            self.relationship = self.relationship_to_life_assured
        if self.bank_account_number and not self.bank_account:
            self.bank_account = self.bank_account_number
        if self.contact_number and not self.contact:
            self.contact = self.contact_number


class LifeAssured(BaseModel):
    name: str = ""
    age: int = 0
    age_at_death: int = 0
    gender: str = ""
    dob: str = ""
    occupation: str = ""
    policy_number: str = ""
    sum_assured: float = 0.0
    smoking_history: bool = False
    alcohol_history: bool = False
    medical_history: List[str] = []
    pre_existing_conditions: List[str] = []

    model_config = {"extra": "allow"}


class DeathInformation(BaseModel):
    date_of_death: str = ""
    cause_of_death: str = ""
    place_of_death: str = ""
    hospital_name: str = ""
    hospital_id: str = ""
    hospital_address: str = ""
    hospital_rohini_id: Optional[str] = None       # NEW: validated by Agent 1 (format: HH-NNNNN)
    doctor_name: str = ""
    attending_doctor: str = ""              # alias
    doctor_registration_number: Optional[str] = None  # NEW: NMR reg number, validated by Agent 1
    fir_number: str = ""
    death_certificate_number: str = ""
    manner_of_death: str = ""

    model_config = {"extra": "allow"}

    def model_post_init(self, __context: Any) -> None:
        if self.attending_doctor and not self.doctor_name:
            self.doctor_name = self.attending_doctor


class SubmittedDocument(BaseModel):
    doc_type: str = ""
    document_type: str = ""                 # alias
    file_path: str = ""
    ocr_confidence: float = 0.0
    ocr_text: str = ""
    metadata: Dict[str, Any] = {}
    upload_timestamp: str = ""
    sha256_hash: Optional[str] = None          # NEW: computed by OCR layer; Agent 1 checks registry
    hospital_rohini_id: Optional[str] = None   # NEW: extracted by OCR if present in doc
    doctor_registration_number: Optional[str] = None  # NEW: extracted by OCR if present

    model_config = {"extra": "allow"}

    def model_post_init(self, __context: Any) -> None:
        if self.document_type and not self.doc_type:
            self.doc_type = self.document_type


class MedicalRecord(BaseModel):
    record_type: str = ""
    hospital_name: str = ""
    doctor_name: str = ""
    treating_doctor: str = ""              # alias
    diagnosis: str = ""
    treatment: str = ""
    content_summary: str = ""
    admission_date: str = ""
    discharge_date: str = ""
    date: str = ""
    ocr_text: str = ""
    smoking_history: bool = False
    alcohol_history: bool = False
    chronic_conditions: List[str] = []

    model_config = {"extra": "allow"}

    def model_post_init(self, __context: Any) -> None:
        if self.treating_doctor and not self.doctor_name:
            self.doctor_name = self.treating_doctor


class FIRRecord(BaseModel):
    fir_number: str = ""
    police_station: str = ""
    date_filed: str = ""
    date: str = ""
    description: str = ""
    incident_description: str = ""
    location: str = ""
    investigating_officer: str = ""
    verified: bool = False
    accused_names: List[str] = []

    model_config = {"extra": "allow"}


# ── NEW: Nested Models for expanded ClaimState ─────────────────────────────────

class DocumentQuality(BaseModel):
    """Quality assessment of submitted documents."""
    is_blurry: bool = False
    is_handwritten: bool = False
    is_scanned: bool = False
    is_digitally_altered: bool = False
    resolution_dpi: Optional[int] = None
    overall_quality_score: float = 1.0      # 0.0 (poor) to 1.0 (perfect)
    quality_issues: List[str] = []

    model_config = {"extra": "allow"}


class Contradiction(BaseModel):
    """A contradiction detected between two document sources."""
    source_1: str                           # e.g. "death_certificate"
    source_2: str                           # e.g. "fir_record"
    field: str = ""                         # e.g. "date_of_death"
    value_1: str                            # value from source_1
    value_2: str                            # value from source_2
    severity: str = "MEDIUM"               # LOW / MEDIUM / HIGH
    resolved: bool = False
    resolution_note: str = ""

    model_config = {"extra": "allow"}


class HistoricalClaimLink(BaseModel):
    """A link to a previous claim associated with this claimant/nominee."""
    linked_claim_id: str = ""              # canonical field name
    claim_id: str = ""                     # alias — accepted for backward compat
    policy_number: str = ""
    claimant_name: str = ""
    nominee_id: str = ""
    bank_account: str = ""
    hospital_name: str = ""
    date_of_death: str = ""
    final_recommendation: str = ""         # outcome of that claim
    relationship: str = ""                 # how it's linked (same_nominee / same_bank / etc.)

    model_config = {"extra": "allow"}

    def model_post_init(self, __context: Any) -> None:
        # Sync alias: if caller used 'claim_id', copy it to linked_claim_id
        if self.claim_id and not self.linked_claim_id:
            self.linked_claim_id = self.claim_id
        elif self.linked_claim_id and not self.claim_id:
            self.claim_id = self.linked_claim_id


class GraphRelationship(BaseModel):
    """A relationship edge in the fraud evidence graph."""
    entity_type: str                        # CLAIM / NOMINEE / BANK / HOSPITAL / DOCTOR
    entity_id: str
    relationship_type: str                  # SAME_NOMINEE / SAME_BANK / SAME_HOSPITAL
    connected_claim_ids: List[str] = []
    risk_score: float = 0.0
    is_suspicious: bool = False

    model_config = {"extra": "allow"}


class ProposalForm(BaseModel):
    """Structured proposal form disclosures at the time of policy purchase."""
    in_good_health: Optional[bool] = None
    smokes: Optional[bool] = None
    alcohol_use: Optional[bool] = None
    pre_existing_conditions: List[str] = []
    family_history: List[str] = []
    occupation_hazardous: Optional[bool] = None
    mental_health_history: Optional[bool] = None   # True = disclosed; False = denied
    hazardous_activities: Optional[bool] = None    # True = disclosed; False = denied
    annual_income: Optional[float] = None
    sum_assured_reason: str = ""
    additional_disclosures: Dict[str, Any] = {}

    model_config = {"extra": "allow"}


# ── ClaimState — Main Pipeline State ──────────────────────────────────────────

class ClaimState(BaseModel):
    """
    Central state object passed through all 7 agents.
    Each agent reads from it and writes its output back into it.
    """
    # ── Identifiers ───────────────────────────────────────────────────────────
    claim_case_id: str = Field(default_factory=_new_id)
    policy_number: str = ""

    # ── Policy Details ────────────────────────────────────────────────────────
    policy_issue_date: str = Field("", validation_alias=AliasChoices("policy_issue_date", "policy_start_date"))
    policy_age_days: int = 0
    policy_revival_detected: bool = False
    policy_revival_date: str = ""
    policy_sum_assured: float = 0.0
    policy_premium: float = 0.0
    last_premium_paid_date: str = ""
    premium_payment_history: List[Dict[str, Any]] = Field(default_factory=list, validation_alias=AliasChoices("premium_payment_history", "premium_payments"))

    # ── Core Nested Objects ───────────────────────────────────────────────────
    claimant: Claimant = Field(default_factory=Claimant)
    life_assured: LifeAssured = Field(default_factory=LifeAssured)
    death_information: DeathInformation = Field(default_factory=DeathInformation)

    # ── Documents & Records ───────────────────────────────────────────────────
    submitted_documents: List[SubmittedDocument] = []
    medical_records: List[MedicalRecord] = []
    fir_records: List[FIRRecord] = []
    ocr_confidence_scores: Dict[str, float] = {}

    # ── Proposal shortcut fields (top-level for backward compatibility) ──────
    proposal_smoking: Optional[bool] = None
    proposal_alcohol_use: Optional[bool] = None
    proposal_pre_existing_conditions: List[str] = []

    # ── NEW: Proposal Form (structured) ──────────────────────────────────────
    proposal_form: ProposalForm = Field(default_factory=ProposalForm)



    # ── NEW: Document Quality ─────────────────────────────────────────────────
    document_quality: Optional[DocumentQuality] = None

    # ── NEW: Contradictions detected across documents ─────────────────────────
    contradictions: List[Contradiction] = []

    # ── NEW: Historical claims linked to same claimant/nominee/bank ───────────
    historical_claim_links: List[HistoricalClaimLink] = []

    # ── NEW: Graph relationships for fraud ring detection ─────────────────────
    graph_relationships: List[GraphRelationship] = []

    # ── Validation & Flags ────────────────────────────────────────────────────
    validation_flags: List[str] = []

    # ── Agent Outputs (populated during pipeline run) ─────────────────────────
    external_verification: Optional[Dict[str, Any]] = None
    fraud_analysis: Optional[Dict[str, Any]] = None
    early_claim_analysis: Optional[Dict[str, Any]] = None
    non_disclosure_analysis: Optional[Dict[str, Any]] = None
    conflict_resolution: Optional[Dict[str, Any]] = None
    trust_analysis: Optional[Dict[str, Any]] = None
    graph_analysis: Optional[Dict[str, Any]] = None
    final_recommendation: str = ""
    escalation_required: bool = False
    
    # ── LLM Outputs ───────────────────────────────────────────────────────────
    anomaly_explanation: Optional[str] = None

    model_config = {"extra": "allow"}

    def model_post_init(self, __context: Any) -> None:
        """Sync top-level shortcut fields into the structured ProposalForm."""
        if self.proposal_smoking is not None and self.proposal_form.smokes is None:
            self.proposal_form.smokes = self.proposal_smoking
        if self.proposal_alcohol_use is not None and self.proposal_form.alcohol_use is None:
            self.proposal_form.alcohol_use = self.proposal_alcohol_use
        if self.proposal_pre_existing_conditions and not self.proposal_form.pre_existing_conditions:
            self.proposal_form.pre_existing_conditions = self.proposal_pre_existing_conditions

    @field_validator("claimant", mode="before")
    @classmethod
    def _coerce_claimant(cls, v):
        if isinstance(v, dict):
            return Claimant(**v)
        return v

    @field_validator("life_assured", mode="before")
    @classmethod
    def _coerce_life_assured(cls, v):
        if isinstance(v, dict):
            return LifeAssured(**v)
        return v

    @field_validator("death_information", mode="before")
    @classmethod
    def _coerce_death_information(cls, v):
        if isinstance(v, dict):
            return DeathInformation(**v)
        return v

    @field_validator("submitted_documents", mode="before")
    @classmethod
    def _coerce_documents(cls, v):
        if isinstance(v, list):
            return [SubmittedDocument(**i) if isinstance(i, dict) else i for i in v]
        return v

    @field_validator("medical_records", mode="before")
    @classmethod
    def _coerce_medical_records(cls, v):
        if isinstance(v, list):
            return [MedicalRecord(**i) if isinstance(i, dict) else i for i in v]
        return v

    @field_validator("fir_records", mode="before")
    @classmethod
    def _coerce_fir_records(cls, v):
        if isinstance(v, list):
            return [FIRRecord(**i) if isinstance(i, dict) else i for i in v]
        return v

    @field_validator("proposal_form", mode="before")
    @classmethod
    def _coerce_proposal_form(cls, v):
        if isinstance(v, dict):
            return ProposalForm(**v)
        return v

    @field_validator("document_quality", mode="before")
    @classmethod
    def _coerce_document_quality(cls, v):
        if isinstance(v, dict):
            return DocumentQuality(**v)
        return v

    @field_validator("contradictions", mode="before")
    @classmethod
    def _coerce_contradictions(cls, v):
        if isinstance(v, list):
            return [Contradiction(**i) if isinstance(i, dict) else i for i in v]
        return v

    @field_validator("historical_claim_links", mode="before")
    @classmethod
    def _coerce_historical_links(cls, v):
        if isinstance(v, list):
            return [HistoricalClaimLink(**i) if isinstance(i, dict) else i for i in v]
        return v

    @field_validator("graph_relationships", mode="before")
    @classmethod
    def _coerce_graph_relationships(cls, v):
        if isinstance(v, list):
            return [GraphRelationship(**i) if isinstance(i, dict) else i for i in v]
        return v

    @classmethod
    def from_dict(cls, d: dict) -> "ClaimState":
        """Convenience constructor — same as cls(**d) but explicit."""
        return cls(**d)
