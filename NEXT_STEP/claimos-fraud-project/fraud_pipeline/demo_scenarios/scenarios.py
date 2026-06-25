"""
Demo Scenarios — 4 realistic ClaimState inputs for testing the pipeline.

Demo 1: Fraud Ring          — shared nominee + shared bank account
Demo 2: Early Claim         — policy issued 58 days before death
Demo 3: Non-Disclosure      — proposal says "no smoking", PMR shows lung disease
Demo 4: Trust Reduction     — blurry document, OCR confidence = 0.41
"""

from ..schemas.base import (
    ClaimState, Claimant, LifeAssured, DeathInformation,
    SubmittedDocument, MedicalRecord, FIRRecord,
)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 1 — FRAUD RING
# Same nominee (NOM-001) and same bank account (ACC-9901) already in graph
# linked to 2 previous claims. Should trigger fraud ring detection.
# ─────────────────────────────────────────────────────────────────────────────
DEMO_1_FRAUD_RING = ClaimState(
    claim_case_id="CLM-DEMO-001",
    policy_number="POL-2023-8871",
    policy_issue_date="2021-03-15",
    policy_age_days=900,

    claimant=Claimant(
        name="Rajan Mehta",
        relationship_to_life_assured="Brother",
        nominee_id="NOM-001",                     # REPEAT nominee in graph!
        bank_account_number="ACC-9901",            # REPEAT account in graph!
        bank_name="State Bank of India",
        contact_number="9876543210",
        address="12 Gandhi Nagar, Mumbai",
    ),
    life_assured=LifeAssured(
        name="Suresh Mehta",
        dob="1978-04-20",
        age_at_death=45,
        policy_number="POL-2023-8871",
        sum_assured=5_000_000,
        occupation="Driver",
    ),
    death_information=DeathInformation(
        date_of_death="2023-09-12",
        cause_of_death="Cardiac arrest",
        place_of_death="Mumbai",
        hospital_name="RURAL HEALTH CLINIC 7",    # suspicious hospital
        attending_doctor="DR. ANON",
        manner_of_death="natural",
    ),
    submitted_documents=[
        SubmittedDocument(
            document_type="death_certificate",
            ocr_confidence=0.85,
            ocr_text="Death Certificate — Suresh Mehta — 12-09-2023 — Cardiac Arrest",
            metadata={"creation_date": "2023-09-13", "dpi": 300},
        ),
    ],
    medical_records=[
        MedicalRecord(
            record_type="discharge_summary",
            hospital_name="RURAL HEALTH CLINIC 7",
            diagnosis="Cardiac arrest",
            treating_doctor="Dr. Anon",
            content_summary="Patient brought dead on arrival. Cardiac arrest.",
        ),
    ],
    ocr_confidence_scores={"death_certificate": 0.85},
    proposal_smoking=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2 — EARLY CLAIM
# Policy issued only 58 days before death.
# ─────────────────────────────────────────────────────────────────────────────
DEMO_2_EARLY_CLAIM = ClaimState(
    claim_case_id="CLM-DEMO-002",
    policy_number="POL-2024-0012",
    policy_issue_date="2024-01-05",
    policy_age_days=58,
    policy_revival_detected=True,

    claimant=Claimant(
        name="Priya Sharma",
        relationship_to_life_assured="Wife",
        nominee_id="NOM-555",
        bank_account_number="ACC-3344",
        bank_name="HDFC Bank",
        contact_number="9823456789",
        address="45 Rose Garden, Delhi",
    ),
    life_assured=LifeAssured(
        name="Anil Sharma",
        dob="1980-07-14",
        age_at_death=43,
        policy_number="POL-2024-0012",
        sum_assured=10_000_000,
        occupation="Businessman",
    ),
    death_information=DeathInformation(
        date_of_death="2024-03-04",
        cause_of_death="Road accident",
        place_of_death="Delhi",
        hospital_name="APOLLO HOSPITAL",
        attending_doctor="DR. RAJESH SHARMA",
        manner_of_death="accidental",
        fir_number="FIR-2024-789",
    ),
    submitted_documents=[
        SubmittedDocument(
            document_type="death_certificate",
            ocr_confidence=0.88,
            ocr_text="Death Certificate — Anil Sharma — 04-03-2024 — Road accident",
            metadata={"creation_date": "2024-03-05", "dpi": 300},
        ),
        SubmittedDocument(
            document_type="fir",
            ocr_confidence=0.82,
            ocr_text="FIR-2024-789 — Road accident — 04-03-2024",
            metadata={"creation_date": "2024-03-04", "dpi": 300},
        ),
    ],
    fir_records=[
        FIRRecord(
            fir_number="FIR-2024-789",
            police_station="Connaught Place PS",
            date="2024-03-04",
            incident_description="Road accident on NH-48. Victim Anil Sharma.",
            investigating_officer="SI Rajendra",
            verified=True,
        ),
    ],
    medical_records=[
        MedicalRecord(
            record_type="emergency",
            hospital_name="APOLLO HOSPITAL",
            date="2024-03-04",
            diagnosis="Multiple trauma injuries due to road accident",
            treating_doctor="Dr. Rajesh Sharma",
            content_summary="Patient brought via ambulance after road accident. Succumbed to injuries.",
        ),
    ],
    ocr_confidence_scores={"death_certificate": 0.88, "fir": 0.82},
    proposal_smoking=False,
    proposal_pre_existing_conditions=[],
)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3 — NON-DISCLOSURE
# Proposal: No smoking. Medical record: Chronic obstructive pulmonary disease
# ─────────────────────────────────────────────────────────────────────────────
DEMO_3_NON_DISCLOSURE = ClaimState(
    claim_case_id="CLM-DEMO-003",
    policy_number="POL-2020-5532",
    policy_issue_date="2020-06-01",
    policy_age_days=1195,

    claimant=Claimant(
        name="Sunita Verma",
        relationship_to_life_assured="Wife",
        nominee_id="NOM-800",
        bank_account_number="ACC-6677",
        bank_name="Punjab National Bank",
        contact_number="9011223344",
        address="78 Nehru Colony, Lucknow",
    ),
    life_assured=LifeAssured(
        name="Ramesh Verma",
        dob="1965-11-22",
        age_at_death=57,
        policy_number="POL-2020-5532",
        sum_assured=2_000_000,
        occupation="Factory Worker",
        smoking_history=True,           # actual — but not disclosed
    ),
    death_information=DeathInformation(
        date_of_death="2023-10-10",
        cause_of_death="Respiratory failure",
        place_of_death="Lucknow",
        hospital_name="CITY GENERAL HOSPITAL",
        attending_doctor="DR. PRIYA MEHTA",
        manner_of_death="natural",
    ),
    submitted_documents=[
        SubmittedDocument(
            document_type="death_certificate",
            ocr_confidence=0.87,
            ocr_text="Death Certificate — Ramesh Verma — 10-10-2023 — Respiratory failure",
            metadata={"creation_date": "2023-10-11", "dpi": 300},
        ),
        SubmittedDocument(
            document_type="pmr",
            ocr_confidence=0.83,
            ocr_text="Post Mortem Report — Chronic obstructive pulmonary disease (COPD) secondary to chronic smoking. Lung cancer stage III.",
            metadata={"creation_date": "2023-10-12", "dpi": 300},
        ),
    ],
    medical_records=[
        MedicalRecord(
            record_type="pmr",
            hospital_name="CITY GENERAL HOSPITAL",
            date="2023-10-11",
            diagnosis="Chronic obstructive pulmonary disease (COPD), lung cancer stage III, chronic smoking",
            treating_doctor="DR. PRIYA MEHTA",
            content_summary=(
                "Autopsy confirms COPD secondary to 25+ years of heavy smoking. "
                "Lung cancer stage III consistent with long-term tobacco use."
            ),
            smoking_history=True,
        ),
    ],
    # ── KEY: proposal declared NO smoking ──
    proposal_smoking=False,
    proposal_alcohol_use=False,
    proposal_pre_existing_conditions=[],
    ocr_confidence_scores={"death_certificate": 0.87, "pmr": 0.83},
)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4 — TRUST REDUCTION / BLURRY DOCUMENT
# OCR confidence on PMR = 0.41 → trust degraded → human review triggered
# ─────────────────────────────────────────────────────────────────────────────
DEMO_4_TRUST_REDUCTION = ClaimState(
    claim_case_id="CLM-DEMO-004",
    policy_number="POL-2019-2210",
    policy_issue_date="2019-01-01",
    policy_age_days=1765,

    claimant=Claimant(
        name="Kavitha Rajan",
        relationship_to_life_assured="Daughter",
        nominee_id="NOM-200",
        bank_account_number="ACC-1122",
        bank_name="Canara Bank",
        contact_number="9955667788",
        address="34 MG Road, Chennai",
    ),
    life_assured=LifeAssured(
        name="Krishnamurthy Rajan",
        dob="1950-03-08",
        age_at_death=73,
        policy_number="POL-2019-2210",
        sum_assured=1_500_000,
        occupation="Retired",
    ),
    death_information=DeathInformation(
        date_of_death="2023-11-20",
        cause_of_death="Heart failure",
        place_of_death="Chennai",
        hospital_name="SUNRISE MEDICAL CENTER",
        attending_doctor="DR. PRIYA MEHTA",
        manner_of_death="natural",
    ),
    submitted_documents=[
        SubmittedDocument(
            document_type="death_certificate",
            ocr_confidence=0.85,
            ocr_text="Death Certificate — Krishnamurthy Rajan — 20-11-2023 — Heart failure",
            metadata={"creation_date": "2023-11-21", "dpi": 300},
        ),
        SubmittedDocument(
            document_type="pmr",
            ocr_confidence=0.41,              # BLURRY / LOW QUALITY
            ocr_text="[ILLEGIBLE — low quality scan] Post mortem... heart... failure...",
            metadata={
                "creation_date": "2023-11-22",
                "dpi": 60,                    # very low DPI — blurry
                "scan_quality": "poor",
            },
        ),
    ],
    medical_records=[
        MedicalRecord(
            record_type="pmr",
            hospital_name="SUNRISE MEDICAL CENTER",
            date="2023-11-21",
            diagnosis="Heart failure — elderly patient",
            treating_doctor="DR. PRIYA MEHTA",
            content_summary="Elderly patient, 73 years, heart failure.",
        ),
    ],
    ocr_confidence_scores={
        "death_certificate": 0.85,
        "pmr": 0.41,                           # This will trigger trust degradation
    },
    proposal_smoking=False,
    proposal_pre_existing_conditions=[],
)


ALL_DEMOS = {
    "demo1_fraud_ring":       DEMO_1_FRAUD_RING,
    "demo2_early_claim":      DEMO_2_EARLY_CLAIM,
    "demo3_non_disclosure":   DEMO_3_NON_DISCLOSURE,
    "demo4_trust_reduction":  DEMO_4_TRUST_REDUCTION,
}
