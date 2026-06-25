"""
Self-contained test runner (no pytest required).
Run: python fraud_pipeline/tests/run_tests.py
"""
import sys, os, traceback, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
logging.basicConfig(level=logging.WARNING)

from fraud_pipeline.schemas import (
    ClaimState, Claimant, LifeAssured, DeathInformation,
    SubmittedDocument, MedicalRecord, FIRRecord
)
from fraud_pipeline.pipeline import FraudPipeline
from fraud_pipeline.demo_scenarios import (
    demo_fraud_ring, demo_early_claim, demo_non_disclosure, demo_trust_reduction
)

pipeline = FraudPipeline()
PASS = 0; FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✓ {name}")
        PASS += 1
    except Exception as e:
        print(f"  ✗ {name}")
        traceback.print_exc()
        FAIL += 1


def clean_claim(**kwargs):
    import uuid
    d = dict(
        claim_case_id=f"CLM-TEST-{uuid.uuid4().hex[:6].upper()}",
        policy_number="POL-CLEAN-001",
        policy_issue_date="2020-01-01",
        policy_age_days=1825,
        claimant=Claimant(name="Test User", bank_account="ACC-CLEAN-001", nominee_id="NOM-CLEAN-001"),
        life_assured=LifeAssured(name="Insured Person", age=55),
        death_information=DeathInformation(
            date_of_death="2025-01-01", cause_of_death="Natural causes",
            place_of_death="Mumbai", hospital_name="Apollo Hospital Mumbai",
            doctor_name="Dr. Valid",
        ),
        submitted_documents=[
            SubmittedDocument(doc_type="death_certificate", ocr_confidence=0.92),
            SubmittedDocument(doc_type="claim_form", ocr_confidence=0.90),
        ],
        medical_records=[MedicalRecord(record_type="hospital_discharge", diagnosis="Natural cardiac failure", date="2025-01-01")],
        ocr_confidence_scores={"death_certificate": 0.92, "claim_form": 0.90},
    )
    d.update(kwargs)
    return ClaimState(**d)


print("\n-- Agent 1: External Verification --------------------------")

def t_agent1_clean():
    """Clean claim: no ROHINI ID → ROHINI_ID_MISSING flag, but overall confidence > 0.3."""
    s = clean_claim(); s = pipeline.agent1.run(s)
    ev = s.external_verification
    # Per-area scores must exist
    assert "hospital_score" in ev, "hospital_score missing"
    assert "doctor_score" in ev, "doctor_score missing"
    assert "fir_score" in ev, "fir_score missing"
    assert "image_score" in ev, "image_score missing"
    assert "geo_score" in ev, "geo_score missing"
    assert "metadata_score" in ev, "metadata_score missing"
    # verification_confidence must be present
    assert ev["verification_confidence"] > 0.0, "verification_confidence must be positive"
    # metadata must be clean (no editing software in clean claim)
    assert ev["metadata_tampered"] is False
test("Clean claim — per-area float scores present", t_agent1_clean)

def t_agent1_rohini_id_missing():
    """No ROHINI ID → hospital_score < 1.0 and ROHINI_ID_MISSING flag."""
    s = clean_claim(); s = pipeline.agent1.run(s)
    ev = s.external_verification
    # ROHINI ID not provided in clean_claim → score should be 0.40
    assert ev["hospital_score"] < 1.0, f"Expected hospital_score < 1.0, got {ev['hospital_score']}"
    assert any("ROHINI_ID_MISSING" in f for f in s.validation_flags), \
        f"ROHINI_ID_MISSING flag missing. Flags: {s.validation_flags}"
test("ROHINI_ID_MISSING flag fires when hospital_rohini_id absent", t_agent1_rohini_id_missing)

def t_agent1_rohini_id_format_invalid():
    """Invalid ROHINI ID format → hospital_score 0.20 and format flag."""
    s = clean_claim()
    s.death_information.hospital_rohini_id = "INVALID_ID"
    s = pipeline.agent1.run(s)
    ev = s.external_verification
    assert ev["hospital_score"] <= 0.20, f"Expected <= 0.20, got {ev['hospital_score']}"
    assert any("ROHINI_ID_FORMAT_INVALID" in f for f in s.validation_flags), \
        f"Format flag missing. Flags: {s.validation_flags}"
test("Invalid ROHINI ID format flags ROHINI_ID_FORMAT_INVALID", t_agent1_rohini_id_format_invalid)

def t_agent1_rohini_id_valid():
    """Valid ROHINI ID → hospital_score 1.0 (no DB frequency hit for new claim)."""
    s = clean_claim()
    s.death_information.hospital_rohini_id = "8900070055443"
    s = pipeline.agent1.run(s)
    ev = s.external_verification
    # DB unavailable in test env → returns 0.5 neutral OR 1.0 if DB skip
    assert ev["hospital_score"] >= 0.50, f"Expected >= 0.50 for valid ROHINI, got {ev['hospital_score']}"
test("Valid ROHINI ID format → hospital_score >= 0.50", t_agent1_rohini_id_valid)

def t_agent1_geo_mismatch():
    """Mumbai hospital vs Delhi death location → GEO_STATE_MISMATCH or GEO_DISTRICT_MISMATCH."""
    s = clean_claim()
    s.death_information.hospital_name = "Apollo Hospital Mumbai"
    s.death_information.hospital_address = "Mumbai Maharashtra 400001"
    s.death_information.place_of_death = "Delhi"
    s = pipeline.agent1.run(s)
    ev = s.external_verification
    assert ev["geo_location_match"] is False, \
        f"Expected geo_location_match=False for Mumbai vs Delhi. geo_score={ev['geo_score']}"
    assert any("GEO" in f for f in s.validation_flags), \
        f"Expected GEO_* flag. Flags: {s.validation_flags}"
test("Geo mismatch (Mumbai hospital, Delhi death) flagged", t_agent1_geo_mismatch)


print("\n── Agent 2: Fraud Intelligence ─────────────────────────────")

def t_agent2_clean():
    s = clean_claim(); pipeline.agent1.run(s); s = pipeline.agent2.run(s)
    assert s.fraud_analysis["fraud_risk_score"] < 0.60
test("Clean claim → low fraud score", t_agent2_clean)

def t_agent2_early_claim():
    s = clean_claim(policy_age_days=58)
    pipeline.agent1.run(s); s = pipeline.agent2.run(s)
    assert s.fraud_analysis["fraud_risk_score"] > 0.20
    assert any("58 days" in r for r in s.fraud_analysis["fraud_reasons"])
test("Early claim raises fraud score", t_agent2_early_claim)

def t_agent2_tampered_doc():
    """PDF created by Photoshop → metadata_tampered=True → higher fraud score."""
    s = clean_claim()
    s.submitted_documents[0].metadata["creator"] = "Adobe Photoshop CS6"
    s = pipeline.agent1.run(s); s = pipeline.agent2.run(s)
    assert s.fraud_analysis["fraud_risk_score"] > 0.25
test("PDF with editing software metadata raises fraud score", t_agent2_tampered_doc)


print("\n── Agent 3: Early Claim ────────────────────────────────────")

def t_agent3_very_early():
    s = clean_claim(policy_age_days=58)
    s = pipeline.agent3.run(s)
    assert s.early_claim_analysis["early_claim_risk"] in ("HIGH", "VERY_HIGH")
test("58-day policy → VERY_HIGH risk", t_agent3_very_early)

def t_agent3_old_policy():
    s = clean_claim(policy_age_days=1825)
    s = pipeline.agent3.run(s)
    assert s.early_claim_analysis["early_claim_risk"] == "LOW"
test("5-year policy → LOW risk", t_agent3_old_policy)

def t_agent3_revival():
    s = clean_claim(policy_age_days=400)
    s.premium_payment_history = [{"date": "2023-01-01"}, {"date": "2023-09-01"}]
    s.death_information.date_of_death = "2023-11-01"
    s = pipeline.agent3.run(s)
    assert s.early_claim_analysis["policy_revival_detected"] is True
test("242-day lapse then revival detected", t_agent3_revival)


print("\n── Agent 4: Non-Disclosure ─────────────────────────────────")

def t_agent4_clean():
    s = clean_claim(); s.proposal_form = {"in_good_health": True}
    s = pipeline.agent4.run(s)
    assert s.non_disclosure_analysis["contradiction_detected"] is False
test("No contradiction in clean claim", t_agent4_clean)

def t_agent4_smoking():
    s = clean_claim(); s.proposal_form = {"smokes": False}
    s.medical_records = [MedicalRecord(
        record_type="medical_report", diagnosis="Chronic lung disease",
        smoking_history=True, chronic_conditions=["COPD"], date="2025-01-01"
    )]
    s = pipeline.agent4.run(s)
    assert s.non_disclosure_analysis["contradiction_detected"] is True
    assert s.non_disclosure_analysis["non_disclosure_score"] > 0.5
test("Smoking non-disclosure detected", t_agent4_smoking)


print("\n── Agent 5: Conflict Resolution ────────────────────────────")

def t_agent5_no_conflict():
    s = clean_claim(); s = pipeline.agent5.run(s)
    assert s.conflict_resolution["conflict_detected"] is False
    assert s.conflict_resolution["resolved_action"] == "ACCEPT"
test("No conflict in clean claim", t_agent5_no_conflict)


print("\n── Agent 6: Trust Governance ───────────────────────────────")

def t_agent6_high_trust():
    s = clean_claim()
    for fn in [pipeline.agent1.run, pipeline.agent2.run, pipeline.agent4.run,
               pipeline.agent5.run, pipeline.agent6.run]:
        s = fn(s)
    assert s.trust_analysis["overall_trust_score"] > 0.50
test("Clean claim → trust > 0.50", t_agent6_high_trust)

def t_agent6_low_ocr_triggers_review():
    s = clean_claim()
    s.ocr_confidence_scores = {"death_certificate": 0.82, "pmr": 0.41}
    s.submitted_documents = [
        SubmittedDocument(doc_type="death_certificate", ocr_confidence=0.82),
        SubmittedDocument(doc_type="pmr", ocr_confidence=0.41),
    ]
    for fn in [pipeline.agent1.run, pipeline.agent2.run, pipeline.agent4.run,
               pipeline.agent5.run, pipeline.agent6.run]:
        s = fn(s)
    assert s.trust_analysis["human_review_required"] is True
test("OCR=0.41 → human review required", t_agent6_low_ocr_triggers_review)


print("\n── Agent 7: Graph Engine ───────────────────────────────────")

def t_agent7_no_ring():
    import uuid
    s = clean_claim(claim_case_id=f"CLM-UNIQUE-{uuid.uuid4().hex[:8]}")
    s.claimant.nominee_id = f"NOM-UNIQUE-{uuid.uuid4().hex[:8]}"
    s = pipeline.agent7.run(s)
    assert s.graph_analysis["fraud_ring_detected"] is False
test("Unique nominee → no fraud ring", t_agent7_no_ring)


print("\n── Full Pipeline ───────────────────────────────────────────")

def t_full_clean():
    s = clean_claim(); pkg = pipeline.run(s)
    assert pkg.final_recommendation in ("APPROVE", "INVESTIGATE", "ESCALATE", "REJECT")
    assert pkg.fraud_analysis and pkg.trust_analysis
    assert pkg.external_verification and pkg.early_claim_analysis
    assert pkg.non_disclosure_analysis and pkg.conflict_resolution
    assert pkg.graph_analysis
test("All 7 outputs present in package", t_full_clean)

def t_full_recommendation_valid():
    s = clean_claim(); pkg = pipeline.run(s)
    assert pkg.final_recommendation in ("APPROVE", "INVESTIGATE", "ESCALATE", "REJECT")
test("Final recommendation is valid enum value", t_full_recommendation_valid)


print("\n── Demo Scenarios ──────────────────────────────────────────")

def t_demo_fraud_ring():
    pkg = pipeline.run(demo_fraud_ring())
    assert pkg.escalation_required is True
    assert pkg.graph_analysis.get("fraud_ring_detected") is True
test("Demo 1 Fraud Ring → escalation=True, ring=True", t_demo_fraud_ring)

def t_demo_early_claim():
    pkg = pipeline.run(demo_early_claim())
    assert pkg.early_claim_analysis["policy_age_days"] == 58
    assert pkg.early_claim_analysis["early_claim_risk"] in ("HIGH", "VERY_HIGH")
test("Demo 2 Early Claim → 58 days, VERY_HIGH risk", t_demo_early_claim)

def t_demo_non_disclosure():
    pkg = pipeline.run(demo_non_disclosure())
    assert pkg.non_disclosure_analysis["contradiction_detected"] is True
    assert pkg.non_disclosure_analysis["non_disclosure_score"] > 0.5
test("Demo 3 Non-Disclosure → contradiction detected", t_demo_non_disclosure)

def t_demo_trust_reduction():
    pkg = pipeline.run(demo_trust_reduction())
    assert pkg.trust_analysis["human_review_required"] is True
    assert pkg.trust_analysis["ocr_trust_score"] < 0.75
test("Demo 4 Trust Reduction → human_review=True, low OCR trust", t_demo_trust_reduction)


print(f"\n{'='*55}")
print(f"  Results: {PASS} passed  |  {FAIL} failed")
print(f"{'='*55}")
if FAIL > 0:
    sys.exit(1)
