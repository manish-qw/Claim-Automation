# api/validation_rules.py

REQUIRED_FIELDS = {
    # "CLAIMANT_STATEMENT_FORM": {
    #     "critical": [
    #         "Claimant.Name",
    #         "Claimant.DateOfBirth",
    #         "Claimant.MobileNumber",
    #         "Claimant.PAN",
    #         "LifeAssured.DateOfDeath",
    #         "LifeAssured.CauseOfDeath",
    #         "LifeAssured.PlaceOfDeath",
    #     ],
    #     "important": [
    #         "Claimant.Gender",
    #         "Claimant.HouseOrFlatNumber",
    #         "Claimant.Street",
    #         "Claimant.VillageTownCity",
    #         "Claimant.State",
    #         "Claimant.Pincode",
    #         "Claimant.EmailId",
    #         "Claimant.AlternateMobileNumber",
    #         "Claimant.IsPoliticallyExposedPerson",
    #         "LifeAssured.IntimationDate",
    #         "LifeAssured.PlaceOfIntimationOfDeath",
    #         "Witness.Name",
    #         "Witness.Relationship",
    #         "Witness.MobileNumber"
    #     ],
    #     "optional": [
    #         "Claimant.Occupation",
    #         "Hospital.MobileNumber",
    #         "Hospital.AlternateMobileNumber",
    #         "Hospital.EmailId",
    #         "TreatingDoctor.EmailId",
    #         "TreatingDoctor.LandlineNumber"
    #     ],
    #     "bank_critical": [
    #         "Bank.AccountNumber",
    #         "Bank.AccountHolderName",
    #         "Bank.BankName",
    #         "Bank.IFSCCode",
    #         "Bank.AccountType"
    #     ]
    # },
    # "DEATH_CERTIFICATE": {
    #     "critical": [
    #         "Deceased.DateOfDeath",
    #         "Deceased.CauseOfDeath",
    #         "Deceased.Name",
    #         "Registration.RegistrationNumber",
    #         "Registration.DateOfIssue"
    #     ],
    #     "important": [
    #         "Deceased.DateOfCremation",
    #         "DeathLocation.PlaceOfDeath",
    #         "IssuingAuthority.AuthorityName",
    #         "IssuingAuthority.IssuingOfficerPositionName",
    #         "IssuingAuthority.District",
    #         "IssuingAuthority.State"
    #     ]
    # },
    # "AADHAAR_CARD": {
    #     "critical": [
    #         "Identifiers.AadhaarNumber",
    #         "Person.FullName"
    #     ],
    #     "important": [
    #         "Person.DateOfBirth",
    #         "Person.Gender",
    #         "Address.FullAddress",
    #         "Address.Pincode",
    #         "Address.State"
    #     ],
    #     "front_only_indicators": [
    #         "Address.FullAddress",
    #         "Address.Pincode",
    #         "Address.District",
    #         "Address.State"
    #     ]
    # },
    # "DRIVING_LICENCE": {
    #     "critical": [
    #         "Identifiers.DrivingLicenceNumber",
    #         "Person.FullName",
    #         "Person.DateOfBirth",
    #         "Licence.ValidTo"
    #     ],
    #     "important": [
    #         "Family.FatherName",
    #         "Address.PermanentAddress",
    #         "Licence.IssuingRTO",
    #         "Licence.IssueDate"
    #     ]
    # },
    # "PASSPORT": {
    #     "critical": [
    #         "Passport.PassportNumber",
    #         "Person.FullName",
    #         "Person.DateOfBirth",
    #         "Passport.DateOfExpiry",
    #         "Person.Nationality"
    #     ],
    #     "important": [
    #         "TravelDocument.MRZLine1",
    #         "TravelDocument.MRZLine2",
    #         "Person.PlaceOfBirth",
    #         "Passport.DateOfIssue",
    #         "Passport.FileNumber"
    #     ]
    # },
    # "VOTER_ID": {
    #     "critical": [
    #         "Identifiers.EPICNumber",
    #         "Person.FullName"
    #     ],
    #     "important": [
    #         "Person.DateOfBirth",
    #         "Family.FatherName",
    #         "Address.FullAddress",
    #         "Address.State"
    #     ]
    # },
    # "PAN_CARD": {
    #     "critical": [
    #         "Identifiers.PANNumber",
    #         "Person.FullName"
    #     ],
    #     "important": [
    #         "Person.DateOfBirth",
    #         "Family.FatherName"
    #     ]
    # }
}

def get_nested_value(data, path):
    parts = path.split('.')
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current

def validate_document(doc_type, extracted_data):
    config = REQUIRED_FIELDS.get(doc_type)
    if not config:
        return {"status": "OK"}

    # Special rules for Aadhaar
    if doc_type == "AADHAAR_CARD":
        name_val = get_nested_value(extracted_data, "Person.FullName")
        name_present = name_val and name_val.get("value")
        
        all_front_null = True
        for f in config.get("front_only_indicators", []):
            val = get_nested_value(extracted_data, f)
            if val and val.get("value"):
                all_front_null = False
                break
                
        if name_present and all_front_null:
            return build_error("FRONT_SIDE_ONLY", doc_type, ["Address", "District", "State", "Pincode"], [])
            
    # Special rules for Passport
    if doc_type == "PASSPORT":
        pp_val = get_nested_value(extracted_data, "Passport.PassportNumber")
        pp_present = pp_val and pp_val.get("value")
        
        mrz1 = get_nested_value(extracted_data, "TravelDocument.MRZLine1")
        mrz2 = get_nested_value(extracted_data, "TravelDocument.MRZLine2")
        
        if pp_present and (not mrz1 or not mrz1.get("value")) and (not mrz2 or not mrz2.get("value")):
            return build_error("BACK_PAGE_MISSING", doc_type, ["MRZLine1", "MRZLine2"], [])

    missing_critical = []
    missing_important = []
    low_confidence = []

    for field in config.get("critical", []):
        val = get_nested_value(extracted_data, field) or {}
        if not val.get("value"):
            missing_critical.append(field)
        elif val.get("ocr_confidence", 100) < 60 or val.get("extraction_confidence", 1) < 0.6:
            low_confidence.append(field)

    for field in config.get("important", []):
        val = get_nested_value(extracted_data, field) or {}
        if not val.get("value"):
            missing_important.append(field)

    if doc_type == "CLAIMANT_STATEMENT_FORM":
        missing_bank = []
        for field in config.get("bank_critical", []):
            val = get_nested_value(extracted_data, field) or {}
            if not val.get("value"):
                missing_bank.append(field)
        if len(missing_bank) >= 2:
            return build_error("BANK_DETAILS_UNCLEAR", doc_type, missing_bank, [])

    if len(missing_critical) >= 1:
        return build_error("CRITICAL_MISSING", doc_type, missing_critical, missing_important)

    if len(missing_important) >= 3:
        return build_error("POOR_SCAN", doc_type, missing_critical, missing_important)

    if len(low_confidence) >= 2:
        return build_error("LOW_QUALITY", doc_type, missing_critical, missing_important)

    return {"status": "OK"}

def build_error(error_type, doc_type, missing_critical, missing_important):
    reasons = missing_critical + missing_important
    display_reasons = [r.split('.')[-1] for r in reasons]
    doc_label = doc_type.replace('_', ' ').title()
    
    messages = {
        "CRITICAL_MISSING": {
            "title": f"{doc_label} — Required information missing",
            "message": f"The following critical fields could not be read: {', '.join(display_reasons)}.",
            "action": "Please upload a clearer image of your document."
        },
        "POOR_SCAN": {
            "title": f"{doc_label} — Document unclear",
            "message": f"Several fields are missing: {', '.join(display_reasons)}. This usually means the document is partially cut off or too blurry.",
            "action": "Please retake the photo ensuring the full document is visible and in good lighting."
        },
        "LOW_QUALITY": {
            "title": f"{doc_label} — Image quality too low",
            "message": "The document was detected but text confidence is too low to verify accurately.",
            "action": "Please upload a higher quality image."
        },
        "FRONT_SIDE_ONLY": {
            "title": "Aadhaar Card — Back side missing",
            "message": "We received only the front side of your Aadhaar card. Address details are on the back side.",
            "action": "Please upload both sides of your Aadhaar card."
        },
        "BACK_PAGE_MISSING": {
            "title": "Passport — Back page missing",
            "message": "Passport back page is missing. MRZ lines are required for verification.",
            "action": "Please upload the back page of your passport."
        },
        "BANK_DETAILS_UNCLEAR": {
            "title": "Claim Form — Bank details missing",
            "message": "Bank details section is unclear or missing.",
            "action": "Please re-upload the page containing your bank details."
        }
    }
    
    return {
        "status": "REUPLOAD_REQUIRED",
        "error_type": error_type,
        "document": doc_type,
        "missing_fields": display_reasons,
        **messages.get(error_type, messages["POOR_SCAN"])
    }
