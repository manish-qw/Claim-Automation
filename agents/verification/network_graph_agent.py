"""
agents/verification/network_graph_agent.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Builds and queries a Neo4j fraud network graph. Detects suspicious
    connections between claimants, hospitals, doctors, lawyers, and bank
    accounts across multiple claims. Runs in parallel with fraud Layers 1 and 2.

WHAT GOES HERE:

    NEO4J GRAPH SCHEMA:

    Node Types (7):
        Claimant     — {claimant_id, name, aadhaar_masked, pan_masked}
        Hospital     — {hospital_name, nha_id, district, state}
        Doctor       — {doctor_name, nmc_registration, specialisation}
        Lawyer       — {lawyer_name, bar_council_id, city}
        BankAccount  — {account_last4, ifsc, bank_name}
        PolicyId     — {policy_id, insurer}
        NomineeAccount — {account_last4, ifsc, holder_name}

    Edge Types (6):
        FILED_CLAIM          Claimant → PolicyId
        TREATED_AT           Claimant → Hospital
        PERFORMED_PMR        Doctor → Claimant
        FILED_FIR            (police_station string) → Claimant
        REGISTERED_AS_NOMINEE NomineeAccount → PolicyId
        REPRESENTED_BY       Lawyer → Claimant

    ENTITY RESOLUTION (before every INSERT):
        Before creating a new node, check if a similar entity already exists:
        Use fuzzy name matching (Levenshtein ≥ 85%) to detect:
            "AutoFix Pune" and "Auto Fix, Pune" → same hospital node
            "Dr. Sharma R." and "Dr. R. Sharma" → same doctor node
        If match found: MERGE into existing node (update properties).
        If no match: CREATE new node.
        This prevents graph fragmentation from minor name variations.

    GRAPH OPERATIONS PER CLAIM:
        1. Upsert all entities extracted from claim documents into graph.
        2. Create all relevant edges for this claim.

    FRAUD SIGNAL QUERIES (run after upsert):
        Query 1: Doctors connected to 3+ separate unrelated claims
            MATCH (d:Doctor)-[:PERFORMED_PMR]->(:Claimant) WITH d, count(*) as degree
            WHERE degree > 3 RETURN d
        Query 2: Hospitals connected to claimants from 3+ different states
        Query 3: Lawyers appearing in 2+ claims for different families same month
        Query 4: BankAccounts registered as nominee on 3+ unrelated policies

    NetworkGraphResult (dataclass):
        suspicious_doctors    — list[{doctor_name, claim_count, claim_ids}]
        suspicious_hospitals  — list[{hospital_name, state_count, states}]
        suspicious_lawyers    — list[{lawyer_name, claim_count, month}]
        suspicious_accounts   — list[{account_last4, policy_count, policy_ids}]
        graph_updated_at      — UTC timestamp

    FALLBACK:
        If Neo4j is unavailable: log WARNING, return empty NetworkGraphResult.
        Never block the pipeline. Graph grows over time — at first deployment
        it has no historical data and returns nothing (expected).

DEPENDENCIES:
    neo4j (Python driver), Levenshtein,
    shared.config.settings, shared.audit.audit_service
"""
