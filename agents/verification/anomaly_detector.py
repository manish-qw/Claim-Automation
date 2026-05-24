"""
agents/verification/anomaly_detector.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Layer 2 of fraud detection. Isolation Forest ML model trained on
    historical claim patterns. Unsupervised — no fraud labels needed.
    Detects statistically anomalous claims regardless of known fraud rules.

WHAT GOES HERE:

    MODEL:
        Algorithm: Isolation Forest (scikit-learn IsolationForest)
        Training: Monthly retraining on all claims in the system.
        Storage: Versioned pickle file in AWS S3 bucket (settings.AWS_S3_MODEL_BUCKET).
        Loading: Loaded at agent startup, cached in memory. Thread-safe.
        If model file unavailable (S3 error, first deployment):
            → Log WARNING
            → Return anomaly_score = 0.5 (neutral, neither clean nor suspicious)
            → Continue pipeline — never block

    FEATURE VECTOR (8 features — all numeric):
        1. days_from_inception_to_claim
           (date_of_death - policy_inception_date).days
        2. days_from_death_to_intimation
           (claim_intimation_date - date_of_death).days
        3. sum_assured_to_annual_premium_ratio
           base_sum_assured / annual_premium_amount
        4. days_from_last_revival_to_death
           (date_of_death - last_revival_date).days  (0 if no revival)
        5. nominee_claim_count_last_12m
           from claim_history_checker result
        6. hospital_claim_count_last_12m
           from network_graph: how many recent claims involve this hospital
        7. doctor_claim_count_last_12m
           from network_graph: how many recent claims involve this doctor
        8. number_of_policies_on_deceased
           total live policies found on the same policy holder ID

    AnomalyResult (dataclass):
        anomaly_score       — float 0–1 (higher = more anomalous)
        is_anomalous        — bool (True if score > 0.30)
        top_3_anomalous_features — list[{feature_name, value, explanation}]
                              The three features that contributed most
                              to the anomaly score for this claim.
        model_version       — version tag of the model used (from S3 filename)

    FEATURE IMPORTANCE (for top_3_anomalous_features):
        Use permutation importance on the fitted model to determine
        which of the 8 features deviate most from the training distribution
        for this specific claim instance.

    MODEL VERSIONING:
        S3 key format: models/isolation_forest/v{YYYYMMDD}.pkl
        Always load the latest version (sorted by date prefix).
        Log the model version in every AnomalyResult for audit.

DEPENDENCIES:
    scikit-learn, boto3 (S3), numpy, pickle,
    shared.config.settings, shared.audit.audit_service
"""
