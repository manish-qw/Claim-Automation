"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Shared pytest fixtures for unit and integration tests.
    Provides mock objects, test data factories, and test infrastructure setup.

WHAT GOES HERE:

    FIXTURES:

    mock_claim_context() → ClaimContextObject
        Returns a fully populated ClaimContextObject for testing.
        Uses realistic test data (not production data).
        Cause of death: NATURAL. Contestability: OUT_OF_WINDOW.
        Aggregate trust: 0.85. Uncertainty score: 0.20.

    mock_fraud_report_clean() → FraudReport
        Returns a FraudReport with risk_level=LOW, risk_score=0.05.
        No fraud signals.

    mock_fraud_report_critical() → FraudReport
        Returns a FraudReport with risk_level=CRITICAL, risk_score=0.95.
        Includes AADHAAR_ACTIVE_ON_DECEASED and CRS_NOT_FOUND signals.

    mock_policy_assessment() → PolicyAssessment
        Returns a PolicyAssessment with FULL coverage, base_sum_assured=2000000,
        all_citations_verified=True.

    mock_llm_client(monkeypatch)
        Monkeypatches shared.llm.llm_client.complete() to return a
        deterministic LLMResponse without making real API calls.
        Configurable response text via fixture parameter.

    mock_db(monkeypatch)
        Monkeypatches all claim_repository functions to use in-memory
        dict storage. Tests do not touch a real PostgreSQL database.

    mock_kafka(monkeypatch)
        Monkeypatches kafka_client.publish() to collect events in a list
        for assertion. Tests can assert which events were published.

    test_settings() → Settings
        Returns a Settings instance with test-safe values:
            AUTO_APPROVE_MAX_SUM_ASSURED = 10_000_000
            AGENT_CONFIDENCE_MINIMUM = 0.50
            No real API keys (empty strings).

DEPENDENCIES:
    pytest, pytest-asyncio, factory_boy,
    shared.schemas.claim_context, shared.schemas.agent_output
"""
