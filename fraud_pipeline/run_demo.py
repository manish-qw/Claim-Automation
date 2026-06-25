import sys
import os
import time
from dotenv import load_dotenv

# Load local .env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Add the parent directory to sys.path so python can find the 'fraud_pipeline' module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraud_pipeline.pipeline import FraudPipeline
from fraud_pipeline.demo_scenarios.scenarios import ALL_DEMOS


def run_all_demos():
    pipeline = FraudPipeline()

    for i, (demo_name, state) in enumerate(ALL_DEMOS.items()):
        print(f"\n{'='*60}")
        print(f"Executing Scenario: {demo_name.upper()}")
        print(f"{'='*60}")

        try:
            result_package = pipeline.run(state)

            print(f"Decision:     {result_package.final_recommendation}")
            print(f"Escalation:   {result_package.escalation_required}")

            # Fraud risk score
            fraud_data = getattr(state, "fraud_analysis", {}) or {}
            if fraud_data.get("fraud_risk_score") is not None:
                print(f"Fraud Score:  {fraud_data['fraud_risk_score']:.3f}  ({fraud_data.get('fraud_risk_level', '?')})")

            # Trust score
            trust_data = getattr(state, "trust_analysis", {}) or {}
            if trust_data.get("overall_trust_score") is not None:
                print(f"Trust Score:  {trust_data['overall_trust_score']:.3f}")

            # ── AI Analysis Summaries ──────────────────────────────────────────
            print("\n--- AI Analysis Summaries ---")

            # Agent 2: Llama fraud explanation
            agent2_explanation = getattr(state, "anomaly_explanation", None)
            if agent2_explanation:
                print(f"\nAgent 2 (Llama 3.2 - Fraud Explanation):\n{agent2_explanation}")

            # Agent 6: Llama executive summary
            agent6_summary = trust_data.get("executive_summary")
            if agent6_summary and "unavailable" not in agent6_summary.lower():
                print(f"\nAgent 6 (Llama 3.2 - Executive Summary):\n{agent6_summary}")

            # Agent 7: Llama graph network summary
            agent7_data = getattr(state, "graph_analysis", {}) or {}
            agent7_summary = agent7_data.get("graph_summary")
            if agent7_summary:
                print(f"\nAgent 7 (Llama 3.2 - Network Summary):\n{agent7_summary}")

            # Validation flags
            if state.validation_flags:
                print(f"\nValidation Flags: {', '.join(state.validation_flags)}")

        except Exception as e:
            print(f"Error running scenario {demo_name}: {e}")
            import traceback
            traceback.print_exc()

        if i < len(ALL_DEMOS) - 1:
            print("\n[INFO] Sleeping 15s to respect Gemini rate limits...")
            time.sleep(15)


if __name__ == "__main__":
    run_all_demos()
