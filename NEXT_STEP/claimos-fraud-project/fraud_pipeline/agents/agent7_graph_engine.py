"""
Agent 7 — Claim Evidence Graph Engine (Phase 4 — Full Implementation)
=======================================================================
Responsibilities:
  1. Persist full claim graph to Neo4j (8 nodes, 7 relationships)
  2. Run all 5 graph intelligence queries against Neo4j
  3. Fall back to in-process NetworkX analysis if Neo4j unavailable
  4. Produce explainable graph findings
  5. Compute graph risk score from all signals
  6. Detect fraud rings
  7. Escalate claim on confirmed ring detection

Output schema:
  {
    "connected_claims":     int,
    "shared_nominees":      int,
    "shared_bank_accounts": int,
    "shared_hospitals":     int,
    "shared_doctors":       int,
    "fraud_ring_detected":  bool,
    "graph_risk_score":     float,
    "graph_findings":       List[str],
    "high_risk_relationships": List[str],
    "ring_members":         List[str],
    "neo4j_persisted":      bool,
  }

Architecture:
  Primary path  — Neo4j (when connected):
    Neo4jGraphStore.persist_claim_graph()
    Neo4jGraphQueries.full_claim_analysis()

  Fallback path — NetworkX (always available):
    GraphIntelligenceService.ingest_claim()
    GraphIntelligenceService.analyse_claim()

  Both paths write to state.graph_analysis.
  Results are merged — Neo4j enhances but never replaces NetworkX.
"""

import logging
import os

from ..schemas import ClaimState, GraphAnalysisOutput
from ..services.graph_intelligence_service import GraphIntelligenceService
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)

# ── Risk weights for graph signals ────────────────────────────────────────────
# Added to rule_score in Agent 2 via state.graph_analysis fields
_RISK_WEIGHTS = {
    "shared_nominee":      0.30,
    "shared_bank_account": 0.28,
    "suspicious_hospital": 0.15,
    "shared_doctor":       0.18,
    "fraud_ring":          0.40,   # flat bonus when ring confirmed
}


def _load_neo4j_store():
    """
    Lazily load Neo4jGraphStore using environment variables.
    Returns (store, queries) tuple or (None, None) if Neo4j unavailable.
    """
    uri      = os.environ.get("NEO4J_URI", "")
    user     = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")

    if not uri:
        return None, None

    try:
        from ..services.neo4j_graph_store  import Neo4jGraphStore
        from ..services.neo4j_graph_queries import Neo4jGraphQueries

        store   = Neo4jGraphStore(uri=uri, user=user, password=password)
        queries = Neo4jGraphQueries(store=store) if store.is_available else None
        return store, queries
    except Exception as exc:
        logger.warning("[Agent 7] Neo4j load failed: %s", exc)
        return None, None


class ClaimEvidenceGraphEngine:
    """
    Graph-based fraud detection agent.

    Two-layer analysis:
      Layer 1 (NetworkX)  — always runs, in-process, fast
      Layer 2 (Neo4j)     — runs when connected, richer cross-claim queries
    """

    def __init__(self, neo4j_driver=None, postgres_service=None):
        # Layer 1 — in-process NetworkX (always available)
        self.graph_service = GraphIntelligenceService(
            neo4j_driver   = neo4j_driver,
            postgres_service = postgres_service,
        )

        # Layer 2 — Neo4j (loaded from env vars)
        self._neo4j_store   = None
        self._neo4j_queries = None
        self._neo4j_loaded  = False   # lazy init flag
        
        self.llm = LLMService(redis_client=None)
        
        gemini_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'agent7_gemini.txt')
        with open(gemini_path, 'r') as f:
            self.gemini_prompt = f.read()
            
        local_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'agent7_local.txt')
        with open(local_path, 'r') as f:
            self.local_prompt = f.read()

    def _ensure_neo4j(self):
        """Lazy-initialize Neo4j on first use."""
        if not self._neo4j_loaded:
            self._neo4j_store, self._neo4j_queries = _load_neo4j_store()
            self._neo4j_loaded = True

    def run(self, state: ClaimState) -> ClaimState:
        logger.info(
            "[Agent 7] Graph Engine → %s | historical_links=%d",
            state.claim_case_id,
            len(getattr(state, "historical_claim_links", [])),
        )

        self._ensure_neo4j()

        # ── Layer 1: NetworkX (always runs) ────────────────────────────────────
        self.graph_service.ingest_claim(state)
        nx_analysis = self.graph_service.analyse_claim(state)

        # ── Layer 2: Neo4j (runs when connected) ───────────────────────────────
        neo4j_persisted = False
        neo4j_analysis  = {}

        if self._neo4j_store and self._neo4j_store.is_available:
            # 2a. Persist full graph
            neo4j_persisted = self._neo4j_store.persist_claim_graph(state)

            # 2b. Run all 5 intelligence queries
            if self._neo4j_queries and neo4j_persisted:
                try:
                    neo4j_analysis = self._neo4j_queries.full_claim_analysis(
                        claim_id=state.claim_case_id
                    )
                    logger.info(
                        "[Agent 7] Neo4j analysis: nominees=%d accounts=%d "
                        "hospitals=%d doctors=%d ring=%s",
                        neo4j_analysis.get("shared_nominees", 0),
                        neo4j_analysis.get("shared_bank_accounts", 0),
                        neo4j_analysis.get("suspicious_hospitals", 0),
                        neo4j_analysis.get("shared_doctors", 0),
                        neo4j_analysis.get("fraud_ring_detected", False),
                    )
                except Exception as exc:
                    logger.warning("[Agent 7] Neo4j query failed: %s", exc)

        # ── Merge results (Neo4j enriches NetworkX baseline) ──────────────────
        analysis = self._merge_analyses(nx_analysis, neo4j_analysis)
        analysis["neo4j_persisted"] = neo4j_persisted

        # ── Compute graph risk score ───────────────────────────────────────────
        graph_risk_score = self._compute_risk_score(analysis)
        analysis["risk_score"] = graph_risk_score

        # ── Build explainable findings list ────────────────────────────────────
        findings = self._build_findings(analysis, state.claim_case_id)
        analysis["graph_findings"] = findings

        # ── Build output schema ────────────────────────────────────────────────
        output = GraphAnalysisOutput(
            connected_claims        = analysis["connected_claims"],
            shared_nominees         = analysis["shared_nominees"],
            shared_bank_accounts    = analysis["shared_bank_accounts"],
            shared_hospitals        = analysis.get("suspicious_hospitals", 0),
            shared_entities         = analysis.get("shared_entities", {}),
            high_risk_relationships = analysis.get("high_risk_relationships", []) + findings,
            fraud_ring_detected     = analysis["fraud_ring_detected"],
            graph_risk_score        = graph_risk_score,
            network_risk_score      = graph_risk_score,
            confidence_score        = 0.92 if analysis["fraud_ring_detected"] else 0.70,
            trust_score             = round(max(0.05, 1.0 - graph_risk_score), 4),
            validation_flags        = (
                ["FRAUD_RING_DETECTED"] if analysis["fraud_ring_detected"] else []
            ),
        )

        # Write full analysis dict (includes neo4j fields)
        state.graph_analysis = {
            **output.model_dump(),
            "graph_findings":    findings,
            "ring_members":      analysis.get("ring_members", []),
            "shared_doctors":    analysis.get("shared_doctors", 0),
            "neo4j_persisted":   neo4j_persisted,
        }

        # ── Escalation ────────────────────────────────────────────────────────
        if analysis["fraud_ring_detected"]:
            state.escalation_required = True
            if "FRAUD_RING_DETECTED" not in state.validation_flags:
                state.validation_flags.append("FRAUD_RING_DETECTED")
            logger.warning(
                "[Agent 7] FRAUD RING detected → %s risk=%.2f connected=%d ring=%s",
                state.claim_case_id,
                graph_risk_score,
                analysis["connected_claims"],
                analysis.get("ring_members", []),
            )

        # ── Step 8: Build Local LLM prompt from NetworkX findings (always available) ──
        # This must happen BEFORE the Gemini call so the local prompt always has data
        nx_findings_text = ", ".join(findings) if findings else "No critical network overlaps found in NetworkX analysis."
        local_prompt = self.local_prompt.replace("{graph_findings}", nx_findings_text)

        # ── Step 9: Gemini API Complex Network Detection ─────────────────────
        g_prompt = self.gemini_prompt.replace("{connected_claims}", str(analysis["connected_claims"]))\
            .replace("{shared_nominees}", str(analysis["shared_nominees"]))\
            .replace("{shared_banks}", str(analysis["shared_bank_accounts"]))\
            .replace("{shared_hospitals}", str(analysis.get("suspicious_hospitals", 0)))\
            .replace("{shared_doctors}", str(analysis.get("shared_doctors", 0)))
        logger.info("[Agent 7] Calling Gemini for Complex Network Detection...")
        g_result = self.llm.route_to_gemini(g_prompt, enforce_json=True)
        if g_result and g_result.get("network_overlap_detected"):
            if "FRAUD_RING_DETECTED" not in state.validation_flags:
                state.validation_flags.append("FRAUD_RING_DETECTED")
            findings.append(f"Gemini AI detected complex network overlap: {g_result.get('network_reasoning')}")

        # ── Step 10: Local LLM Graph Summary ──────────────────────────────────
        logger.info("[Agent 7] Calling Local LLM for Graph Summary...")
        explanation = self.llm.route_to_local(local_prompt)
        if explanation:
            state.graph_analysis["graph_summary"] = explanation

        return state

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _merge_analyses(
        self,
        nx: dict,
        neo4j: dict,
    ) -> dict:
        """
        Merge NetworkX and Neo4j results.
        Neo4j counts take precedence (more accurate cross-claim).
        NetworkX provides fallback and shared_entities.
        """
        # Shared nominees — take max (Neo4j more accurate but NetworkX catches in-process)
        shared_nominees = max(
            nx.get("shared_nominees", 0),
            neo4j.get("shared_nominees", 0),
        )
        shared_bank_accounts = max(
            nx.get("shared_bank_accounts", 0),
            neo4j.get("shared_bank_accounts", 0),
        )

        # Fraud ring — either source can confirm it
        fraud_ring = (
            nx.get("fraud_ring_detected", False)
            or neo4j.get("fraud_ring_detected", False)
        )

        # Connected claims — take max
        connected_claims = max(
            nx.get("connected_claims", 0),
            neo4j.get("connected_claims", 0),
        )

        return {
            "connected_claims":    connected_claims,
            "shared_nominees":     shared_nominees,
            "shared_bank_accounts": shared_bank_accounts,
            "suspicious_hospitals": neo4j.get("suspicious_hospitals", 0),
            "shared_doctors":      neo4j.get("shared_doctors", 0),
            "fraud_ring_detected": fraud_ring,
            "ring_members":        neo4j.get("ring_members", []),
            "shared_entities":     nx.get("shared_entities", {}),
            "high_risk_relationships": nx.get("high_risk_relationships", []),
            "component_size":      nx.get("component_size", 1),

            # Neo4j detailed lists (for audit trail)
            "shared_nominees_list":      neo4j.get("shared_nominees_list", []),
            "shared_bank_accounts_list": neo4j.get("shared_bank_accounts_list", []),
            "suspicious_hospitals_list": neo4j.get("suspicious_hospitals_list", []),
            "shared_doctors_list":       neo4j.get("shared_doctors_list", []),
        }

    def _compute_risk_score(self, analysis: dict) -> float:
        """
        Compute graph risk score from merged analysis results.
        Additive — each confirmed signal adds to total (capped at 0.99).
        """
        score = 0.0

        # Shared nominees (each nominee adds weight)
        n = analysis.get("shared_nominees", 0)
        if n >= 1:
            score += _RISK_WEIGHTS["shared_nominee"] * min(n, 3)

        # Shared bank accounts
        b = analysis.get("shared_bank_accounts", 0)
        if b >= 1:
            score += _RISK_WEIGHTS["shared_bank_account"] * min(b, 3)

        # Suspicious hospitals
        h = analysis.get("suspicious_hospitals", 0)
        if h >= 1:
            score += _RISK_WEIGHTS["suspicious_hospital"] * min(h, 3)

        # Shared doctors
        d = analysis.get("shared_doctors", 0)
        if d >= 1:
            score += _RISK_WEIGHTS["shared_doctor"] * min(d, 3)

        # Fraud ring flat bonus
        if analysis.get("fraud_ring_detected"):
            score += _RISK_WEIGHTS["fraud_ring"]

        return round(min(0.99, score), 4)

    def _build_findings(self, analysis: dict, claim_id: str) -> list:
        """
        Build human-readable fraud findings list for audit trail.
        Each finding maps directly to a triggered signal.
        """
        findings = []

        # Shared nominees
        for r in analysis.get("shared_nominees_list", []):
            findings.append(
                f"Nominee '{r.get('nominee_id', '')}' shared across "
                f"{r.get('claim_count', 0)} claims: {r.get('claim_ids', [])}"
            )
        if analysis.get("shared_nominees", 0) > 0 and not analysis.get("shared_nominees_list"):
            # NetworkX fallback finding
            findings.append(
                f"Nominee appears in {analysis['shared_nominees']} other claim(s) "
                f"(in-process registry)"
            )

        # Shared bank accounts
        for r in analysis.get("shared_bank_accounts_list", []):
            findings.append(
                f"Bank account '{r.get('account_id', '')}' ({r.get('bank_name', '')}) "
                f"shared across {r.get('claim_count', 0)} claims: {r.get('claim_ids', [])}"
            )
        if analysis.get("shared_bank_accounts", 0) > 0 and not analysis.get("shared_bank_accounts_list"):
            findings.append(
                f"Bank account appears in {analysis['shared_bank_accounts']} other claim(s) "
                f"(in-process registry)"
            )

        # Suspicious hospitals
        for r in analysis.get("suspicious_hospitals_list", []):
            findings.append(
                f"Hospital '{r.get('hospital_name', '')}' linked to "
                f"{r.get('claim_count', 0)} death claims: {r.get('claim_ids', [])}"
            )

        # Shared doctors
        for r in analysis.get("shared_doctors_list", []):
            findings.append(
                f"Doctor '{r.get('doctor_name', '')}' appears across "
                f"{r.get('claim_count', 0)} claims: {r.get('claim_ids', [])}"
            )

        # Fraud ring
        if analysis.get("fraud_ring_detected"):
            members = analysis.get("ring_members", [])
            findings.append(
                f"FRAUD RING CONFIRMED: {claim_id} is connected to "
                f"{len(members)} claim(s): {members}"
            )

        return findings
