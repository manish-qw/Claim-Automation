from .scenarios import ALL_DEMOS, DEMO_1_FRAUD_RING, DEMO_2_EARLY_CLAIM, DEMO_3_NON_DISCLOSURE, DEMO_4_TRUST_REDUCTION


def demo_fraud_ring():
    """Seed prior claims into graph before returning the fraud ring scenario."""
    from ..services.graph_intelligence_service import _ENTITY_CLAIMS, _GRAPH
    nominee_id = DEMO_1_FRAUD_RING.claimant.nominee_id  # NOM-001
    bank_acct  = DEMO_1_FRAUD_RING.claimant.bank_account_number  # ACC-9901
    for prior_id in ["CLM-PRIOR-FR-001", "CLM-PRIOR-FR-002"]:
        _GRAPH.add_node(prior_id, type="claim")
        if nominee_id:
            node_id = f"nominee_id::{nominee_id}"
            _GRAPH.add_node(node_id, type="nominee_id", value=nominee_id)
            _GRAPH.add_edge(prior_id, node_id, relation="HAS_NOMINEE_ID")
            _ENTITY_CLAIMS.setdefault(f"nominee_id::{nominee_id.lower()}", set()).add(prior_id)
        if bank_acct:
            node_id = f"bank_account_number::{bank_acct}"
            _GRAPH.add_node(node_id, type="bank_account_number", value=bank_acct)
            _GRAPH.add_edge(prior_id, node_id, relation="HAS_BANK_ACCOUNT_NUMBER")
            _ENTITY_CLAIMS.setdefault(f"bank_account_number::{bank_acct.lower()}", set()).add(prior_id)
    return DEMO_1_FRAUD_RING


def demo_early_claim(): return DEMO_2_EARLY_CLAIM
def demo_non_disclosure(): return DEMO_3_NON_DISCLOSURE
def demo_trust_reduction(): return DEMO_4_TRUST_REDUCTION
