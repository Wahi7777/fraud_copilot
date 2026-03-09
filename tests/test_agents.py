from datetime import datetime
from typing import List

from models import (
    EnrichedTransactionRecord,
    EvidenceItem,
    EvidenceSeverity,
    FraudSignalCode,
)
from agents.pattern_agent import PatternAgent
from agents.behaviour_agent import BehaviourAgent
from agents.network_agent import NetworkAgent
from agents.typology_agent import TypologyAgent
from agents.decision_agent import DecisionAgent


class _FakeGraphBuilder:
    """
    Lightweight stand-in for GraphBuilder used in unit tests.
    """

    def __init__(self, high_degree_beneficiary: str, fraud_link_account: str) -> None:
        self.high_degree_beneficiary = high_degree_beneficiary
        self.fraud_link_account = fraud_link_account

    def get_beneficiary_degree(self, beneficiary_id: str) -> int:
        return 7 if beneficiary_id == self.high_degree_beneficiary else 0

    def has_path_to_flagged_node(self, account_id: str) -> bool:
        return account_id == self.fraud_link_account

    def detect_simple_circular_pattern(self, account_id: str) -> bool:
        return False


def _base_tx() -> EnrichedTransactionRecord:
    return EnrichedTransactionRecord(
        transaction_id="TXN-1",
        account_id="ACC-1",
        beneficiary_id="ACC-2",
        amount=5000.0,
        transaction_type="TRANSFER",
        timestamp=datetime.fromisoformat("2026-03-06T14:02:00"),
        device_id="iPhone-XYZ",
        device_type="MOBILE",
        ip_address="10.0.0.1",
        country="NG",
        account_age_days=10,
        beneficiary_age_days=5,
        transaction_velocity_10min=6.0,
        transaction_velocity_1hr=12.0,
        merchant_category="MONEY_TRANSFER",
        device_risk_score=0.85,
        email_domain_risk=0.7,
        geo_distance_jump_km=5000.0,
        impossible_travel_flag=True,
    )


def _validate_evidence_json(payload) -> List[EvidenceItem]:
    assert "agent" in payload
    assert "evidence_items" in payload
    items_json = payload["evidence_items"]
    assert isinstance(items_json, list)
    items: List[EvidenceItem] = []
    for raw in items_json:
        items.append(EvidenceItem.model_validate(raw))
    return items


def test_pattern_and_behaviour_agents_return_structured_evidence():
    tx = _base_tx()

    p_agent = PatternAgent()
    b_agent = BehaviourAgent()

    p_payload = p_agent.run(tx)
    b_payload = b_agent.run(tx)

    p_items = _validate_evidence_json(p_payload)
    b_items = _validate_evidence_json(b_payload)

    assert any(i.signal_code == FraudSignalCode.HIGH_VELOCITY for i in p_items)
    assert any(i.signal_code in {FraudSignalCode.NEW_DEVICE, FraudSignalCode.IMPOSSIBLE_TRAVEL} for i in b_items)


def test_network_agent_uses_graph_context():
    tx = _base_tx()
    fake_graph = _FakeGraphBuilder(high_degree_beneficiary=tx.beneficiary_id, fraud_link_account=tx.account_id)
    n_agent = NetworkAgent(fake_graph)

    payload = n_agent.run(tx)
    items = _validate_evidence_json(payload)

    # Expect at least one mule or linked-fraud signal from the fake graph setup.
    assert any(
        i.signal_code in {FraudSignalCode.MULE_PATTERN, FraudSignalCode.LINKED_FRAUD_ACCOUNT}
        for i in items
    )


def test_typology_and_decision_agents_output_schema():
    # Build a small evidence set that looks like account takeover.
    ev = [
        EvidenceItem(
            source_agent="PatternAgent",
            signal_code=FraudSignalCode.HIGH_VELOCITY,
            severity=EvidenceSeverity.HIGH,
            summary="High velocity",
            details=None,
        ),
        EvidenceItem(
            source_agent="BehaviourAgent",
            signal_code=FraudSignalCode.NEW_DEVICE,
            severity=EvidenceSeverity.HIGH,
            summary="New device",
            details=None,
        ),
    ]

    t_agent = TypologyAgent()
    t_payload = t_agent.run(ev)

    assert t_payload["agent"] == "TypologyAgent"
    assert t_payload["typology"] in {"Account Takeover", "Mule Network", "Generic Fraud"}

    d_agent = DecisionAgent()
    d_payload = d_agent.run(ev, typology=t_payload["typology"])

    for key in ("agent", "risk_score", "typology", "recommendation", "confidence", "decision_rationale"):
        assert key in d_payload

    assert isinstance(d_payload["risk_score"], float)
    assert 0.0 <= d_payload["risk_score"] <= 0.99
    assert isinstance(d_payload["confidence"], float)
    assert isinstance(d_payload["decision_rationale"], str)

