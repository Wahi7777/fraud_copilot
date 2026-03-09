from datetime import datetime
from typing import Iterable, List

from models import (
    EnrichedTransactionRecord,
    EvidenceItem,
    EvidenceSeverity,
    FraudSignalCode,
    InvestigationStatus,
)
from services.investigation_pipeline import InvestigationPipeline
from services.investigation_feature_builder import InvestigationFeatureBuilder, InvestigationContext
from services.evidence_registry import EvidenceRegistry
from agents.pattern_agent import PatternAgent
from agents.behaviour_agent import BehaviourAgent
from agents.network_agent import NetworkAgent
from agents.typology_agent import TypologyAgent
from agents.decision_agent import DecisionAgent
from orchestration.crew import CrewRunResult, InvestigationCrew


class _FakeRepository:
    def __init__(self, record: EnrichedTransactionRecord) -> None:
        self.record = record

    def get_transaction(self, transaction_id: str) -> EnrichedTransactionRecord | None:
        if transaction_id == self.record.transaction_id:
            return self.record
        return None

    def get_account_history(self, account_id: str, limit: int = 50) -> list[EnrichedTransactionRecord]:
        if account_id == self.record.account_id:
            return [self.record]
        return []


class _FakeGraphBuilder:
    def __init__(self, beneficiary_id: str, account_id: str) -> None:
        self.beneficiary_id = beneficiary_id
        self.account_id = account_id

    def get_beneficiary_degree(self, beneficiary_id: str) -> int:
        return 6 if beneficiary_id == self.beneficiary_id else 0

    def has_path_to_flagged_node(self, account_id: str) -> bool:
        return account_id == self.account_id

    def detect_simple_circular_pattern(self, account_id: str) -> bool:
        return False


class _FakeFeatureBuilder(InvestigationFeatureBuilder):
    """
    Override the real feature builder to avoid dataset access in unit tests.
    """

    def __init__(self, record: EnrichedTransactionRecord) -> None:
        self._record = record

    def build_for_transaction(self, ctx: InvestigationContext) -> EnrichedTransactionRecord:  # type: ignore[override]
        return self._record


def _base_tx() -> EnrichedTransactionRecord:
    return EnrichedTransactionRecord(
        transaction_id="TXN-PIPE-1",
        account_id="ACC-PIPE-1",
        beneficiary_id="ACC-PIPE-2",
        amount=7500.0,
        transaction_type="TRANSFER",
        timestamp=datetime.fromisoformat("2026-03-06T14:02:00"),
        device_id="Device-123",
        device_type="MOBILE",
        ip_address="10.0.0.2",
        country="NG",
        account_age_days=20,
        beneficiary_age_days=10,
        transaction_velocity_10min=4.0,
        transaction_velocity_1hr=8.0,
        merchant_category="MONEY_TRANSFER",
        device_risk_score=0.8,
        email_domain_risk=0.6,
        geo_distance_jump_km=3000.0,
        impossible_travel_flag=True,
    )


def test_successful_end_to_end_investigation():
    tx = _base_tx()
    repo = _FakeRepository(tx)
    feature_builder = _FakeFeatureBuilder(tx)
    graph_builder = _FakeGraphBuilder(beneficiary_id=tx.beneficiary_id, account_id=tx.account_id)

    # Use a real InvestigationCrew to ensure the pipeline delegates to CrewAI orchestration.
    pattern = PatternAgent()
    behaviour = BehaviourAgent()
    network = NetworkAgent(graph_builder)  # type: ignore[arg-type]
    typology = TypologyAgent()
    decision = DecisionAgent()

    crew = InvestigationCrew(
        pattern_agent=pattern,
        behaviour_agent=behaviour,
        network_agent=network,
        typology_agent=typology,
        decision_agent=decision,
    )

    pipeline = InvestigationPipeline(
        repository=repo,
        feature_builder=feature_builder,
        graph_builder=graph_builder,  # type: ignore[arg-type]
        evidence_registry=EvidenceRegistry(),
        pattern_agent=pattern,
        behaviour_agent=behaviour,
        network_agent=network,  # type: ignore[arg-type]
        typology_agent=typology,
        decision_agent=decision,
        investigation_crew=crew,
    )

    output = pipeline.run_investigation(tx.transaction_id).to_dict()

    assert output["status"] == "success"
    inv = output["investigation"]
    assert inv["transaction_id"] == tx.transaction_id
    assert inv["status"] == InvestigationStatus.COMPLETED.value

    decision = output["decision"]
    assert 0.0 <= decision["risk_score"] <= 0.99
    assert isinstance(decision["recommendation"], str)
    assert isinstance(decision["decision_rationale"], str)

    evidence = output["evidence"]
    assert isinstance(evidence, list)
    assert len(evidence) > 0
    # Ensure evidence fields match schema.
    for raw in evidence:
        e = EvidenceItem.model_validate(raw)
        assert e.source_agent
        assert isinstance(e.signal_code, FraudSignalCode)


def test_missing_transaction_handling():
    tx = _base_tx()
    repo = _FakeRepository(tx)
    feature_builder = _FakeFeatureBuilder(tx)
    graph_builder = _FakeGraphBuilder(beneficiary_id=tx.beneficiary_id, account_id=tx.account_id)

    pipeline = InvestigationPipeline(
        repository=repo,
        feature_builder=feature_builder,
        graph_builder=graph_builder,  # type: ignore[arg-type]
        evidence_registry=EvidenceRegistry(),
        pattern_agent=PatternAgent(),
        behaviour_agent=BehaviourAgent(),
        network_agent=NetworkAgent(graph_builder),  # type: ignore[arg-type]
        typology_agent=TypologyAgent(),
        decision_agent=DecisionAgent(),
    )

    output = pipeline.run_investigation("NON_EXISTENT").to_dict()
    assert output["status"] == "not_found"
    assert output["investigation"] is None
    assert "message" in output["decision"]

