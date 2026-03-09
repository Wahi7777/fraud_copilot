from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional

from fastapi.encoders import jsonable_encoder
from models import (
    EnrichedTransactionRecord,
    EvidenceSeverity,
    EvidenceItem,
    FraudSignalCode,
    InvestigationState,
    InvestigationStatus,
)
from services.data_repository import DataRepository, DataRepositoryConfig
from services.evidence_registry import EvidenceRegistry
from services.graph_builder import GraphBuilder
from services.investigation_feature_builder import InvestigationContext, InvestigationFeatureBuilder
from services.scoring_engine import ScoringEngine
from services.typology_classifier import classify_typology_from_signals
from agents.pattern_agent import PatternAgent
from agents.behaviour_agent import BehaviourAgent
from agents.network_agent import NetworkAgent
from agents.typology_agent import TypologyAgent
from agents.decision_agent import DecisionAgent
from orchestration.crew import CrewRunResult, InvestigationCrew


@dataclass
class InvestigationOutput:
    """
    Structured output of a single investigation run.
    """

    status: str
    investigation: Optional[InvestigationState]
    evidence: List[EvidenceItem]
    decision: Optional[Dict[str, Any]]
    agent_trace: List[Dict[str, str]]
    llm_provider_used: str
    model_used: str
    signal_breakdown: Dict[str, float]
    triggered_signals: List[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "investigation": self.investigation.model_dump(mode="json") if self.investigation else None,
            "evidence": [e.model_dump(mode="json") for e in self.evidence],
            "decision": self.decision,
            "agent_trace": self.agent_trace,
            "llm_provider_used": self.llm_provider_used,
            "model_used": self.model_used,
            "signal_breakdown": self.signal_breakdown,
            "triggered_signals": self.triggered_signals,
        }
        return jsonable_encoder(payload)


class InvestigationPipeline:
    """
    Orchestrates the investigation flow end-to-end for a single transaction.
    """

    def __init__(
        self,
        repository: DataRepository,
        feature_builder: InvestigationFeatureBuilder | None = None,
        graph_builder: GraphBuilder | None = None,
        evidence_registry: EvidenceRegistry | None = None,
        pattern_agent: PatternAgent | None = None,
        behaviour_agent: BehaviourAgent | None = None,
        network_agent: NetworkAgent | None = None,
        typology_agent: TypologyAgent | None = None,
        decision_agent: DecisionAgent | None = None,
        scoring_engine: ScoringEngine | None = None,
        investigation_crew: InvestigationCrew | None = None,
    ) -> None:
        self.repository = repository
        self.feature_builder = feature_builder or InvestigationFeatureBuilder(repository)
        self.graph_builder = graph_builder or GraphBuilder(repository)
        self.registry = evidence_registry or EvidenceRegistry()

        self.pattern_agent = pattern_agent or PatternAgent()
        self.behaviour_agent = behaviour_agent or BehaviourAgent()
        self.network_agent = network_agent or NetworkAgent(self.graph_builder)
        self.typology_agent = typology_agent or TypologyAgent()
        self.decision_agent = decision_agent or DecisionAgent()
        self.scoring_engine = scoring_engine or ScoringEngine()

        # CrewAI orchestration layer wrapping the domain agents.
        self.investigation_crew = investigation_crew or InvestigationCrew(
            pattern_agent=self.pattern_agent,
            behaviour_agent=self.behaviour_agent,
            network_agent=self.network_agent,
            typology_agent=self.typology_agent,
            decision_agent=self.decision_agent,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run_investigation(
        self,
        transaction_id: str,
        *,
        progress_callback: Callable[[Dict[str, Any]], None] | None = None,
    ) -> InvestigationOutput:
        """
        Execute the full investigation workflow for a given transaction id.
        """
        base_tx = self.repository.get_transaction(transaction_id)
        if base_tx is None:
            return InvestigationOutput(
                status="not_found",
                investigation=None,
                evidence=[],
                decision={
                    "message": f"Transaction {transaction_id} not found in repository.",
                },
                agent_trace=[],
                llm_provider_used="deterministic",
                model_used="n/a",
                signal_breakdown={},
                triggered_signals=[],
            )

        ctx = InvestigationContext(transaction_id=transaction_id)
        enriched = self.feature_builder.build_for_transaction(ctx)

        state = InvestigationState(
            investigation_id=f"INV-{transaction_id}",
            transaction_id=transaction_id,
            account_id=enriched.account_id,
            status=InvestigationStatus.IN_PROGRESS,
            base_transaction=enriched,
        )
        # Deterministic evidence layer: 15-signal score before OpenAI reasoning.
        account_history = self.repository.get_account_history(enriched.account_id, limit=10_000)
        beneficiary_incoming_24h = self._beneficiary_incoming_count_24h(enriched)
        has_hist_fraud_link = self._historical_fraud_link(enriched)
        has_beneficiary_flagged = False
        if hasattr(self.graph_builder, "build_graph"):
            graph = self.graph_builder.build_graph()
            has_beneficiary_flagged = (
                bool(graph.nodes.get(enriched.beneficiary_id, {}).get("fraud_flagged"))
                if enriched.beneficiary_id in graph
                else False
            )
        score = self.scoring_engine.score_transaction(
            enriched,
            account_history=account_history,
            beneficiary_incoming_count_24h=beneficiary_incoming_24h,
            beneficiary_degree=self.graph_builder.get_beneficiary_degree(enriched.beneficiary_id),
            has_sender_path_to_flagged=self.graph_builder.has_path_to_flagged_node(enriched.account_id),
            has_beneficiary_path_to_flagged=has_beneficiary_flagged,
            has_historical_fraud_link=has_hist_fraud_link,
        )
        state.risk_score = score.risk_score
        state.signal_breakdown = score.signal_breakdown
        state.triggered_signals = score.triggered_signals
        state.synthetic_telemetry_signals = score.synthetic_telemetry_signals
        typology_candidate = classify_typology_from_signals(
            triggered_signals=score.triggered_signals,
            signal_breakdown=score.signal_breakdown,
        )
        state.typology = typology_candidate.fraud_typology
        state.typology_confidence = typology_candidate.typology_confidence
        state.typology_definition = typology_candidate.typology_definition
        state.typology_reason = typology_candidate.typology_reason
        self._seed_scoring_evidence(state, score)
        # Delegate to the CrewAI-based orchestration layer (fixed sequential order).
        crew_result: CrewRunResult = self.investigation_crew.run(
            state,
            self.registry,
            progress_callback=progress_callback,
        )
        state = crew_result.state
        state.status = InvestigationStatus.COMPLETED

        return InvestigationOutput(
            status="success",
            investigation=state,
            evidence=crew_result.evidence,
            decision=crew_result.decision,
            agent_trace=crew_result.agent_trace,
            llm_provider_used=crew_result.llm_provider_used,
            model_used=crew_result.model_used,
            signal_breakdown=score.signal_breakdown,
            triggered_signals=score.triggered_signals,
        )

    def _seed_scoring_evidence(self, state: InvestigationState, score: Any) -> None:
        code_map: Dict[str, FraudSignalCode] = {
            "amount_spike": FraudSignalCode.AMOUNT_SPIKE,
            "balance_drain": FraudSignalCode.BALANCE_DRAIN,
            "account_emptying": FraudSignalCode.ACCOUNT_EMPTYING,
            "tx_burst_1h": FraudSignalCode.TX_BURST_1H,
            "unique_beneficiaries_1h": FraudSignalCode.UNIQUE_BENEFICIARIES_1H,
            "beneficiary_incoming_volume": FraudSignalCode.BENEFICIARY_INCOMING_VOLUME,
            "fanout_pattern": FraudSignalCode.FANOUT_PATTERN,
            "transfer_cashout_pattern": FraudSignalCode.TRANSFER_CASHOUT_PATTERN,
            "risky_beneficiary_link": FraudSignalCode.RISKY_BENEFICIARY_LINK,
            "historical_fraud_link": FraudSignalCode.HISTORICAL_FRAUD_LINK,
            "device_reuse_risk": FraudSignalCode.DEVICE_REUSE_RISK,
            "new_device_indicator": FraudSignalCode.NEW_DEVICE_INDICATOR,
            "geo_anomaly": FraudSignalCode.GEO_ANOMALY,
            "ip_risk": FraudSignalCode.IP_RISK,
            "time_of_day_deviation": FraudSignalCode.TIME_OF_DAY_DEVIATION,
        }
        items: List[EvidenceItem] = []
        for signal in state.triggered_signals:
            code = code_map.get(signal)
            if code is None:
                continue
            contribution = float(state.signal_breakdown.get(signal, 0.0))
            severity = (
                EvidenceSeverity.CRITICAL
                if contribution >= 0.08
                else EvidenceSeverity.HIGH
                if contribution >= 0.05
                else EvidenceSeverity.MEDIUM
            )
            items.append(
                EvidenceItem(
                    source_agent="ScoringEngine",
                    signal_code=code,
                    severity=severity,
                    summary=str(score.signal_reasons.get(signal) or f"Triggered {signal}"),
                    details={"signal": signal, "contribution": contribution},
                )
            )
        if items:
            self.registry.extend(items)

    def _beneficiary_incoming_count_24h(self, tx: EnrichedTransactionRecord) -> int:
        if not hasattr(self.repository, "iter_records_with_labels"):
            return 0
        start = tx.timestamp - timedelta(hours=24)
        count = 0
        for rec, _label in self.repository.iter_records_with_labels(limit=200_000):
            if rec.beneficiary_id == tx.beneficiary_id and start <= rec.timestamp <= tx.timestamp:
                count += 1
        return count

    def _historical_fraud_link(self, tx: EnrichedTransactionRecord) -> bool:
        if not hasattr(self.repository, "iter_records_with_labels"):
            return False
        start = tx.timestamp - timedelta(days=30)
        for rec, label in self.repository.iter_records_with_labels(limit=200_000):
            if not label:
                continue
            if rec.timestamp > tx.timestamp or rec.timestamp < start:
                continue
            if rec.account_id == tx.account_id or rec.beneficiary_id == tx.beneficiary_id:
                return True
        return False


def run_investigation(transaction_id: str, *, max_rows: int | None = 10_000) -> Dict[str, Any]:
    """
    Convenience entrypoint used by the API layer to execute an investigation.
    """
    repo = DataRepository.load_from_csvs(DataRepositoryConfig(max_rows=max_rows))
    pipeline = InvestigationPipeline(repository=repo)
    output = pipeline.run_investigation(transaction_id)
    return output.to_dict()

