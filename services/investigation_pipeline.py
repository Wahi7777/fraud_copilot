from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from fastapi.encoders import jsonable_encoder
from models import (
    EnrichedTransactionRecord,
    EvidenceItem,
    InvestigationState,
    InvestigationStatus,
)
from services.data_repository import DataRepository, DataRepositoryConfig
from services.evidence_registry import EvidenceRegistry
from services.graph_builder import GraphBuilder
from services.investigation_feature_builder import InvestigationContext, InvestigationFeatureBuilder
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

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "investigation": self.investigation.model_dump(mode="json") if self.investigation else None,
            "evidence": [e.model_dump(mode="json") for e in self.evidence],
            "decision": self.decision,
            "agent_trace": self.agent_trace,
            "llm_provider_used": self.llm_provider_used,
            "model_used": self.model_used,
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
        )


def run_investigation(transaction_id: str, *, max_rows: int | None = 10_000) -> Dict[str, Any]:
    """
    Convenience entrypoint used by the API layer to execute an investigation.
    """
    repo = DataRepository.load_from_csvs(DataRepositoryConfig(max_rows=max_rows))
    pipeline = InvestigationPipeline(repository=repo)
    output = pipeline.run_investigation(transaction_id)
    return output.to_dict()

