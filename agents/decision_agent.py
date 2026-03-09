from __future__ import annotations

from typing import Any, Dict, Iterable, List

from models import EvidenceItem
from services.scoring_engine import ScoringEngine, ScoreBreakdown
from .base_agent import BaseInvestigationAgent


class DecisionAgent(BaseInvestigationAgent):
    """
    Produces final risk score, typology, recommendation and confidence.
    """

    def __init__(self, scoring_engine: ScoringEngine | None = None) -> None:
        super().__init__(
            name="DecisionAgent",
            description="Aggregates scores and evidence into a final decision.",
            domain="decisioning",
        )
        self.scoring_engine = scoring_engine or ScoringEngine()

    def run(
        self,
        evidence_items: Iterable[EvidenceItem],
        *,
        typology: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        items_list: List[EvidenceItem] = list(evidence_items)
        breakdown: ScoreBreakdown = self.scoring_engine.score(items_list)

        inferred_typology = typology or "Generic Fraud"
        recommendation = self._recommendation_for_band(breakdown.risk_band)
        rationale = self._build_rationale(inferred_typology, breakdown)

        return {
            "agent": self.name,
            "risk_score": breakdown.risk_score,
            "typology": inferred_typology,
            "recommendation": recommendation,
            "confidence": breakdown.confidence,
            "decision_rationale": rationale,
        }

    @staticmethod
    def _recommendation_for_band(risk_band: str) -> str:
        if risk_band == "high":
            return "Escalate immediately and consider account and beneficiary freeze."
        if risk_band == "medium":
            return "Prioritise manual review within SLA and monitor closely."
        return "Monitor with no immediate blocking action."

    @staticmethod
    def _build_rationale(typology: str, breakdown: ScoreBreakdown) -> str:
        if not breakdown.signal_weights:
            return f"Typology {typology} inferred with low confidence and no strong fraud signals."

        top_signals = sorted(
            breakdown.signal_weights.items(), key=lambda kv: kv[1], reverse=True
        )
        top_str = ", ".join(f"{code.value} ({weight})" for code, weight in top_signals)
        return (
            f"Typology {typology} inferred based on signals: {top_str}. "
            f"Total risk points {breakdown.risk_points} mapped to score {breakdown.risk_score:.2f}."
        )

