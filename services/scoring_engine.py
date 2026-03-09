from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict

from models import EvidenceItem, FraudSignalCode


@dataclass
class ScoreBreakdown:
    """
    Auditable scoring result for an investigation.
    """

    risk_points: int
    risk_score: float
    confidence: float
    confidence_band: str
    risk_band: str
    signal_weights: Dict[FraudSignalCode, int]


class ScoringEngine:
    """
    Deterministic, explainable fraud scoring engine.

    Weights are derived from the PRD with additional values for auxiliary
    signals so that the total risk can exceed 100 points (before capping).
    """

    WEIGHTS: Dict[FraudSignalCode, int] = {
        FraudSignalCode.HIGH_VELOCITY: 25,
        FraudSignalCode.NEW_DEVICE: 20,
        FraudSignalCode.NEW_COUNTRY: 15,
        FraudSignalCode.LINKED_FRAUD_ACCOUNT: 30,
        FraudSignalCode.IMPOSSIBLE_TRAVEL: 10,
        FraudSignalCode.MULE_PATTERN: 20,
    }

    def score(self, evidence: Iterable[EvidenceItem]) -> ScoreBreakdown:
        """
        Compute an auditable score from a collection of evidence items.

        The engine:
        - Collapses evidence by unique FraudSignalCode
        - Applies deterministic weights
        - Caps the final score at 0.99 as per PRD
        - Derives human-readable confidence and risk bands
        """
        seen_codes: set[FraudSignalCode] = set()
        contributions: Dict[FraudSignalCode, int] = {}

        for item in evidence:
            code = item.signal_code
            if code in seen_codes:
                continue
            seen_codes.add(code)

            weight = self.WEIGHTS.get(code, 0)
            if weight > 0:
                contributions[code] = weight

        risk_points = sum(contributions.values())
        risk_score = min(risk_points / 100.0, 0.99)

        confidence = risk_score  # direct mapping keeps interpretation simple
        confidence_band = self._confidence_band(confidence)
        risk_band = self._risk_band(risk_score)

        return ScoreBreakdown(
            risk_points=risk_points,
            risk_score=risk_score,
            confidence=confidence,
            confidence_band=confidence_band,
            risk_band=risk_band,
            signal_weights=contributions,
        )

    @staticmethod
    def _risk_band(score: float) -> str:
        if score < 0.3:
            return "low"
        if score < 0.7:
            return "medium"
        return "high"

    @staticmethod
    def _confidence_band(confidence: float) -> str:
        if confidence < 0.3:
            return "LOW"
        if confidence < 0.7:
            return "MEDIUM"
        return "HIGH"

