from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

from models import EvidenceItem, FraudSignalCode
from services.typology_classifier import TYPOLOGY_DEFINITIONS, classify_typology_from_signals
from .base_agent import BaseInvestigationAgent


class TypologyAgent(BaseInvestigationAgent):
    """
    Maps evidence collections to high-level fraud typologies.
    """

    def __init__(self) -> None:
        super().__init__(
            name="TypologyAgent",
            description="Classifies fraud typology based on evidence signals.",
            domain="fraud_typology",
        )

    def run(self, evidence_items: Iterable[EvidenceItem], **kwargs: Any) -> Dict[str, Any]:
        items = list(evidence_items)
        counts = Counter(e.signal_code for e in items)
        triggered_signals = kwargs.get("triggered_signals")
        if not isinstance(triggered_signals, list):
            triggered_signals = [code.value.lower() for code in counts.keys()]
        signal_breakdown = kwargs.get("signal_breakdown")
        if not isinstance(signal_breakdown, dict):
            signal_breakdown = {code.value.lower(): 0.08 for code in counts.keys()}
        candidate = kwargs.get("candidate_typology")

        cls = classify_typology_from_signals(
            triggered_signals=[str(s) for s in triggered_signals],
            signal_breakdown={str(k): float(v) for k, v in signal_breakdown.items()},
        )
        typology = str(candidate or cls.fraud_typology)

        return {
            "agent": self.name,
            "fraud_typology": typology,
            "typology_confidence": cls.typology_confidence,
            "typology_definition": TYPOLOGY_DEFINITIONS.get(typology, cls.typology_definition),
            "typology_reason": cls.typology_reason,
            "candidate_typologies": cls.candidate_typologies,
            "typology": typology,  # backward compatibility
            "supporting_signals": [code.value for code in counts.keys()],
        }

