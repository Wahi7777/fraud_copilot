from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

from models import EvidenceItem, FraudSignalCode
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

        typology = "Generic Fraud"
        if (
            counts[FraudSignalCode.HIGH_VELOCITY] > 0
            and (
                counts[FraudSignalCode.NEW_DEVICE] > 0
                or counts[FraudSignalCode.NEW_COUNTRY] > 0
                or counts[FraudSignalCode.IMPOSSIBLE_TRAVEL] > 0
            )
        ):
            typology = "Account Takeover"
        elif counts[FraudSignalCode.MULE_PATTERN] > 0 or counts[FraudSignalCode.LINKED_FRAUD_ACCOUNT] > 0:
            typology = "Mule Network"

        return {
            "agent": self.name,
            "typology": typology,
            "supporting_signals": [code.value for code in counts.keys()],
        }

