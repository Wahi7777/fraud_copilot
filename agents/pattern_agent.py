from __future__ import annotations

from typing import Any, Dict, List

from models import (
    EnrichedTransactionRecord,
    EvidenceItem,
    EvidenceSeverity,
    FraudSignalCode,
)
from .base_agent import BaseInvestigationAgent


class PatternAgent(BaseInvestigationAgent):
    """
    Analyses transaction velocity and spike patterns.
    """

    def __init__(self) -> None:
        super().__init__(
            name="PatternAgent",
            description="Detects velocity anomalies and transaction spikes.",
            domain="transaction_patterns",
        )

    def run(self, tx: EnrichedTransactionRecord, **kwargs: Any) -> Dict[str, Any]:
        evidence: List[EvidenceItem] = []

        v10 = tx.transaction_velocity_10min or 0.0
        v1h = tx.transaction_velocity_1hr or 0.0
        amount = tx.amount

        if v10 >= 3 or v1h >= 5:
            severity = EvidenceSeverity.CRITICAL if v10 >= 5 or v1h >= 10 else EvidenceSeverity.HIGH
            evidence.append(
                EvidenceItem(
                    source_agent=self.name,
                    signal_code=FraudSignalCode.HIGH_VELOCITY,
                    severity=severity,
                    summary="High outbound transaction velocity detected.",
                    details={
                        "velocity_10min": v10,
                        "velocity_1hr": v1h,
                    },
                )
            )

        median_like_amount = 500.0
        if amount >= 4 * median_like_amount:
            severity = EvidenceSeverity.HIGH if amount < 10 * median_like_amount else EvidenceSeverity.CRITICAL
            evidence.append(
                EvidenceItem(
                    source_agent=self.name,
                    signal_code=FraudSignalCode.MULE_PATTERN,
                    severity=severity,
                    summary="Large spike in outbound transfer amount.",
                    details={
                        "amount": amount,
                        "threshold": 4 * median_like_amount,
                    },
                )
            )

        return {
            "agent": self.name,
            "evidence_items": self._serialize_evidence(evidence),
        }

