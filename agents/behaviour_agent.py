from __future__ import annotations

from typing import Any, Dict, List

from models import (
    EnrichedTransactionRecord,
    EvidenceItem,
    EvidenceSeverity,
    FraudSignalCode,
)
from .base_agent import BaseInvestigationAgent


class BehaviourAgent(BaseInvestigationAgent):
    """
    Analyses device, location and impossible-travel behaviour.
    """

    def __init__(self) -> None:
        super().__init__(
            name="BehaviourAgent",
            description="Detects device, geo and impossible-travel anomalies.",
            domain="behavioural_signals",
        )

    def run(self, tx: EnrichedTransactionRecord, **kwargs: Any) -> Dict[str, Any]:
        evidence: List[EvidenceItem] = []

        device_risk = tx.device_risk_score or 0.0
        account_age = tx.account_age_days or 0
        if device_risk >= 0.7 or account_age < 30:
            severity = EvidenceSeverity.HIGH if device_risk >= 0.8 else EvidenceSeverity.MEDIUM
            evidence.append(
                EvidenceItem(
                    source_agent=self.name,
                    signal_code=FraudSignalCode.NEW_DEVICE,
                    severity=severity,
                    summary="Suspicious or newly-observed device for this account.",
                    details={
                        "device_id": tx.device_id,
                        "device_type": tx.device_type,
                        "device_risk_score": device_risk,
                        "account_age_days": account_age,
                    },
                )
            )

        if (tx.geo_distance_jump_km or 0.0) >= 1000.0:
            severity = EvidenceSeverity.HIGH
            evidence.append(
                EvidenceItem(
                    source_agent=self.name,
                    signal_code=FraudSignalCode.NEW_COUNTRY,
                    severity=severity,
                    summary="Significant geo-location jump detected.",
                    details={
                        "country": tx.country,
                        "geo_distance_jump_km": tx.geo_distance_jump_km,
                    },
                )
            )

        if tx.impossible_travel_flag:
            severity = EvidenceSeverity.CRITICAL
            evidence.append(
                EvidenceItem(
                    source_agent=self.name,
                    signal_code=FraudSignalCode.IMPOSSIBLE_TRAVEL,
                    severity=severity,
                    summary="Impossible travel pattern between recent logins.",
                    details={
                        "geo_distance_jump_km": tx.geo_distance_jump_km,
                        "impossible_travel": True,
                    },
                )
            )

        return {
            "agent": self.name,
            "evidence_items": self._serialize_evidence(evidence),
        }

