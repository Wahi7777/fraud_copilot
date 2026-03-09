from __future__ import annotations

from typing import Any, Dict, List

from models import EvidenceItem, EvidenceSeverity, FraudSignalCode, EnrichedTransactionRecord
from services.graph_builder import GraphBuilder
from .base_agent import BaseInvestigationAgent


class NetworkAgent(BaseInvestigationAgent):
    """
    Analyses network relationships and fraud-linked nodes using the graph.
    """

    def __init__(self, graph_builder: GraphBuilder) -> None:
        super().__init__(
            name="NetworkAgent",
            description="Inspects account graph for fraud-linked and mule-like patterns.",
            domain="network_analysis",
        )
        self.graph_builder = graph_builder

    def run(self, tx: EnrichedTransactionRecord, **kwargs: Any) -> Dict[str, Any]:
        evidence: List[EvidenceItem] = []

        degree = self.graph_builder.get_beneficiary_degree(tx.beneficiary_id)
        if degree >= 5:
            severity = EvidenceSeverity.HIGH if degree < 10 else EvidenceSeverity.CRITICAL
            evidence.append(
                EvidenceItem(
                    source_agent=self.name,
                    signal_code=FraudSignalCode.MULE_PATTERN,
                    severity=severity,
                    summary="Beneficiary node has high inbound degree, consistent with mule behaviour.",
                    details={
                        "beneficiary_id": tx.beneficiary_id,
                        "in_degree": degree,
                    },
                )
            )

        if self.graph_builder.has_path_to_flagged_node(tx.account_id):
            evidence.append(
                EvidenceItem(
                    source_agent=self.name,
                    signal_code=FraudSignalCode.LINKED_FRAUD_ACCOUNT,
                    severity=EvidenceSeverity.HIGH,
                    summary="Origin account has direct or indirect path to fraud-flagged node.",
                    details={
                        "account_id": tx.account_id,
                    },
                )
            )

        in_cycle = self.graph_builder.detect_simple_circular_pattern(tx.account_id)
        if in_cycle:
            evidence.append(
                EvidenceItem(
                    source_agent=self.name,
                    signal_code=FraudSignalCode.MULE_PATTERN,
                    severity=EvidenceSeverity.MEDIUM,
                    summary="Account participates in a small circular transaction pattern.",
                    details={
                        "account_id": tx.account_id,
                        "pattern": "small_cycle",
                    },
                )
            )

        return {
            "agent": self.name,
            "evidence_items": self._serialize_evidence(evidence),
        }

