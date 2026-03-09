"""
Pydantic models for the Fraud Co-Pilot MVP.

Key models:
- EnrichedTransactionRecord
- EvidenceItem
- InvestigationState
"""

from .enriched_transaction import EnrichedTransactionRecord
from .evidence import EvidenceItem, EvidenceSeverity, FraudSignalCode
from .investigation import InvestigationState, InvestigationStatus

__all__ = [
    "EnrichedTransactionRecord",
    "EvidenceItem",
    "EvidenceSeverity",
    "FraudSignalCode",
    "InvestigationState",
    "InvestigationStatus",
]

