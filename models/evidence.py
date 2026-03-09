from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EvidenceSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FraudSignalCode(str, Enum):
    HIGH_VELOCITY = "HIGH_VELOCITY"
    NEW_DEVICE = "NEW_DEVICE"
    NEW_COUNTRY = "NEW_COUNTRY"
    LINKED_FRAUD_ACCOUNT = "LINKED_FRAUD_ACCOUNT"
    IMPOSSIBLE_TRAVEL = "IMPOSSIBLE_TRAVEL"
    MULE_PATTERN = "MULE_PATTERN"


class EvidenceItem(BaseModel):
    """
    Normalised evidence unit recorded in the Evidence Registry.

    All CrewAI agents emit lists of `EvidenceItem` instances, which are later
    consumed by the scoring engine and the dashboard.
    """

    source_agent: str = Field(..., description="Logical name of the producing agent (e.g. PatternAgent).")
    signal_code: FraudSignalCode = Field(..., description="Stable fraud signal code.")
    severity: EvidenceSeverity = Field(..., description="Normalised severity for this signal.")
    summary: str = Field(..., description="Short human-readable explanation of the signal.")
    details: dict | None = Field(
        default=None,
        description="Structured JSON payload with additional context (amounts, device ids, graph metrics, etc.).",
    )

