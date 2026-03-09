from __future__ import annotations

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field

from .enriched_transaction import EnrichedTransactionRecord
from .evidence import EvidenceItem


class InvestigationStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class InvestigationState(BaseModel):
    """
    Aggregate view of the investigation state for a single alert/transaction.

    This is the primary object flowing through the investigation pipeline and
    exposed via the API layer to the frontend.
    """

    investigation_id: str = Field(..., description="Stable identifier for this investigation instance.")
    transaction_id: str = Field(..., description="Transaction identifier under investigation.")
    account_id: str = Field(..., description="Originating customer account identifier.")
    status: InvestigationStatus = Field(
        default=InvestigationStatus.PENDING,
        description="Lifecycle status of the investigation.",
    )
    base_transaction: EnrichedTransactionRecord = Field(
        ..., description="Feature-complete transaction record used as investigation context."
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items accumulated from CrewAI agents.",
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=0.99,
        description="Deterministic aggregate fraud risk score in range [0, 0.99].",
    )
    typology: str | None = Field(
        default=None,
        description="Primary fraud typology label inferred by TypologyAgent (e.g. Account Takeover).",
    )
    typology_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for assigned fraud typology.",
    )
    typology_definition: str | None = Field(
        default=None,
        description="Analyst-facing definition for the assigned typology.",
    )
    typology_reason: List[str] = Field(
        default_factory=list,
        description="Short rationale bullets supporting typology selection.",
    )
    recommendation: str | None = Field(
        default=None,
        description="DecisionAgent recommendation (e.g. 'Escalate and freeze beneficiary account').",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised confidence value for the overall recommendation.",
    )
    signal_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-signal deterministic contribution breakdown from scoring engine.",
    )
    triggered_signals: List[str] = Field(
        default_factory=list,
        description="List of triggered deterministic fraud signals.",
    )
    synthetic_telemetry_signals: List[str] = Field(
        default_factory=list,
        description="Triggered synthetic telemetry signal names (deterministically generated).",
    )

