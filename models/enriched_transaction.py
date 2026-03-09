from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EnrichedTransactionRecord(BaseModel):
    """
    Deterministic, feature-complete representation of a transaction under investigation.

    This model is populated by the `investigation_feature_builder` service using data
    from the underlying CSV datasets (PaySim, IEEE, etc.).
    """

    transaction_id: str = Field(..., description="Unique identifier for the transaction under investigation.")
    account_id: str = Field(..., description="Originating customer account identifier.")
    beneficiary_id: str = Field(..., description="Destination / beneficiary account identifier.")
    amount: float = Field(..., ge=0.0, description="Transaction amount in the configured currency.")
    transaction_type: str = Field(..., description="Transaction type/category (e.g. CASH_OUT, TRANSFER).")
    timestamp: datetime = Field(..., description="Timestamp of the transaction event.")

    # Device / channel
    device_id: str | None = Field(
        default=None, description="Stable device or fingerprint identifier associated with the session."
    )
    device_type: str | None = Field(
        default=None,
        description="High-level device type classification (e.g. MOBILE, DESKTOP).",
    )
    ip_address: str | None = Field(
        default=None,
        description="Synthetic or real IP address string associated with the transaction.",
    )
    country: str | None = Field(
        default=None, description="ISO country code inferred from device/IP/geo enrichment."
    )
    # Age / tenure signals
    account_age_days: int | None = Field(
        default=None,
        ge=0,
        description="Age of the customer account in days at transaction time.",
    )
    beneficiary_age_days: int | None = Field(
        default=None,
        ge=0,
        description="Age (or first-seen window) of the beneficiary account in days.",
    )
    # Velocity and behavioural features
    transaction_velocity_10min: float | None = Field(
        default=None,
        ge=0.0,
        description="Number of outbound transactions in the preceding 10 minutes window.",
    )
    transaction_velocity_1hr: float | None = Field(
        default=None,
        ge=0.0,
        description="Number of outbound transactions in the preceding 1 hour window.",
    )
    merchant_category: str | None = Field(
        default=None, description="Merchant category code or coarse-grained category label."
    )
    # Risk scores and derived signals
    device_risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Deterministic 0-1 risk score derived from device and IP reputation signals.",
    )
    email_domain_risk: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Deterministic 0-1 heuristic risk score for the customer's email domain.",
    )
    geo_distance_jump_km: float | None = Field(
        default=None,
        ge=0.0,
        description="Approximate geo-distance jump in km from the previous login/transaction.",
    )
    impossible_travel_flag: bool | None = Field(
        default=None,
        description="True when the geo-distance jump implies impossible travel given elapsed time.",
    )

