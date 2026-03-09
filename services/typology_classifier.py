from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


CONTROLLED_TYPOLOGIES = {
    "Potential Mule Transfer",
    "Velocity Fraud",
    "Account Takeover",
    "Beneficiary Risk",
    "Transaction Anomaly",
    "Structured Cash-Out Pattern",
    "Unknown / Mixed Pattern",
}

TYPOLOGY_DEFINITIONS: Dict[str, str] = {
    "Potential Mule Transfer": "Funds appear to be routed through intermediary beneficiary accounts in a pattern consistent with mule-network behavior.",
    "Velocity Fraud": "Rapid transaction activity in a short window suggests unusually high transfer velocity inconsistent with typical account behavior.",
    "Account Takeover": "Device, network, and location telemetry indicate possible unauthorized account access and control.",
    "Beneficiary Risk": "The destination beneficiary shows elevated risk based on historical or network-linked fraud indicators.",
    "Transaction Anomaly": "The transaction deviates from normal account behavior but lacks a stronger, specific fraud typology pattern.",
    "Structured Cash-Out Pattern": "Transfer and cash-out sequence suggests staged movement of funds for rapid extraction.",
    "Unknown / Mixed Pattern": "Suspicious transaction pattern detected that deviates from normal account behavior.",
}


@dataclass
class TypologyClassification:
    fraud_typology: str
    typology_confidence: float
    typology_definition: str
    typology_reason: List[str]
    candidate_typologies: List[str]


def classify_typology_from_signals(
    *,
    triggered_signals: List[str],
    signal_breakdown: Dict[str, float],
) -> TypologyClassification:
    signals = set(triggered_signals or [])
    reasons: List[str] = []
    candidates: List[str] = []

    mule = {"risky_beneficiary_link", "beneficiary_incoming_volume", "fanout_pattern"}
    velocity = {"tx_burst_1h", "unique_beneficiaries_1h"}
    ato = {"new_device_indicator", "device_reuse_risk", "geo_anomaly", "ip_risk"}

    if mule.issubset(signals):
        candidates.append("Potential Mule Transfer")
        reasons.append("Beneficiary-risk linkage, incoming volume, and fanout pattern are all present.")
    if velocity.issubset(signals):
        candidates.append("Velocity Fraud")
        reasons.append("Burst activity and multiple unique beneficiaries in a short window are present.")
    if len(ato & signals) >= 3:
        candidates.append("Account Takeover")
        reasons.append("Multiple device/geo/IP anomaly indicators suggest potential account takeover.")
    if "transfer_cashout_pattern" in signals:
        candidates.append("Structured Cash-Out Pattern")
        reasons.append("Transfer followed by cash-out pattern is explicitly triggered.")
    if "risky_beneficiary_link" in signals or "historical_fraud_link" in signals:
        candidates.append("Beneficiary Risk")
        reasons.append("Beneficiary or historical fraud linkage indicators are present.")

    strong_primary = [c for c in candidates if c in {"Potential Mule Transfer", "Velocity Fraud", "Account Takeover", "Structured Cash-Out Pattern"}]
    if not candidates:
        weak = sorted(signal_breakdown.items(), key=lambda x: x[1], reverse=True)
        top = weak[0][1] if weak else 0.0
        if len(signals) <= 2 and top < 0.06:
            typology = "Transaction Anomaly"
            reasons.append("Only isolated weak signals are present.")
            confidence = 0.62
        else:
            typology = "Unknown / Mixed Pattern"
            reasons.append("Signal pattern is mixed and does not strongly match one typology.")
            confidence = 0.55
    elif len(set(strong_primary)) > 1:
        # Conflicting strong primary candidates should surface as mixed.
        typology = "Unknown / Mixed Pattern"
        confidence = 0.58
    else:
        priority = [
            "Potential Mule Transfer",
            "Structured Cash-Out Pattern",
            "Account Takeover",
            "Velocity Fraud",
            "Beneficiary Risk",
            "Transaction Anomaly",
        ]
        uniq = set(candidates)
        typology = next((name for name in priority if name in uniq), "Unknown / Mixed Pattern")
        confidence = 0.82

    if typology not in CONTROLLED_TYPOLOGIES:
        typology = "Unknown / Mixed Pattern"
    return TypologyClassification(
        fraud_typology=typology,
        typology_confidence=confidence,
        typology_definition=TYPOLOGY_DEFINITIONS.get(typology, TYPOLOGY_DEFINITIONS["Unknown / Mixed Pattern"]),
        typology_reason=reasons[:4] if reasons else ["Signal pattern indicates suspicious activity requiring analyst review."],
        candidate_typologies=sorted(set(candidates)),
    )
