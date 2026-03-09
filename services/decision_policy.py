from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DecisionPolicyResult:
    risk_score: float
    recommendation: str
    workflow_status: str
    decision_confidence: float
    decision_reason: str
    triggered_signals: List[str]
    agent_votes: Dict[str, str]
    override_applied: bool
    override_reason: str | None
    base_recommendation: str


class DecisionPolicyEngine:
    """
    Deterministic, agent-aware final decision policy.

    Diagnosis note:
    Earlier queue behavior was too coarse because recommendation was inferred
    from risk bands in UI paths and workflow-like placeholders (for example
    Pending/Open) leaked into recommendation display. This policy explicitly
    separates recommendation from workflow status and synthesizes score,
    signals, confidence, and agent votes.
    """

    def evaluate(
        self,
        *,
        risk_score: float,
        signals: Dict[str, Dict[str, str]],
        network: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        agent_trace: List[Dict[str, str]] | None = None,
    ) -> DecisionPolicyResult:
        score = min(max(float(risk_score), 0.0), 0.99)
        triggered_signals = sorted(
            [
                key
                for key, payload in (signals or {}).items()
                if isinstance(payload, dict)
                and str(payload.get("severity", "")).lower() in {"high", "medium"}
            ]
        )
        signal_set = set(triggered_signals)

        vote_tx = self._vote_transaction(signal_set)
        vote_beh = self._vote_behavior(signal_set, evidence)
        vote_net = self._vote_network(network)
        vote_policy = self._vote_policy(score)
        agent_votes = {
            "transaction_agent": vote_tx,
            "behavior_agent": vote_beh,
            "network_agent": vote_net,
            "policy_agent": vote_policy,
        }

        base_recommendation = self._recommendation_from_score(score)
        confidence = self._confidence(score=score, votes=agent_votes, triggered_signals=triggered_signals)

        strong_confirming = self._strong_confirming(signal_set, network, evidence)
        exonerating = self._exonerating(signal_set, network, evidence)
        conflicting_votes = len({vote_tx, vote_beh, vote_net, vote_policy}) >= 3

        recommendation = base_recommendation
        override_applied = False
        override_reason: str | None = None

        if base_recommendation == "Decline" and not strong_confirming:
            recommendation = "Escalate"
            override_applied = True
            override_reason = "downgraded_decline_due_to_missing_strong_confirmation"
        if base_recommendation == "Decline" and confidence < 0.55:
            recommendation = "Escalate"
            override_applied = True
            override_reason = "downgraded_decline_due_to_low_confidence"
        if base_recommendation in {"Clear", "Escalate"} and strong_confirming and (
            confidence >= 0.55 or bool(network.get("fraud_link"))
        ):
            recommendation = "Decline"
            override_applied = True
            override_reason = "upgraded_due_to_strong_confirming_signals"
        if base_recommendation in {"Decline", "Escalate"} and exonerating and confidence >= 0.65:
            recommendation = "Clear" if score < 0.45 else "Escalate"
            override_applied = True
            override_reason = "downgraded_due_to_exonerating_context"
        if (conflicting_votes or confidence < 0.45) and not (strong_confirming and bool(network.get("fraud_link"))):
            recommendation = "Escalate"
            override_applied = True
            override_reason = "forced_escalate_due_to_conflict_or_low_confidence"

        if recommendation == "Clear":
            workflow_status = "Closed"
        elif recommendation in {"Escalate", "Decline"}:
            workflow_status = "In Review"
        else:
            workflow_status = "In Review"

        reason = (
            f"base={base_recommendation}, final={recommendation}, score={score:.2f}, "
            f"signals={','.join(triggered_signals) if triggered_signals else 'none'}, "
            f"votes={agent_votes}, override={override_reason or 'none'}"
        )

        return DecisionPolicyResult(
            risk_score=score,
            recommendation=recommendation,
            workflow_status=workflow_status,
            decision_confidence=confidence,
            decision_reason=reason,
            triggered_signals=triggered_signals,
            agent_votes=agent_votes,
            override_applied=override_applied,
            override_reason=override_reason,
            base_recommendation=base_recommendation,
        )

    @staticmethod
    def _recommendation_from_score(score: float) -> str:
        if score < 0.35:
            return "Clear"
        if score < 0.70:
            return "Escalate"
        return "Decline"

    @staticmethod
    def _vote_transaction(signals: set[str]) -> str:
        if {"velocity", "transfer_amount", "balance_drain"} & signals:
            if "velocity" in signals and ("transfer_amount" in signals or "balance_drain" in signals):
                return "high_risk"
            return "medium_risk"
        return "low_risk"

    @staticmethod
    def _vote_behavior(signals: set[str], evidence: List[Dict[str, Any]]) -> str:
        text_blob = " ".join(f"{e.get('summary', '')} {e.get('details', '')}" for e in evidence).lower()
        risky_device = "device_reuse" in signals
        risky_geo = "geo_anomaly" in signals
        proxy = any(k in text_blob for k in ["proxy", "vpn", "tor", "anonym", "masked ip"])
        if risky_device and risky_geo and proxy:
            return "high_risk"
        if risky_device or risky_geo or proxy:
            return "medium_risk"
        if any(k in text_blob for k in ["known device", "trusted device"]):
            return "low_risk"
        return "review"

    @staticmethod
    def _vote_network(network: Dict[str, Any]) -> str:
        if network.get("fraud_link"):
            return "high_risk"
        if network.get("suspected_network_risk"):
            return "review"
        return "low_risk"

    @staticmethod
    def _vote_policy(score: float) -> str:
        if score >= 0.70:
            return "high_risk"
        if score >= 0.35:
            return "review"
        return "low_risk"

    @staticmethod
    def _confidence(score: float, votes: Dict[str, str], triggered_signals: List[str]) -> float:
        counts = {"high_risk": 0, "medium_risk": 0, "review": 0, "low_risk": 0}
        for v in votes.values():
            if v in counts:
                counts[v] += 1
        agreement = max(counts.values()) / max(len(votes), 1)
        signal_strength = min(len(triggered_signals) / 4.0, 1.0)
        confidence = 0.35 + (0.35 * agreement) + (0.20 * signal_strength) + (0.10 * score)
        return min(max(round(confidence, 3), 0.0), 1.0)

    @staticmethod
    def _strong_confirming(signals: set[str], network: Dict[str, Any], evidence: List[Dict[str, Any]]) -> bool:
        text_blob = " ".join(f"{e.get('summary', '')} {e.get('details', '')}" for e in evidence).lower()
        patterns = [
            {"velocity", "device_reuse"},
            {"device_reuse", "geo_anomaly"},
            {"velocity", "balance_drain"},
            {"velocity", "transfer_amount", "device_reuse"},
        ]
        combo_match = any(p.issubset(signals) for p in patterns)
        repeated_failed_then_success = "failed attempts" in text_blob and "success" in text_blob
        takeover = any(k in text_blob for k in ["account takeover", "ato"])
        return bool(network.get("fraud_link") or combo_match or repeated_failed_then_success or takeover)

    @staticmethod
    def _exonerating(signals: set[str], network: Dict[str, Any], evidence: List[Dict[str, Any]]) -> bool:
        if network.get("fraud_link") or network.get("suspected_network_risk"):
            return False
        text_blob = " ".join(f"{e.get('summary', '')} {e.get('details', '')}" for e in evidence).lower()
        known_device = any(k in text_blob for k in ["known device", "trusted device"])
        known_beneficiary = any(k in text_blob for k in ["known beneficiary", "known payee", "trusted payee"])
        normal_geo = any(k in text_blob for k in ["normal geography", "expected location", "usual location"])
        weak_signal_set = signals.issubset({"transfer_amount"}) or len(signals) <= 1
        return (known_device or known_beneficiary or normal_geo) and weak_signal_set
