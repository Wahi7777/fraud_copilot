from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Tuple

from models import EnrichedTransactionRecord, EvidenceItem, FraudSignalCode


@dataclass
class ScoreBreakdown:
    """
    Auditable scoring result for an investigation.
    """

    risk_points: float
    risk_score: float
    confidence: float
    confidence_band: str
    risk_band: str
    signal_weights: Dict[FraudSignalCode, int]
    signal_breakdown: Dict[str, float]
    triggered_signals: List[str]
    synthetic_telemetry_signals: List[str]
    signal_reasons: Dict[str, str]


class ScoringEngine:
    """
    Deterministic, explainable fraud scoring engine.

    Weights are derived from the PRD with additional values for auxiliary
    signals so that the total risk can exceed 100 points (before capping).
    """

    WEIGHTS: Dict[FraudSignalCode, int] = {
        FraudSignalCode.HIGH_VELOCITY: 14,
        FraudSignalCode.NEW_DEVICE: 9,
        FraudSignalCode.NEW_COUNTRY: 8,
        FraudSignalCode.LINKED_FRAUD_ACCOUNT: 18,
        FraudSignalCode.IMPOSSIBLE_TRAVEL: 12,
        FraudSignalCode.MULE_PATTERN: 15,
    }
    SIGNAL_MAX: Dict[str, float] = {
        "amount_spike": 0.10,
        "balance_drain": 0.07,
        "account_emptying": 0.06,
        "tx_burst_1h": 0.08,
        "unique_beneficiaries_1h": 0.06,
        "beneficiary_incoming_volume": 0.07,
        "fanout_pattern": 0.06,
        "transfer_cashout_pattern": 0.08,
        "risky_beneficiary_link": 0.10,
        "historical_fraud_link": 0.10,
        "device_reuse_risk": 0.05,
        "new_device_indicator": 0.04,
        "geo_anomaly": 0.05,
        "ip_risk": 0.04,
        "time_of_day_deviation": 0.04,
    }

    def score(self, evidence: Iterable[EvidenceItem]) -> ScoreBreakdown:
        """
        Compute an auditable score from a collection of evidence items.

        The engine:
        - Collapses evidence by unique FraudSignalCode
        - Applies deterministic weights
        - Caps the final score at 0.99 as per PRD
        - Derives human-readable confidence and risk bands
        """
        seen_codes: set[FraudSignalCode] = set()
        contributions: Dict[FraudSignalCode, int] = {}

        for item in evidence:
            code = item.signal_code
            if code in seen_codes:
                continue
            seen_codes.add(code)

            weight = self.WEIGHTS.get(code, 0)
            if weight > 0:
                contributions[code] = weight

        risk_points = float(sum(contributions.values()))
        risk_score = min(risk_points / 100.0, 0.99)

        confidence = risk_score  # direct mapping keeps interpretation simple
        confidence_band = self._confidence_band(confidence)
        risk_band = self._risk_band(risk_score)

        return ScoreBreakdown(
            risk_points=risk_points,
            risk_score=risk_score,
            confidence=confidence,
            confidence_band=confidence_band,
            risk_band=risk_band,
            signal_weights=contributions,
            signal_breakdown={k.value: round(v / 100.0, 4) for k, v in contributions.items()},
            triggered_signals=[k.value for k in contributions.keys()],
            synthetic_telemetry_signals=[],
            signal_reasons={k.value: f"Triggered weight {v}" for k, v in contributions.items()},
        )

    def score_transaction(
        self,
        tx: EnrichedTransactionRecord,
        *,
        account_history: List[EnrichedTransactionRecord],
        beneficiary_incoming_count_24h: int,
        beneficiary_degree: int,
        has_sender_path_to_flagged: bool,
        has_beneficiary_path_to_flagged: bool,
        has_historical_fraud_link: bool,
    ) -> ScoreBreakdown:
        prior = [h for h in account_history if h.timestamp < tx.timestamp and h.transaction_id != tx.transaction_id]
        prior_1h = [h for h in prior if h.timestamp >= tx.timestamp - timedelta(hours=1)]
        prior_24h = [h for h in prior if h.timestamp >= tx.timestamp - timedelta(hours=24)]
        avg_amount = sum(h.amount for h in prior[-50:]) / max(len(prior[-50:]), 1)
        amount = max(float(tx.amount or 0.0), 0.0)

        signal_values: Dict[str, float] = {}
        signal_reasons: Dict[str, str] = {}
        triggered: List[str] = []
        synthetic = {
            "device_reuse_risk",
            "new_device_indicator",
            "geo_anomaly",
            "ip_risk",
            "time_of_day_deviation",
        }

        def add_signal(name: str, raw_score_0_1: float, threshold: float, reason: str) -> None:
            raw = min(max(raw_score_0_1, 0.0), 1.0)
            contribution = round((raw**1.7) * self.SIGNAL_MAX[name], 4)
            signal_values[name] = contribution
            signal_reasons[name] = reason
            if raw >= threshold:
                triggered.append(name)

        # 1) amount_spike
        ratio = amount / max(avg_amount, 1.0)
        add_signal("amount_spike", min(max((ratio - 1.0) / 5.0, 0.0), 1.0), 0.45, f"Amount ratio vs baseline={ratio:.2f}")
        # 2) balance_drain (proxy denominator from history due to limited telemetry)
        proxy_balance = max(avg_amount * 3.0, amount, 1.0)
        drain_ratio = amount / proxy_balance
        add_signal("balance_drain", drain_ratio, 0.60, f"Drain proxy ratio={drain_ratio:.2f}")
        # 3) account_emptying
        add_signal("account_emptying", 1.0 if drain_ratio >= 0.9 else 0.0, 0.50, "Potential account emptying pattern")
        # 4) tx_burst_1h
        burst_1h = float(tx.transaction_velocity_1hr or len(prior_1h))
        add_signal("tx_burst_1h", min(burst_1h / 8.0, 1.0), 0.50, f"Outbound burst in 1h={burst_1h:.0f}")
        # 5) unique_beneficiaries_1h
        uniq_benef_1h = len({h.beneficiary_id for h in prior_1h})
        add_signal("unique_beneficiaries_1h", min(uniq_benef_1h / 6.0, 1.0), 0.50, f"Unique beneficiaries in 1h={uniq_benef_1h}")
        # 6) beneficiary_incoming_volume
        add_signal(
            "beneficiary_incoming_volume",
            min(float(beneficiary_incoming_count_24h) / 15.0, 1.0),
            0.45,
            f"Beneficiary incoming tx count 24h={beneficiary_incoming_count_24h}",
        )
        # 7) fanout_pattern
        fanout = len({h.beneficiary_id for h in prior_24h})
        add_signal("fanout_pattern", min(fanout / 10.0, 1.0), 0.45, f"Sender fanout in 24h={fanout}")
        # 8) transfer_cashout_pattern
        has_transfer = any(str(h.transaction_type).upper() == "TRANSFER" for h in prior_1h) or str(tx.transaction_type).upper() == "TRANSFER"
        has_cashout = any(str(h.transaction_type).upper() == "CASH_OUT" for h in prior_1h) or str(tx.transaction_type).upper() == "CASH_OUT"
        add_signal(
            "transfer_cashout_pattern",
            1.0 if has_transfer and has_cashout else 0.0,
            0.5,
            "Transfer and cash-out pattern in short window",
        )
        # 9) risky_beneficiary_link
        beneficiary_link_score = 1.0 if has_beneficiary_path_to_flagged else min(beneficiary_degree / 10.0, 1.0)
        add_signal("risky_beneficiary_link", beneficiary_link_score, 0.5, f"Beneficiary degree={beneficiary_degree}, path_to_flagged={has_beneficiary_path_to_flagged}")
        # 10) historical_fraud_link
        add_signal(
            "historical_fraud_link",
            1.0 if (has_historical_fraud_link or has_sender_path_to_flagged) else 0.0,
            0.5,
            f"Historical fraud linkage sender_or_graph={has_historical_fraud_link or has_sender_path_to_flagged}",
        )

        # Deterministic synthetic telemetry
        base_device = str(tx.device_id or tx.account_id or tx.transaction_id)
        base_ip = str(tx.ip_address or tx.account_id or tx.transaction_id)
        # 11) device_reuse_risk
        device_reuse = (abs(hash(f"reuse:{base_device}")) % 100) / 100.0
        add_signal("device_reuse_risk", device_reuse, 0.65, f"Deterministic device reuse score={device_reuse:.2f}")
        # 12) new_device_indicator
        seen_device_before = any((h.device_id and tx.device_id and h.device_id == tx.device_id) for h in prior)
        new_device = 0.0 if seen_device_before else 1.0 if (abs(hash(f"newdev:{tx.account_id}:{base_device}")) % 5 == 0) else 0.35
        add_signal("new_device_indicator", new_device, 0.60, f"Deterministic new-device indicator={new_device:.2f}")
        # 13) geo_anomaly
        geo_raw = float(tx.geo_distance_jump_km or 0.0)
        geo_score = min(geo_raw / 3000.0, 1.0)
        add_signal("geo_anomaly", geo_score, 0.55, f"Geo distance jump km={geo_raw:.0f}")
        # 14) ip_risk
        ip_risk = (abs(hash(f"iprisk:{base_ip}")) % 100) / 100.0
        add_signal("ip_risk", ip_risk, 0.70, f"Deterministic IP risk={ip_risk:.2f}")
        # 15) time_of_day_deviation
        hour = tx.timestamp.hour
        expected_hour = abs(hash(f"hour:{tx.account_id}")) % 24
        dist = min((hour - expected_hour) % 24, (expected_hour - hour) % 24)
        tod_score = min(dist / 12.0, 1.0)
        add_signal("time_of_day_deviation", tod_score, 0.50, f"Hour deviation={dist}h from expected={expected_hour}")

        high_impact = {
            "amount_spike",
            "tx_burst_1h",
            "transfer_cashout_pattern",
            "risky_beneficiary_link",
            "historical_fraud_link",
        }
        high_count = len([s for s in triggered if s in high_impact])
        base_prior = 0.01
        escalation_boost = 0.0
        if high_count >= 3:
            escalation_boost += min(0.06 * (high_count - 2), 0.22)
        if has_beneficiary_path_to_flagged and has_historical_fraud_link:
            escalation_boost += 0.08
        score = min(max(round(sum(signal_values.values()) + base_prior + escalation_boost, 4), 0.0), 0.99)
        confidence = min(max(round(0.42 + 0.45 * score + 0.13 * min(len(triggered) / 6.0, 1.0), 4), 0.0), 1.0)
        risk_band = self._risk_band(score)
        confidence_band = self._confidence_band(confidence)

        return ScoreBreakdown(
            risk_points=round(score * 100.0, 2),
            risk_score=score,
            confidence=confidence,
            confidence_band=confidence_band,
            risk_band=risk_band,
            signal_weights={},
            signal_breakdown=signal_values,
            triggered_signals=sorted(triggered),
            synthetic_telemetry_signals=sorted([s for s in triggered if s in synthetic]),
            signal_reasons=signal_reasons,
        )

    @staticmethod
    def _risk_band(score: float) -> str:
        if score < 0.35:
            return "low"
        if score < 0.7:
            return "medium"
        return "high"

    @staticmethod
    def _confidence_band(confidence: float) -> str:
        if confidence < 0.3:
            return "LOW"
        if confidence < 0.7:
            return "MEDIUM"
        return "HIGH"

