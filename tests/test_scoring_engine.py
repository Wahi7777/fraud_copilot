from datetime import datetime, timedelta

from models import EnrichedTransactionRecord
from services.scoring_engine import ScoringEngine


def _tx(idx: int, amount: float, ts: datetime) -> EnrichedTransactionRecord:
    return EnrichedTransactionRecord(
        transaction_id=f"T-{idx}",
        account_id="ACC-1",
        beneficiary_id=f"BEN-{idx % 4}",
        amount=amount,
        transaction_type="TRANSFER" if idx % 3 else "CASH_OUT",
        timestamp=ts,
        device_id=f"DEV-{idx % 3}",
        device_type="MOBILE",
        ip_address=f"10.0.0.{idx % 12 + 1}",
        country="US",
        account_age_days=120,
        beneficiary_age_days=14,
        transaction_velocity_10min=float((idx % 5) + 1),
        transaction_velocity_1hr=float((idx % 8) + 1),
        merchant_category="MONEY_TRANSFER",
        device_risk_score=0.4,
        email_domain_risk=0.3,
        geo_distance_jump_km=float((idx % 6) * 200),
        impossible_travel_flag=idx % 9 == 0,
    )


def test_synthetic_signals_are_deterministic():
    engine = ScoringEngine()
    base = datetime(2026, 3, 9, 12, 0, 0)
    current = _tx(50, 420.0, base)
    history = [_tx(i, 100 + i * 2, base - timedelta(minutes=5 * i)) for i in range(1, 40)]
    a = engine.score_transaction(
        current,
        account_history=history,
        beneficiary_incoming_count_24h=3,
        beneficiary_degree=2,
        has_sender_path_to_flagged=False,
        has_beneficiary_path_to_flagged=False,
        has_historical_fraud_link=False,
    )
    b = engine.score_transaction(
        current,
        account_history=history,
        beneficiary_incoming_count_24h=3,
        beneficiary_degree=2,
        has_sender_path_to_flagged=False,
        has_beneficiary_path_to_flagged=False,
        has_historical_fraud_link=False,
    )
    assert a.risk_score == b.risk_score
    assert a.signal_breakdown == b.signal_breakdown
    assert a.synthetic_telemetry_signals == b.synthetic_telemetry_signals


def test_low_risk_transaction_does_not_become_extreme():
    engine = ScoringEngine()
    base = datetime(2026, 3, 9, 15, 0, 0)
    current = _tx(1, 28.0, base)
    current.transaction_velocity_1hr = 1.0
    current.transaction_velocity_10min = 0.0
    history = [_tx(i, 22.0 + (i % 3), base - timedelta(hours=i)) for i in range(1, 30)]
    result = engine.score_transaction(
        current,
        account_history=history,
        beneficiary_incoming_count_24h=1,
        beneficiary_degree=1,
        has_sender_path_to_flagged=False,
        has_beneficiary_path_to_flagged=False,
        has_historical_fraud_link=False,
    )
    assert 0.0 <= result.risk_score < 0.7


def test_batch_distribution_contains_clear_escalate_decline():
    engine = ScoringEngine()
    base = datetime(2026, 3, 9, 18, 0, 0)
    counts = {"Clear": 0, "Escalate": 0, "Decline": 0}
    for i in range(90):
        tx = _tx(i + 100, amount=40 + (i * 12), ts=base - timedelta(minutes=i))
        history = [_tx(j, 35 + (j % 7) * 8, tx.timestamp - timedelta(minutes=6 * (j + 1))) for j in range(25)]
        incoming = i % 18
        degree = i % 12
        sender_flagged = i % 17 == 0
        ben_flagged = i % 13 == 0
        hist_link = i % 19 == 0
        result = engine.score_transaction(
            tx,
            account_history=history,
            beneficiary_incoming_count_24h=incoming,
            beneficiary_degree=degree,
            has_sender_path_to_flagged=sender_flagged,
            has_beneficiary_path_to_flagged=ben_flagged,
            has_historical_fraud_link=hist_link,
        )
        if result.risk_score < 0.35:
            counts["Clear"] += 1
        elif result.risk_score < 0.70:
            counts["Escalate"] += 1
        else:
            counts["Decline"] += 1
    assert counts["Clear"] > 0
    assert counts["Escalate"] > 0
    assert counts["Decline"] > 0


def test_expected_distribution_band_is_realistic():
    engine = ScoringEngine()
    base = datetime(2026, 3, 9, 20, 0, 0)
    buckets = {"Clear": 0, "Escalate": 0, "Decline": 0}
    # 50 lower-risk scenarios
    for i in range(50):
        tx = _tx(i + 1000, amount=30 + (i % 10), ts=base - timedelta(minutes=i))
        tx.transaction_velocity_1hr = 1.0
        tx.transaction_velocity_10min = 0.0
        tx.geo_distance_jump_km = 0.0
        history = [_tx(j + 2000, 28 + (j % 4), tx.timestamp - timedelta(hours=j + 1)) for j in range(20)]
        out = engine.score_transaction(
            tx,
            account_history=history,
            beneficiary_incoming_count_24h=1,
            beneficiary_degree=1,
            has_sender_path_to_flagged=False,
            has_beneficiary_path_to_flagged=False,
            has_historical_fraud_link=False,
        )
        buckets["Clear" if out.risk_score < 0.35 else "Escalate" if out.risk_score < 0.7 else "Decline"] += 1
    # 35 moderate-risk scenarios
    for i in range(35):
        tx = _tx(i + 3000, amount=1100 + (i * 20), ts=base - timedelta(minutes=90 + i))
        tx.transaction_velocity_1hr = 6.0
        tx.transaction_velocity_10min = 2.5
        tx.geo_distance_jump_km = 1200.0
        history = [_tx(j + 4000, 180 + (j % 9) * 35, tx.timestamp - timedelta(minutes=14 * (j + 1))) for j in range(25)]
        out = engine.score_transaction(
            tx,
            account_history=history,
            beneficiary_incoming_count_24h=4,
            beneficiary_degree=3,
            has_sender_path_to_flagged=False,
            has_beneficiary_path_to_flagged=False,
            has_historical_fraud_link=False,
        )
        buckets["Clear" if out.risk_score < 0.35 else "Escalate" if out.risk_score < 0.7 else "Decline"] += 1
    # 15 high-risk scenarios
    for i in range(15):
        tx = _tx(i + 5000, amount=5000 + (i * 100), ts=base - timedelta(minutes=200 + i))
        tx.transaction_velocity_1hr = 10.0
        tx.transaction_velocity_10min = 5.0
        tx.geo_distance_jump_km = 4000.0
        tx.impossible_travel_flag = True
        history = [_tx(j + 6000, 400 + (j % 11) * 60, tx.timestamp - timedelta(minutes=8 * (j + 1))) for j in range(30)]
        out = engine.score_transaction(
            tx,
            account_history=history,
            beneficiary_incoming_count_24h=14,
            beneficiary_degree=12,
            has_sender_path_to_flagged=True,
            has_beneficiary_path_to_flagged=True,
            has_historical_fraud_link=True,
        )
        buckets["Clear" if out.risk_score < 0.35 else "Escalate" if out.risk_score < 0.7 else "Decline"] += 1
    total = sum(buckets.values())
    clear_pct = buckets["Clear"] / total
    esc_pct = buckets["Escalate"] / total
    dec_pct = buckets["Decline"] / total
    assert 0.50 <= clear_pct <= 0.70
    assert 0.20 <= esc_pct <= 0.40
    assert 0.05 <= dec_pct <= 0.15

