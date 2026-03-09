from models import EvidenceItem, EvidenceSeverity, FraudSignalCode
from services.scoring_engine import ScoringEngine


def _ev(signal: FraudSignalCode) -> EvidenceItem:
    return EvidenceItem(
        source_agent="TestAgent",
        signal_code=signal,
        severity=EvidenceSeverity.HIGH,
        summary=f"Signal {signal.value}",
        details=None,
    )


def test_single_signal_scoring():
    engine = ScoringEngine()
    result = engine.score([_ev(FraudSignalCode.HIGH_VELOCITY)])

    assert result.risk_points == 25
    assert result.risk_score == 0.25
    assert result.risk_band == "low"
    assert result.confidence_band == "LOW"
    assert result.signal_weights[FraudSignalCode.HIGH_VELOCITY] == 25


def test_multi_signal_scoring():
    engine = ScoringEngine()
    evidence = [
        _ev(FraudSignalCode.HIGH_VELOCITY),
        _ev(FraudSignalCode.NEW_DEVICE),
        _ev(FraudSignalCode.LINKED_FRAUD_ACCOUNT),
    ]
    result = engine.score(evidence)

    # 25 + 20 + 30 = 75 points
    assert result.risk_points == 75
    assert result.risk_score == 0.75
    assert result.risk_band == "high"
    assert result.confidence_band == "HIGH"


def test_score_cap_and_confidence_derivation():
    engine = ScoringEngine()
    evidence = [
        _ev(FraudSignalCode.HIGH_VELOCITY),
        _ev(FraudSignalCode.NEW_DEVICE),
        _ev(FraudSignalCode.NEW_COUNTRY),
        _ev(FraudSignalCode.LINKED_FRAUD_ACCOUNT),
        _ev(FraudSignalCode.IMPOSSIBLE_TRAVEL),
        _ev(FraudSignalCode.MULE_PATTERN),
    ]

    result = engine.score(evidence)

    # Sum of all configured weights = 120 => capped to 0.99.
    assert result.risk_points == 120
    assert result.risk_score == 0.99
    assert result.risk_band == "high"
    assert result.confidence_band == "HIGH"

