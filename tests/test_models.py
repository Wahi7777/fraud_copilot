from datetime import datetime

from models import (
    EnrichedTransactionRecord,
    EvidenceItem,
    EvidenceSeverity,
    FraudSignalCode,
    InvestigationState,
    InvestigationStatus,
)


def _sample_enriched_transaction() -> EnrichedTransactionRecord:
    return EnrichedTransactionRecord(
        transaction_id="TXN-123",
        account_id="ACC-001",
        beneficiary_id="ACC-002",
        amount=4200.0,
        transaction_type="TRANSFER",
        timestamp=datetime.fromisoformat("2026-03-06T14:02:00"),
        device_id="device-xyz",
        country="NG",
        account_age_days=365,
        transaction_velocity_10min=5.0,
        merchant_category="MONEY_TRANSFER",
        device_risk_score=0.8,
    )


def _sample_evidence_item() -> EvidenceItem:
    return EvidenceItem(
        source_agent="PatternAgent",
        signal_code=FraudSignalCode.HIGH_VELOCITY,
        severity=EvidenceSeverity.HIGH,
        summary="5 outbound transfers within 170 seconds.",
        details={"window_seconds": 170, "transaction_count": 5},
    )


def test_enriched_transaction_record_fields_parsed():
    record = _sample_enriched_transaction()
    assert record.transaction_id == "TXN-123"
    assert record.account_id == "ACC-001"
    assert record.amount == 4200.0
    assert record.account_age_days == 365
    assert record.device_risk_score == 0.8
    assert isinstance(record.timestamp, datetime)


def test_evidence_item_uses_enums_and_details():
    evidence = _sample_evidence_item()
    assert evidence.source_agent == "PatternAgent"
    assert evidence.signal_code is FraudSignalCode.HIGH_VELOCITY
    assert evidence.severity is EvidenceSeverity.HIGH
    assert evidence.details is not None
    assert evidence.details["transaction_count"] == 5


def test_investigation_state_defaults_and_nesting():
    base_tx = _sample_enriched_transaction()
    ev = _sample_evidence_item()

    state = InvestigationState(
        investigation_id="INV-1",
        transaction_id=base_tx.transaction_id,
        account_id=base_tx.account_id,
        base_transaction=base_tx,
        evidence=[ev],
    )

    assert state.investigation_id == "INV-1"
    assert state.status is InvestigationStatus.PENDING
    assert state.base_transaction.transaction_id == "TXN-123"
    assert len(state.evidence) == 1
    assert state.risk_score == 0.0
    assert state.typology is None
    assert state.recommendation is None
    assert state.confidence == 0.0

