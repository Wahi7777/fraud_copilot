from pathlib import Path

import pytest

from models import EnrichedTransactionRecord
from services.data_repository import DataRepository, DataRepositoryConfig


def _make_repo(max_rows: int = 200) -> DataRepository:
    config = DataRepositoryConfig(data_dir=Path("data"), max_rows=max_rows)
    return DataRepository.load_from_csvs(config)


def test_paysim_loads_with_required_columns():
    repo = _make_repo(max_rows=100)
    df = repo.load_paysim()

    assert not df.empty, "PaySim dataset should load some rows"
    required = {
        "transaction_id",
        "account_id",
        "beneficiary_id",
        "amount",
        "transaction_type",
        "timestamp",
        "is_fraud",
        "dataset",
    }
    assert required.issubset(df.columns)


def test_ieee_loads_with_required_columns():
    repo = _make_repo(max_rows=100)
    df = repo.load_ieee_transactions()

    assert not df.empty, "IEEE transactions dataset should load some rows"
    required = {
        "transaction_id",
        "account_id",
        "beneficiary_id",
        "amount",
        "transaction_type",
        "timestamp",
        "is_fraud",
        "dataset",
    }
    assert required.issubset(df.columns)


def test_transaction_lookup_by_id():
    repo = _make_repo(max_rows=200)
    df = repo.load_paysim()
    assert not df.empty

    tx_id = df.iloc[0]["transaction_id"]
    record = repo.get_transaction(tx_id)

    assert isinstance(record, EnrichedTransactionRecord)
    assert record.transaction_id == tx_id


def test_account_history_lookup_returns_consistent_account():
    repo = _make_repo(max_rows=200)
    df = repo.load_paysim()
    assert not df.empty

    account_id = df.iloc[0]["account_id"]
    history = repo.get_account_history(account_id, limit=50)

    assert history, "Expected non-empty account history"
    assert all(r.account_id == account_id for r in history)


def test_alert_queue_returns_records_with_labels():
    repo = _make_repo(max_rows=500)
    alerts = repo.get_alert_queue(limit=25)

    assert alerts, "Expected non-empty alert queue"
    assert len(alerts) <= 25
    first = alerts[0]
    assert isinstance(first, EnrichedTransactionRecord)
    label = repo.get_label(first.transaction_id)
    assert label in (0, 1)

