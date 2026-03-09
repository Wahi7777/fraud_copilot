from pathlib import Path

from models import EnrichedTransactionRecord
from services import DataRepository, DataRepositoryConfig
from services.investigation_feature_builder import (
    InvestigationContext,
    InvestigationFeatureBuilder,
)


def _make_repo(max_rows: int = 200) -> DataRepository:
    config = DataRepositoryConfig(data_dir=Path("data"), max_rows=max_rows)
    return DataRepository.load_from_csvs(config)


def test_feature_builder_produces_enriched_record_and_is_deterministic():
    repo = _make_repo(max_rows=150)
    paysim_df = repo.load_paysim()
    assert not paysim_df.empty

    tx_id = paysim_df.iloc[0]["transaction_id"]
    builder = InvestigationFeatureBuilder(repo)
    ctx = InvestigationContext(transaction_id=tx_id)

    record_1 = builder.build_for_transaction(ctx)
    record_2 = builder.build_for_transaction(ctx)

    assert isinstance(record_1, EnrichedTransactionRecord)
    # Deterministic output for same input.
    assert record_1.model_dump() == record_2.model_dump()

    # Important enrichment fields should be populated.
    assert record_1.device_id is not None
    assert record_1.device_type is not None
    assert record_1.ip_address is not None
    assert record_1.account_age_days is not None
    assert record_1.beneficiary_age_days is not None
    assert record_1.transaction_velocity_10min is not None
    assert record_1.transaction_velocity_1hr is not None
    assert record_1.device_risk_score is not None
    assert record_1.email_domain_risk is not None
    assert record_1.geo_distance_jump_km is not None
    assert record_1.impossible_travel_flag is not None


def test_feature_builder_handles_missing_history_gracefully():
    repo = _make_repo(max_rows=5)

    # Create a synthetic, isolated transaction with no prior history.
    base_ts = repo.load_paysim().iloc[0]["timestamp"]

    tx = EnrichedTransactionRecord(
        transaction_id="PAYSIM-SYNTH-0",
        account_id="ACC-SYNTH",
        beneficiary_id="ACC-SYNTH-B",
        amount=100.0,
        transaction_type="TRANSFER",
        timestamp=base_ts,
    )
    repo._records[tx.transaction_id] = tx  # type: ignore[attr-defined]

    builder = InvestigationFeatureBuilder(repo)
    ctx = InvestigationContext(transaction_id=tx.transaction_id)

    enriched = builder.build_for_transaction(ctx)

    assert enriched.transaction_id == tx.transaction_id
    # With no previous history and no country information, geo signals should be sane.
    assert enriched.geo_distance_jump_km == 0.0
    assert enriched.impossible_travel_flag is False
    # Age and velocity fields should still be set deterministically.
    assert enriched.account_age_days is not None
    assert enriched.beneficiary_age_days is not None
    assert enriched.transaction_velocity_10min is not None
    assert enriched.transaction_velocity_1hr is not None

