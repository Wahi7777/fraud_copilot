from pathlib import Path

import pytest

from services import DataRepository, DataRepositoryConfig
from services.graph_builder import GraphBuilder


def _make_repo(max_rows: int = 500) -> DataRepository:
    config = DataRepositoryConfig(data_dir=Path("data"), max_rows=max_rows)
    return DataRepository.load_from_csvs(config)


def test_graph_builds_with_nodes_and_edges():
    repo = _make_repo(max_rows=200)
    builder = GraphBuilder(repo)
    G = builder.build_graph()

    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0


def test_connected_accounts_lookup_includes_direct_beneficiary():
    repo = _make_repo(max_rows=200)
    paysim_df = repo.load_paysim()
    assert not paysim_df.empty

    src = paysim_df.iloc[0]["account_id"]
    dst = paysim_df.iloc[0]["beneficiary_id"]

    builder = GraphBuilder(repo)
    connected = builder.get_connected_accounts(src, depth=1)

    assert dst in connected


def test_has_path_to_flagged_node_when_fraud_transactions_exist():
    repo = _make_repo(max_rows=500)
    paysim_df = repo.load_paysim()
    assert not paysim_df.empty

    fraud_rows = paysim_df[paysim_df["is_fraud"] == 1]
    if fraud_rows.empty:
        pytest.skip("No fraud-labelled PaySim rows in sampled subset")

    row = fraud_rows.iloc[0]
    src_account = row["account_id"]

    builder = GraphBuilder(repo)

    assert builder.has_path_to_flagged_node(src_account) is True


def test_circular_pattern_detection_handles_no_match_safely():
    repo = _make_repo(max_rows=100)
    builder = GraphBuilder(repo)

    # For an obviously non-existent account, this should simply return False.
    assert builder.detect_simple_circular_pattern("NON_EXISTENT_ACCOUNT") is False

