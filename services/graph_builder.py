from __future__ import annotations

from pathlib import Path
from typing import List

import networkx as nx

from services.data_repository import DataRepository, DataRepositoryConfig


class GraphBuilder:
    """
    Builds and queries a transaction graph for network-based fraud analysis.

    Nodes represent accounts; directed edges represent money flowing from
    sender (origin account) to beneficiary account.
    """

    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository
        self._graph: nx.DiGraph | None = None

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def build_graph(self) -> nx.DiGraph:
        """
        Build (or return cached) NetworkX DiGraph from PaySim transactions.

        - Nodes: account ids
        - Edges: sender -> beneficiary, with `amount` and `is_fraud` attributes
        - Fraud-flagged nodes: beneficiaries that receive at least one
          transaction labelled as fraud (is_fraud == 1)
        """
        if self._graph is not None:
            return self._graph

        df = self.repository.load_paysim()
        G = nx.DiGraph()

        for _, row in df.iterrows():
            src = row["account_id"]
            dst = row["beneficiary_id"]
            amount = float(row["amount"])
            is_fraud = int(row["is_fraud"])

            G.add_node(src)
            G.add_node(dst)
            G.add_edge(src, dst, amount=amount, is_fraud=is_fraud)

            if is_fraud == 1:
                # Beneficiary accounts that receive fraudulent funds are
                # flagged so downstream analysis can check reachability.
                G.nodes[dst]["fraud_flagged"] = True

        self._graph = G
        return G

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #
    def get_connected_accounts(self, account_id: str, depth: int = 2) -> List[str]:
        """
        Return accounts connected to the given account within a given depth.

        Connectivity is evaluated on an undirected view of the transaction
        graph (i.e. either inbound or outbound edges contribute).
        """
        G = self.build_graph()
        if account_id not in G:
            return []

        undirected = G.to_undirected()
        lengths = nx.single_source_shortest_path_length(undirected, account_id, cutoff=depth)
        return sorted(n for n in lengths.keys() if n != account_id)

    def get_beneficiary_degree(self, beneficiary_id: str) -> int:
        """
        Return the in-degree (number of distinct senders) for a beneficiary.
        """
        G = self.build_graph()
        if beneficiary_id not in G:
            return 0
        return int(G.in_degree(beneficiary_id))

    def has_path_to_flagged_node(self, account_id: str) -> bool:
        """
        Return True if there exists a directed path from `account_id` to any
        fraud-flagged node in the graph.
        """
        G = self.build_graph()
        if account_id not in G:
            return False

        flagged = {n for n, data in G.nodes(data=True) if data.get("fraud_flagged")}
        if not flagged:
            return False

        descendants = nx.descendants(G, account_id)
        return bool(flagged & descendants)

    def detect_simple_circular_pattern(self, account_id: str, max_cycle_length: int = 4) -> bool:
        """
        Detect whether the account participates in a small directed cycle.

        This is a simple proxy for circular money movement patterns.
        Only cycles up to `max_cycle_length` are considered to keep the
        interpretation simple and computations bounded.
        """
        G = self.build_graph()
        if account_id not in G:
            return False

        # Iterate over simple cycles and short-circuit once we see a
        # relevant one that includes this account.
        for cycle in nx.simple_cycles(G):
            if account_id in cycle and 2 <= len(cycle) <= max_cycle_length:
                return True

        return False


def build_default_graph(max_rows: int | None = 10_000) -> nx.DiGraph:
    """
    Convenience helper for tests and exploratory usage.
    """
    repo = DataRepository.load_from_csvs(DataRepositoryConfig(data_dir=Path("data"), max_rows=max_rows))
    builder = GraphBuilder(repo)
    return builder.build_graph()

