from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from models import EvidenceItem, FraudSignalCode


class EvidenceRegistry:
    """
    In-memory registry for investigation evidence.

    The registry:
    - Accepts `EvidenceItem` instances from agents
    - Deduplicates entries based on (source_agent, signal_code, summary)
    - Exposes query helpers for downstream consumption
    """

    def __init__(self) -> None:
        self._items: list[EvidenceItem] = []
        self._index: set[tuple[str, FraudSignalCode, str]] = set()

    # ------------------------------------------------------------------ #
    # Mutation API
    # ------------------------------------------------------------------ #
    def append(self, item: EvidenceItem, *, deduplicate: bool = True) -> None:
        """
        Append a single evidence item to the registry.

        When `deduplicate` is True (the default), the registry will avoid
        inserting another item with the same (source_agent, signal_code,
        summary) triple.
        """
        key = (item.source_agent, item.signal_code, item.summary)
        if deduplicate and key in self._index:
            return

        self._items.append(item)
        self._index.add(key)

    def extend(self, items: Iterable[EvidenceItem], *, deduplicate: bool = True) -> None:
        """
        Append multiple evidence items.
        """
        for item in items:
            self.append(item, deduplicate=deduplicate)

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #
    def list_all(self) -> List[EvidenceItem]:
        return list(self._items)

    def list_by_agent(self, source_agent: str) -> List[EvidenceItem]:
        return [e for e in self._items if e.source_agent == source_agent]

    def list_by_signal(self, signal_code: FraudSignalCode) -> List[EvidenceItem]:
        return [e for e in self._items if e.signal_code == signal_code]

    def grouped_by_signal(self) -> dict[FraudSignalCode, list[EvidenceItem]]:
        """
        Convenience helper used by scoring to see which items contributed.
        """
        groups: dict[FraudSignalCode, list[EvidenceItem]] = defaultdict(list)
        for e in self._items:
            groups[e.signal_code].append(e)
        return groups

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self._items)

