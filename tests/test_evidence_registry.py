from models import EvidenceItem, EvidenceSeverity, FraudSignalCode
from services.evidence_registry import EvidenceRegistry


def _make_sample_item(summary: str = "velocity anomaly") -> EvidenceItem:
    return EvidenceItem(
        source_agent="PatternAgent",
        signal_code=FraudSignalCode.HIGH_VELOCITY,
        severity=EvidenceSeverity.HIGH,
        summary=summary,
        details={"count": 5},
    )


def test_append_and_retrieve():
    registry = EvidenceRegistry()
    item = _make_sample_item()

    registry.append(item)

    all_items = registry.list_all()
    assert len(all_items) == 1
    assert all_items[0].summary == "velocity anomaly"

    by_agent = registry.list_by_agent("PatternAgent")
    assert len(by_agent) == 1

    by_signal = registry.list_by_signal(FraudSignalCode.HIGH_VELOCITY)
    assert len(by_signal) == 1


def test_duplicate_handling_with_dedup_enabled():
    registry = EvidenceRegistry()
    item1 = _make_sample_item()
    item2 = _make_sample_item()  # identical key

    registry.append(item1)
    registry.append(item2)  # should be ignored due to deduplication

    assert len(registry.list_all()) == 1


def test_duplicates_can_be_forced_when_needed():
    registry = EvidenceRegistry()
    item1 = _make_sample_item()
    item2 = _make_sample_item()

    registry.append(item1)
    registry.append(item2, deduplicate=False)

    assert len(registry.list_all()) == 2

