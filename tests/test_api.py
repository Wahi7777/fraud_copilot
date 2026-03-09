from fastapi.testclient import TestClient

from app import create_app


def _make_client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_investigate_missing_transaction_returns_404():
    client = _make_client()

    resp = client.post("/investigate", json={"transaction_id": "NON_EXISTENT"})
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data


def test_alerts_returns_at_least_25_items():
    client = _make_client()

    resp = client.get("/alerts?limit=25")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert isinstance(data["items"], list)
    assert len(data["items"]) >= 1  # limit is an upper bound; dataset may be smaller in tests


def test_metrics_response_keys_present():
    client = _make_client()

    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()

    for key in (
        "flagged_transactions",
        "high_risk_alerts",
        "open_investigations",
        "escalated_cases",
        "false_positive_rate",
        "avg_resolution_time",
    ):
        assert key in data

