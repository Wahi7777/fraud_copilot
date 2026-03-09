from services.decision_policy import DecisionPolicyEngine


def _signals(*keys: str):
    payload = {}
    for k in keys:
        payload[k] = {"severity": "high", "value": "x", "detail": "x"}
    return payload


def test_high_risk_can_escalate_when_confirmation_is_weak():
    engine = DecisionPolicyEngine()
    out = engine.evaluate(
        risk_score=0.82,
        signals=_signals("transfer_amount"),
        network={"fraud_link": False, "suspected_network_risk": False},
        evidence=[],
        agent_trace=[],
    )
    assert out.recommendation == "Escalate"


def test_medium_risk_escalates():
    engine = DecisionPolicyEngine()
    out = engine.evaluate(
        risk_score=0.55,
        signals=_signals("device_reuse"),
        network={"fraud_link": False, "suspected_network_risk": True},
        evidence=[],
        agent_trace=[],
    )
    assert out.recommendation == "Escalate"


def test_low_risk_can_clear():
    engine = DecisionPolicyEngine()
    out = engine.evaluate(
        risk_score=0.18,
        signals={},
        network={"fraud_link": False, "suspected_network_risk": False},
        evidence=[{"summary": "known device and known beneficiary", "details": "", "severity": "low"}],
        agent_trace=[],
    )
    assert out.recommendation == "Clear"
    assert out.workflow_status == "Closed"


def test_low_risk_can_escalate_if_suspicious_combination():
    engine = DecisionPolicyEngine()
    out = engine.evaluate(
        risk_score=0.28,
        signals=_signals("geo_anomaly", "device_reuse"),
        network={"fraud_link": False, "suspected_network_risk": True},
        evidence=[{"summary": "proxy vpn usage observed", "details": "", "severity": "high"}],
        agent_trace=[],
    )
    assert out.recommendation == "Escalate"


def test_agent_votes_and_override_flags_present():
    engine = DecisionPolicyEngine()
    out = engine.evaluate(
        risk_score=0.74,
        signals=_signals("velocity", "device_reuse", "geo_anomaly"),
        network={"fraud_link": True, "suspected_network_risk": True},
        evidence=[{"summary": "account takeover indicators", "details": "", "severity": "high"}],
        agent_trace=[],
    )
    assert out.agent_votes
    assert set(out.agent_votes.keys()) == {
        "transaction_agent",
        "behavior_agent",
        "network_agent",
        "policy_agent",
    }
    assert isinstance(out.override_applied, bool)


def test_override_logic_exercised_upward_and_downward():
    engine = DecisionPolicyEngine()
    upward = engine.evaluate(
        risk_score=0.52,
        signals=_signals("velocity", "device_reuse", "geo_anomaly"),
        network={"fraud_link": True, "suspected_network_risk": True},
        evidence=[{"summary": "account takeover with proxy usage", "details": "", "severity": "high"}],
        agent_trace=[],
    )
    assert upward.recommendation == "Decline"
    assert upward.override_applied is True

    downward = engine.evaluate(
        risk_score=0.78,
        signals=_signals("transfer_amount"),
        network={"fraud_link": False, "suspected_network_risk": False},
        evidence=[{"summary": "known device and known beneficiary", "details": "", "severity": "low"}],
        agent_trace=[],
    )
    assert downward.recommendation in {"Escalate", "Clear"}
    assert downward.override_applied is True


def test_recommendation_and_workflow_are_separate_fields():
    engine = DecisionPolicyEngine()
    out = engine.evaluate(
        risk_score=0.62,
        signals=_signals("device_reuse"),
        network={"fraud_link": False, "suspected_network_risk": False},
        evidence=[],
        agent_trace=[],
    )
    assert out.recommendation in {"Clear", "Escalate", "Decline"}
    assert out.workflow_status in {"New", "In Review", "Closed"}
    assert out.workflow_status != "Escalate"


def test_recommendation_distribution_non_degenerate_batch():
    engine = DecisionPolicyEngine()
    counts = {"Clear": 0, "Escalate": 0, "Decline": 0}
    for i in range(80):
        if i < 20:
            score = 0.18 + (i % 5) * 0.02
            signals = {}
            evidence = [{"summary": "known device known beneficiary", "details": "", "severity": "low"}]
            network = {"fraud_link": False, "suspected_network_risk": False}
        elif i < 55:
            score = 0.38 + (i % 8) * 0.03
            signals = _signals("transfer_amount") if i % 2 else _signals("device_reuse")
            evidence = []
            network = {"fraud_link": False, "suspected_network_risk": i % 3 == 0}
        else:
            score = 0.74 + (i % 6) * 0.03
            signals = _signals("velocity", "device_reuse", "geo_anomaly")
            evidence = [{"summary": "proxy usage", "details": "", "severity": "high"}]
            network = {"fraud_link": i % 4 == 0, "suspected_network_risk": True}

        out = engine.evaluate(
            risk_score=score,
            signals=signals,
            network=network,
            evidence=evidence,
            agent_trace=[],
        )
        assert out.recommendation != "Pending"
        counts[out.recommendation] += 1

    assert counts["Clear"] > 0
    assert counts["Escalate"] > 0
    assert counts["Decline"] > 0
