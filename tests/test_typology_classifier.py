from services.typology_classifier import TYPOLOGY_DEFINITIONS, classify_typology_from_signals


def test_mule_pattern_maps_to_potential_mule_transfer():
    result = classify_typology_from_signals(
        triggered_signals=["risky_beneficiary_link", "beneficiary_incoming_volume", "fanout_pattern"],
        signal_breakdown={
            "risky_beneficiary_link": 0.10,
            "beneficiary_incoming_volume": 0.06,
            "fanout_pattern": 0.05,
        },
    )
    assert result.fraud_typology == "Potential Mule Transfer"
    assert result.typology_definition == TYPOLOGY_DEFINITIONS["Potential Mule Transfer"]
    assert result.fraud_typology not in {"Generic Fraud", "Mule Network"}


def test_velocity_pattern_maps_to_velocity_fraud():
    result = classify_typology_from_signals(
        triggered_signals=["tx_burst_1h", "unique_beneficiaries_1h"],
        signal_breakdown={"tx_burst_1h": 0.08, "unique_beneficiaries_1h": 0.05},
    )
    assert result.fraud_typology == "Velocity Fraud"
    assert "velocity" in result.typology_definition.lower()


def test_account_takeover_pattern_maps_correctly():
    result = classify_typology_from_signals(
        triggered_signals=["new_device_indicator", "device_reuse_risk", "geo_anomaly", "ip_risk"],
        signal_breakdown={
            "new_device_indicator": 0.04,
            "device_reuse_risk": 0.05,
            "geo_anomaly": 0.05,
            "ip_risk": 0.04,
        },
    )
    assert result.fraud_typology == "Account Takeover"
    assert result.typology_definition == TYPOLOGY_DEFINITIONS["Account Takeover"]


def test_no_specific_pattern_uses_non_generic_anomaly_or_unknown():
    result = classify_typology_from_signals(
        triggered_signals=["time_of_day_deviation"],
        signal_breakdown={"time_of_day_deviation": 0.01},
    )
    assert result.fraud_typology in {"Transaction Anomaly", "Unknown / Mixed Pattern"}
    assert result.typology_definition == TYPOLOGY_DEFINITIONS[result.fraud_typology]
