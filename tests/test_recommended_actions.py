from app import _recommended_actions_for_recommendation


def test_recommended_actions_clear():
    actions = _recommended_actions_for_recommendation("Clear")
    assert actions == [
        "Allow the transaction",
        "Record investigation outcome",
        "Close the case as false positive",
    ]


def test_recommended_actions_escalate():
    actions = _recommended_actions_for_recommendation("Escalate")
    assert actions == [
        "Perform enhanced customer verification",
        "Review recent transaction history",
        "Investigate beneficiary account activity",
        "Escalate to senior fraud analyst if risk persists",
    ]


def test_recommended_actions_decline():
    actions = _recommended_actions_for_recommendation("Decline")
    assert actions == [
        "Block the transaction",
        "Place a temporary hold on the sender account",
        "Investigate linked beneficiary accounts",
        "Escalate to financial crime investigation if suspicious activity continues",
    ]
