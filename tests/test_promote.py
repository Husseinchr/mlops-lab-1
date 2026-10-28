from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.food11.promote import promote, promotion_decision


def test_gate_boundary():
    accepted, margin = promotion_decision(0.9, 0.8, 1000)
    assert accepted
    assert margin == pytest.approx(0.02529822128)
    assert not promotion_decision(0.81, 0.8, 1000)[0]
    assert promotion_decision(0.8 + margin, 0.8, 1000)[0]


def test_invalid_gate_inputs():
    with pytest.raises(ValueError):
        promotion_decision(float("nan"), 0.8, 1000)
    with pytest.raises(ValueError):
        promotion_decision(0.9, 0.8, 0)


@pytest.mark.parametrize("accuracy,alias", [(0.9, "champion"), (0.81, "challenger")])
def test_registry_gate_changes_only_the_expected_alias(accuracy, alias):
    client = Mock()
    candidate = SimpleNamespace(version="2", run_id="candidate")
    champion = SimpleNamespace(version="1", run_id="champion")
    client.get_model_version.return_value = candidate
    client.get_model_version_by_alias.side_effect = lambda name, requested: (
        champion if requested == "champion" else candidate
    )
    params = {"test_size": "1000", "dataset": "mini", "dataset_sha256": "same-data"}
    client.get_run.side_effect = lambda run_id: SimpleNamespace(
        data=SimpleNamespace(
            params=params,
            metrics={"test_accuracy": accuracy if run_id == "candidate" else 0.8},
        )
    )
    result = promote(client, 2)
    assert result["gate.decision"] == alias
    client.set_registered_model_alias.assert_called_once_with("food11", alias, "2")
    assert client.set_model_version_tag.call_count == 6
    if alias == "champion":
        client.delete_registered_model_alias.assert_called_once_with(
            "food11", "challenger"
        )
    else:
        client.delete_registered_model_alias.assert_not_called()
