"""Compute projections, Foundry rate units and concise terminal output."""

import json
import shlex
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.catalog_availability import CatalogAvailability
from hf_cloud.core.estimation import compute_cost
from hf_cloud.providers.foundry.profiles import load_profile, profile_catalog

runner = CliRunner()


def test_replica_costs_include_all_gpus_once():
    row = {"numGpu": 8, "pricePerHourUsd": 10}
    assert compute_cost(row, 24, 2) == {"deploymentPricePerHourUsd": 20, "estimatedCostUsd": 480}
    assert compute_cost(row, 0.5, 3)["estimatedCostUsd"] == 15
    assert compute_cost({"pricePerHourUsd": None}, 24, 2)["estimatedCostUsd"] is None


@pytest.mark.parametrize("hours", ["0", "-1", "nan", "inf"])
def test_invalid_duration_does_not_query_cloud_or_estimator(comparison, estimate_catalog_checks, hours):
    run, _ = comparison
    result = runner.invoke(app, ["estimate", "org/model", "--hours", hours, "--json"])
    assert result.exit_code == 1
    assert "--hours" in json.loads(result.stdout)["error"]
    run.assert_not_called()
    estimate_catalog_checks.assert_not_called()


def test_unknown_prices_do_not_gain_a_projected_cost(comparison):
    result = runner.invoke(app, ["estimate", "org/model", "--hours", "24", "--instance-count", "2", "--json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["costEstimate"] == {
        "hours": 24,
        "instanceCount": 2,
        "basis": "compute-only",
        "replicas": "constant",
    }
    assert data["cheapest"]["pricePerHourUsd"] == 1
    assert data["cheapest"]["deploymentPricePerHourUsd"] == 2
    assert data["cheapest"]["estimatedCostUsd"] == 48
    assert data["instances"][-1]["estimatedCostUsd"] is None


def test_cost_options_are_shared_by_provider_aliases(comparison):
    flags = ["org/model", "--hours", "24", "--instance-count", "2", "--json"]
    main = runner.invoke(app, ["estimate", *flags, "--provider", "sagemaker"])
    shortcut = runner.invoke(app, ["sagemaker", "estimate", *flags])
    assert main.exit_code == shortcut.exit_code == 0
    assert json.loads(main.stdout) == json.loads(shortcut.stdout)


def test_deploy_hint_matches_projected_replica_count(comparison):
    result = runner.invoke(app, ["estimate", "org/model", "--hours", "24", "--instance-count", "2"])
    assert result.exit_code == 0, result.stdout
    hint = result.stdout.split("Deploy cheapest instance\n", 1)[1].split("Match the deployment engine", 1)[0]
    tokens = shlex.split(hint.replace("\\\n", ""))
    assert tokens[tokens.index("--min-replicas") + 1] == "2"
    assert tokens[tokens.index("--max-replicas") + 1] == "2"
    assert "$48.00 for 24 hours" in result.stdout


def test_catalog_table_is_after_prices_and_full_reasons_require_verbose(comparison, monkeypatch):
    marker = "INTERNAL_REGISTRY_ERROR_WITH_LONG_DIAGNOSTICS"
    monkeypatch.setattr(
        "hf_cloud.cli.estimate.check_catalog_availability",
        Mock(
            return_value={
                "foundry": CatalogAvailability("foundry", "unknown", marker),
            }
        ),
    )
    basic = runner.invoke(app, ["estimate", "org/model"])
    assert basic.exit_code == 0, basic.stdout
    assert basic.stdout.index("vertex-fit") < basic.stdout.index("Microsoft catalogs")
    assert marker not in basic.stdout
    assert "Could not verify" in basic.stdout
    detail = runner.invoke(app, ["estimate", "org/model", "--verbose"])
    assert detail.exit_code == 0 and marker in detail.stdout


def test_catalog_command_has_compact_rows_and_opt_in_explanations(monkeypatch):
    reason = "A very long missing-asset explanation with implementation details"
    monkeypatch.setattr(
        "hf_cloud.cli.catalog.check_catalog_availability",
        lambda *_: {
            "azureml": CatalogAvailability("azureml", "unavailable", reason),
        },
    )
    basic = runner.invoke(app, ["catalog", "org/model"])
    assert basic.exit_code == 0 and reason not in basic.stdout
    detail = runner.invoke(app, ["catalog", "org/model", "--verbose"])
    assert detail.exit_code == 0 and reason in detail.stdout


def profile_data():
    return {
        "modelId": "org/model",
        "modelAssetId": "azureml://registries/azure-huggingface/models/org--model/versions/3",
        "deploymentTemplateId": "azureml://registries/azure-huggingface/deploymenttemplates/test/versions/1",
        "acceleratorType": "H100_80GB",
        "acceleratorsPerInstance": 2,
        "maxModelLen": 4096,
        "pricePerAcceleratorHourUsd": 5,
        "pricingSource": "https://example.com/verified-rate",
    }


def test_foundry_accelerator_rate_is_converted_before_replica_projection(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile_data()))
    loaded = load_profile(path, "org/model")
    assert loaded["pricePerInstanceHourUsd"] == 10
    instance = profile_catalog(path, "org/model").instances[0]
    assert compute_cost(instance, 24, 2)["estimatedCostUsd"] == 480
    assert "pricePerInstanceHourUsd" not in json.loads(path.read_text())


@pytest.mark.parametrize(
    "change",
    [
        {"pricePerInstanceHourUsd": 8},
        {"pricePerAcceleratorHourUsd": -1},
        {"pricePerAcceleratorHourUsd": float("nan")},
        {"pricePerAcceleratorHourUsd": True},
        {"pricingSource": ""},
    ],
)
def test_foundry_rate_units_are_unambiguous_and_sourced(tmp_path, change):
    from hf_cloud.core.exceptions import ConfigurationError

    path = tmp_path / "profile.json"
    path.write_text(json.dumps({**profile_data(), **change}))
    with pytest.raises(ConfigurationError):
        load_profile(path, "org/model")


def test_azureml_cost_projection_uses_verified_vm_rate(comparison):
    from hf_cloud.utils.hf_mem import load_instance_data

    _, catalogs = comparison
    catalogs["azureml"] = load_instance_data("azureml")
    result = runner.invoke(
        app,
        ["estimate", "org/model", "--provider", "azureml", "--hours", "8", "--instance-count", "2", "--json"],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["cheapest"]["instanceType"] == "Standard_NC24ads_A100_v4"
    assert data["cheapest"]["deploymentPricePerHourUsd"] == 7.346
    assert data["cheapest"]["estimatedCostUsd"] == 58.768


def test_foundry_cli_projection_uses_complete_template_instance_rate(comparison, tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile_data()))
    result = runner.invoke(
        app,
        [
            "foundry",
            "estimate",
            "org/model",
            "--profile",
            str(path),
            "--hours",
            "24",
            "--instance-count",
            "2",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["cheapest"]["pricePerHourUsd"] == 10
    assert data["cheapest"]["estimatedCostUsd"] == 480
