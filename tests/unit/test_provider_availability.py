"""Deployment evidence is independent of catalog memory fits and prices."""

import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.catalog_availability import CatalogAvailability
from hf_cloud.core.provider_availability import provider_availability
from hf_cloud.utils.hf_mem import EstimateOptions, InstanceData

runner = CliRunner()


def test_all_selected_providers_show_evidence_even_when_table_rows_are_limited(comparison, monkeypatch):
    worker, catalogs = comparison
    catalogs["azureml"] = InstanceData("azureml", [], 0.1)
    checks = Mock(
        return_value={
            "azureml": CatalogAvailability("azureml", "unavailable", "Not in this registry"),
            "foundry": CatalogAvailability("foundry", "unknown", "Catalog request failed"),
        }
    )
    monkeypatch.setattr("hf_cloud.cli.estimate.check_catalog_availability", checks)
    result = runner.invoke(
        app,
        [
            "estimate",
            "org/model",
            "--provider",
            "sagemaker",
            "--provider",
            "vertex",
            "--provider",
            "azureml",
            "--provider",
            "foundry",
            "--limit",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    availability = data["providerAvailability"]
    assert list(availability) == ["sagemaker", "vertex", "azureml", "foundry"]
    assert len(data["instances"]) == 1 and data["instances"][0]["provider"] == "vertex"
    for provider in ("sagemaker", "vertex"):
        assert availability[provider]["deploymentStatus"] == "runtime-unverified"
        assert availability[provider]["estimateStatus"] == "included"
        assert availability[provider]["deploymentPath"] == "hub"
        assert not availability[provider]["runtimeVerified"]
    assert availability["azureml"]["deploymentStatus"] == "unavailable"
    assert availability["foundry"]["deploymentStatus"] == "unknown"
    assert availability["azureml"]["estimateStatus"] == "excluded"
    assert data["catalogAvailability"]["foundry"]["status"] == "unknown"
    worker.assert_called_once()
    checks.assert_called_once()


def test_provider_filtered_summary_does_not_request_other_clouds(comparison, estimate_catalog_checks):
    basic = runner.invoke(app, ["sagemaker", "estimate", "org/model"])
    assert basic.exit_code == 0, basic.stdout
    summary = basic.stdout.split("Provider availability", 1)[1].split("Pricing:", 1)[0]
    assert "sagemaker" in summary and "Runtime unverified" in summary
    assert "vertex" not in summary and "azureml" not in summary and "foundry" not in summary
    assert "ModelBuilder" not in summary
    detail = runner.invoke(app, ["sagemaker", "estimate", "org/model", "--verbose"])
    assert detail.exit_code == 0 and "ModelBuilder" in detail.stdout
    estimate_catalog_checks.assert_not_called()


@pytest.mark.parametrize(
    "flags",
    [
        ["--revision", "dev"],
        ["--file", "model.safetensors"],
        ["--format", "gguf"],
        ["--format", "onnx"],
    ],
)
def test_cost_estimates_do_not_claim_selected_weights_can_be_deployed(comparison, flags):
    result = runner.invoke(app, ["vertex", "estimate", "org/model", "--json", *flags])
    assert result.exit_code == 0, result.stdout
    availability = json.loads(result.stdout)["providerAvailability"]["vertex"]
    assert availability["deploymentStatus"] == "unsupported-selection"
    assert availability["estimateStatus"] == "included"


@pytest.mark.parametrize("model", ["https://huggingface.co/org/model", "./local-model"])
def test_non_hub_inputs_are_not_marked_deployable(model):
    result = provider_availability(model, ["sagemaker"], EstimateOptions(), {}, [], {})
    assert result["sagemaker"]["deploymentStatus"] == "unsupported-selection"


def test_inferred_gguf_format_and_existing_local_directory_are_not_hub_deployments(tmp_path):
    for model, weights_format in (("org/model", "gguf"), (str(tmp_path), "safetensors")):
        result = provider_availability(model, ["vertex"], EstimateOptions(), {}, [], {}, weights_format)
        assert result["vertex"]["deploymentStatus"] == "unsupported-selection"
    # Legacy single-component Hub IDs remain valid for these two deploy commands.
    result = provider_availability("gpt2", ["vertex"], EstimateOptions(), {}, [], {}, "safetensors")
    assert result["vertex"]["deploymentStatus"] == "runtime-unverified"


def test_publication_and_runtime_support_are_independent_of_memory_fit():
    result = provider_availability(
        "org/model",
        ["azureml", "sagemaker"],
        EstimateOptions(),
        {"azureml": CatalogAvailability("azureml", "available")},
        [],
        {},
    )
    assert result["azureml"]["deploymentStatus"] == "catalog-available"
    assert result["sagemaker"]["deploymentStatus"] == "runtime-unverified"
    assert all(row["estimateStatus"] == "no-memory-fit" for row in result.values())
    assert all(not row["runtimeVerified"] for row in result.values())


def test_unknown_price_does_not_hide_existing_memory_fit(comparison):
    _, catalogs = comparison
    for row in catalogs["sagemaker"].instances:
        row["pricePerHourUsd"] = None
    result = runner.invoke(app, ["sagemaker", "estimate", "org/model", "--json"])
    data = json.loads(result.stdout)
    assert result.exit_code == 1 and data["cheapest"] is None
    assert data["providerAvailability"]["sagemaker"]["estimateStatus"] == "unpriced"
    assert data["providerAvailability"]["sagemaker"]["deploymentStatus"] == "runtime-unverified"
