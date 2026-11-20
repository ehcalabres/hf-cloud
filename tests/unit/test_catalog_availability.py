"""Catalog absence, failed verification, profile validation and estimate exclusions."""

import json
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.catalog_availability import CatalogAvailability, check_catalog_availability
from hf_cloud.core.exceptions import ConfigurationError, ProviderError
from hf_cloud.providers.azure_catalog import CatalogModelUnavailable, PublishedModel
from hf_cloud.utils.hf_mem import load_instance_data

MODEL = PublishedModel("org/model", "HuggingFace", "org-model", "3")
TEMPLATE = "azureml://registries/azure-huggingface/deploymenttemplates/org-model-h100/versions/5"
FOUNDRY = PublishedModel(
    "org/model", "azure-huggingface", "org--model", "3", default_template=TEMPLATE, templates=(TEMPLATE,)
)
PROFILE = {
    "modelId": "org/model",
    "modelAssetId": FOUNDRY.asset_id,
    "deploymentTemplateId": TEMPLATE,
    "acceleratorType": "H100_80GB",
    "acceleratorsPerInstance": 1,
    "maxModelLen": 2048,
    "pricePerInstanceHourUsd": 0.5,
    "pricingSource": "https://example.com/pricing",
}


def test_discovery_distinguishes_absence_from_network_or_auth_errors(monkeypatch):
    catalog = Mock()

    def resolve(model, provider, version):
        if provider == "azureml":
            raise CatalogModelUnavailable("Not published in Azure ML")
        raise ProviderError("foundry", "Please run az login")

    catalog.resolve.side_effect = resolve
    monkeypatch.setattr("hf_cloud.core.catalog_availability.AzureCatalog", lambda: catalog)
    result = check_catalog_availability("org/model")
    assert result["azureml"].status == "unavailable"
    assert result["foundry"].status == "unknown"
    assert "az login" in result["foundry"].reason
    assert catalog.resolve.call_count == 2


def test_both_catalogs_are_checked_independently(monkeypatch):
    catalog = Mock()
    catalog.resolve.side_effect = lambda _, provider, version: MODEL if provider == "azureml" else FOUNDRY
    monkeypatch.setattr("hf_cloud.core.catalog_availability.AzureCatalog", lambda: catalog)
    results = check_catalog_availability("org/model")
    assert all(result.status == "available" for result in results.values())
    assert results["azureml"].to_dict()["model"]["asset_id"] == MODEL.asset_id
    assert "registry/azure-huggingface" in results["foundry"].to_dict()["catalogUrl"]


def test_timeout_is_bounded_and_does_not_block_other_provider(monkeypatch):
    release = threading.Event()

    def resolve(_, provider, version):
        if provider == "azureml":
            release.wait(2)
            return MODEL
        return FOUNDRY

    monkeypatch.setattr(
        "hf_cloud.core.catalog_availability.AzureCatalog", lambda: Mock(resolve=Mock(side_effect=resolve))
    )
    start = time.monotonic()
    try:
        results = check_catalog_availability("org/model", timeout_s=0.05)
        assert time.monotonic() - start < 0.5
        assert results["azureml"].status == "unknown"
        assert results["foundry"].status == "available"
    finally:
        release.set()


def test_foundry_profile_requires_published_model_template_and_topology(monkeypatch):
    catalog = Mock()
    catalog.resolve.return_value = FOUNDRY
    catalog.foundry_template.return_value = {"accelerators_per_instance": 1}
    monkeypatch.setattr("hf_cloud.core.catalog_availability.AzureCatalog", lambda: catalog)
    result = check_catalog_availability("org/model", ["foundry"], foundry_profile=PROFILE)
    assert result["foundry"].status == "available"
    catalog.resolve.assert_called_once_with("org/model", "foundry", "3")
    catalog.foundry_template.assert_called_once_with(FOUNDRY, TEMPLATE, "H100_80GB")
    result = check_catalog_availability(
        "org/model", ["foundry"], foundry_profile={**PROFILE, "acceleratorsPerInstance": 8}
    )
    assert result["foundry"].status == "incompatible-profile"
    catalog.foundry_template.side_effect = ConfigurationError("Template is archived")
    result = check_catalog_availability("org/model", ["foundry"], foundry_profile=PROFILE)
    assert "archived" in result["foundry"].reason
    catalog.foundry_template.side_effect = ProviderError("foundry", "Access denied")
    result = check_catalog_availability("org/model", ["foundry"], foundry_profile=PROFILE)
    assert result["foundry"].status == "unknown"


def test_profile_cannot_substitute_another_models_asset(monkeypatch):
    catalog = Mock(resolve=Mock(return_value=FOUNDRY))
    monkeypatch.setattr("hf_cloud.core.catalog_availability.AzureCatalog", lambda: catalog)
    result = check_catalog_availability(
        "org/model",
        ["foundry"],
        foundry_profile={**PROFILE, "modelAssetId": FOUNDRY.asset_id.replace("org--model", "different-model")},
    )
    assert result["foundry"].status == "incompatible-profile"
    catalog.foundry_template.assert_not_called()


@pytest.mark.parametrize("status", ["unavailable", "unknown"])
def test_estimate_never_ranks_unavailable_or_unverified_azureml(comparison, monkeypatch, status):
    run, catalogs = comparison
    catalogs["azureml"] = load_instance_data("azureml")
    check = Mock(
        return_value={"azureml": CatalogAvailability("azureml", status, "Catalog cannot confirm this model")}
    )
    monkeypatch.setattr("hf_cloud.cli.estimate.check_catalog_availability", check)
    result = CliRunner().invoke(
        app, ["estimate", "org/model", "--provider", "azureml", "--provider", "vertex", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert all(row["provider"] == "vertex" for row in data["instances"])
    assert data["cheapest"]["provider"] == "vertex"
    assert data["catalogAvailability"]["azureml"]["status"] == status
    assert "azureml" in data["excludedProviders"]
    run.assert_called_once()
    check.assert_called_once_with("org/model", ["azureml"], foundry_profile=None)


def test_only_unavailable_provider_returns_no_recommendation_or_worker(comparison, monkeypatch):
    run, catalogs = comparison
    catalogs["azureml"] = load_instance_data("azureml")
    monkeypatch.setattr(
        "hf_cloud.cli.estimate.check_catalog_availability",
        Mock(return_value={"azureml": CatalogAvailability("azureml", "unavailable", "Not published")}),
    )
    result = CliRunner().invoke(app, ["azureml", "estimate", "org/model", "--json"])
    data = json.loads(result.stdout)
    assert result.exit_code == 1
    assert data["cheapest"] is None and data["instances"] == []
    assert "Not published" in data["error"]
    run.assert_not_called()


def test_verified_catalogs_enter_same_single_hfmem_comparison(comparison, estimate_catalog_checks):
    run, catalogs = comparison
    catalogs["azureml"] = load_instance_data("azureml")
    result = CliRunner().invoke(app, ["estimate", "org/model", "--json"])
    data = json.loads(result.stdout)
    assert result.exit_code == 0
    assert any(row["provider"] == "azureml" for row in data["instances"])
    assert data["catalogAvailability"]["azureml"]["status"] == "available"
    assert data["catalogAvailability"]["foundry"]["status"] == "available"
    assert "No verified Foundry template" in data["excludedProviders"]["foundry"]
    run.assert_called_once()
    estimate_catalog_checks.assert_called_once()


def test_failed_foundry_profile_cannot_win_cheapest(comparison, monkeypatch, tmp_path):
    run, _ = comparison
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(PROFILE))
    monkeypatch.setattr(
        "hf_cloud.cli.estimate.check_catalog_availability",
        Mock(
            return_value={
                "foundry": CatalogAvailability(
                    "foundry", "incompatible-profile", "Template not published for this model", model=FOUNDRY
                )
            }
        ),
    )
    result = CliRunner().invoke(app, ["estimate", "org/model", "--foundry-profile", str(path), "--json"])
    data = json.loads(result.stdout)
    assert result.exit_code == 0, result.stdout
    assert data["cheapest"]["provider"] != "foundry"
    assert all(row["provider"] != "foundry" for row in data["instances"])
    assert data["catalogAvailability"]["foundry"]["status"] == "incompatible-profile"


def test_availability_command_shows_both_catalogs_and_honest_unknown_exit(monkeypatch):
    reports = {
        "azureml": CatalogAvailability("azureml", "available", model=MODEL),
        "foundry": CatalogAvailability("foundry", "unknown", "Run az login"),
    }
    check = Mock(return_value=reports)
    monkeypatch.setattr("hf_cloud.cli.catalog.check_catalog_availability", check)
    result = CliRunner().invoke(app, ["catalog", "org/model", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["catalogAvailability"]["azureml"]["status"] == "available"
    assert data["catalogAvailability"]["foundry"]["status"] == "unknown"
    human = CliRunner().invoke(app, ["catalog", "org/model"])
    assert "Available" in human.stdout and "Could not verify" in human.stdout
    assert "Run az login" in human.stdout


def test_non_microsoft_estimates_do_not_query_azure(comparison, estimate_catalog_checks):
    result = CliRunner().invoke(app, ["estimate", "org/model", "--provider", "sagemaker", "--json"])
    assert result.exit_code == 0
    estimate_catalog_checks.assert_not_called()


def test_unsupported_weight_selection_skips_catalog_reads(comparison, estimate_catalog_checks):
    _, catalogs = comparison
    catalogs["azureml"] = load_instance_data("azureml")
    result = CliRunner().invoke(app, ["estimate", "org/model", "--revision", "dev", "--json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["catalogAvailability"]["azureml"]["status"] == "unsupported-selection"
    assert all(row["provider"] not in {"azureml", "foundry"} for row in data["instances"])
    estimate_catalog_checks.assert_not_called()


def test_discovery_does_not_invent_a_template_for_a_published_model(monkeypatch):
    published = PublishedModel("org/model", "azure-huggingface", "org--model", "3")
    catalog = Mock(resolve=Mock(return_value=published))
    monkeypatch.setattr("hf_cloud.core.catalog_availability.AzureCatalog", lambda: catalog)
    result = check_catalog_availability("org/model", ["foundry"])["foundry"]
    assert result.status == "no-template"
    assert result.model == published


def test_root_catalog_rejects_bad_provider_without_lookup():
    result = CliRunner().invoke(app, ["catalog", "org/model", "--provider", "sagemaker", "--json"])
    assert result.exit_code == 1
    assert "azureml and foundry" in json.loads(result.stdout)["error"]
