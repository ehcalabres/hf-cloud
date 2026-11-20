"""Discover published template layouts and price them without per-model profiles."""

import json
import shlex
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.catalog_availability import check_catalog_availability
from hf_cloud.providers.azure_catalog import AzureCatalog
from hf_cloud.providers.foundry import FoundryProvider
from hf_cloud.providers.foundry.pricing import template_catalog

PREFIX = "azureml://registries/azure-huggingface/deploymenttemplates/"


@pytest.fixture
def catalog(monkeypatch):
    client = Mock()
    catalog = AzureCatalog()
    catalog._clients["azure-huggingface"] = client
    references = [PREFIX + name + "/versions/3" for name in ("a100", "h100", "h200", "missing-context")]
    references.insert(1, PREFIX + "a100/labels/latest")
    client.models.get.return_value = NS(
        name="org--model",
        version="7",
        tags={"task": "chat-completion"},
        properties={},
        default_deployment_template=NS(asset_id=references[0]),
        allowed_deployment_templates=[NS(asset_id=ref) for ref in references],
    )
    templates = {}
    for name, accelerator, count in (
        ("a100", "A100_80GB", 1),
        ("h100", "H100_80GB", 2),
        ("h200", "H200_141GB", 1),
        ("missing-context", "A100_80GB", 1),
    ):
        templates[name] = NS(
            name=name,
            version="3",
            stage="Active",
            environment_variables={} if name == "missing-context" else {"AZ_VLLM_MAX_MODEL_LEN": "4096"},
            accelerator_maps=[
                NS(
                    accelerator_type=accelerator,
                    number_of_accelerators_per_model_instance=count,
                    default=True,
                )
            ],
        )
    client.deployment_templates.get.side_effect = lambda name, **_: templates[name]
    monkeypatch.setattr(
        "hf_cloud.providers.azure_catalog.lookup_foundry_model",
        lambda _: (
            "azure-huggingface",
            "org--model",
            "7",
        ),
    )
    monkeypatch.setattr("hf_cloud.core.catalog_availability.AzureCatalog", lambda: catalog)
    return catalog, templates


def test_published_layouts_join_shared_accelerator_rates_and_deduplicate_labels(catalog):
    source, _ = catalog
    model = source.resolve("org/model", "foundry")
    result, warnings = template_catalog(source, model)
    assert len(result.instances) == 3
    rows = {row["acceleratorType"]: row for row in result.instances}
    assert rows["A100_80GB"]["pricePerHourUsd"] == 3.95
    assert rows["H100_80GB"]["pricePerHourUsd"] == 15.82
    assert rows["H100_80GB"]["numGpu"] == 2
    assert rows["H200_141GB"]["pricePerHourUsd"] == 9.10
    assert all("/versions/3" in row["deploymentTemplateId"] for row in result.instances)
    assert result.pricing_region == "GlobalManagedCompute"
    assert result.last_updated == "2026-11-13"
    assert result.pricing_source == "https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/microsoft/"
    assert len(warnings) == 1 and "no machine-readable context limit" in warnings[0]


def test_archived_or_invalid_layouts_are_not_recommended(catalog):
    source, templates = catalog
    templates["a100"].stage = "Archived"
    templates["h100"].accelerator_maps[0].number_of_accelerators_per_model_instance = 0
    result, warnings = template_catalog(source, source.resolve("org/model", "foundry"))
    assert [row["acceleratorType"] for row in result.instances] == ["H200_141GB"]
    assert any("archived" in warning for warning in warnings)
    assert any("accelerator count" in warning for warning in warnings)


def test_discovery_is_opt_in_and_preserves_publication_when_templates_cannot_be_estimated(catalog):
    source, templates = catalog
    result = check_catalog_availability("org/model", ["foundry"])["foundry"]
    assert result.status == "available" and result.estimate_catalog is None
    source.registry("azure-huggingface").deployment_templates.get.assert_not_called()
    for template in templates.values():
        template.environment_variables = {}
    result = check_catalog_availability("org/model", ["foundry"], include_estimates=True)["foundry"]
    assert result.status == "available"
    assert result.estimate_catalog.instances == []
    assert result.to_dict()["estimateWarnings"]


def test_profile_free_estimate_ranks_foundry_in_global_table_and_uses_one_memory_session(
    comparison,
    catalog,
    monkeypatch,
):
    worker, _ = comparison
    monkeypatch.setattr("hf_cloud.cli.estimate.check_catalog_availability", check_catalog_availability)
    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "org/model",
            "--provider",
            "sagemaker",
            "--provider",
            "foundry",
            "--hours",
            "24",
            "--instance-count",
            "2",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    foundry = [row for row in data["instances"] if row["provider"] == "foundry"]
    assert len(foundry) == 3
    assert foundry[0]["pricePerHourUsd"] == 3.95
    assert foundry[0]["estimatedCostUsd"] == 189.6
    h200 = next(row for row in foundry if row["acceleratorType"] == "H200_141GB")
    assert h200["pricePerHourUsd"] == 9.10
    assert h200["estimatedCostUsd"] == 436.8
    assert data["instances"][-1]["pricePerHourUsd"] is None
    assert data["catalogAvailability"]["foundry"]["estimateCandidateCount"] == 3
    assert data["pricing"]["source"] == "bundled-catalogs"
    assert "foundry" not in data.get("excludedProviders", {})
    worker.assert_called_once()


def test_automatic_foundry_deploy_hint_pins_estimated_assets_and_replica_count(
    comparison, catalog, monkeypatch
):
    monkeypatch.setattr("hf_cloud.cli.estimate.check_catalog_availability", check_catalog_availability)
    result = CliRunner().invoke(app, ["foundry", "estimate", "org/model", "--instance-count", "2"])
    assert result.exit_code == 0, result.stdout
    hint = result.stdout.split("Deploy cheapest instance\n", 1)[1].split("The published template", 1)[0]
    tokens = shlex.split(hint.replace("\\\n", ""))
    assert "--profile" not in tokens
    assert tokens[tokens.index("--model-asset-id") + 1].endswith("/org--model/versions/7")
    assert tokens[tokens.index("--template") + 1] == PREFIX + "a100/versions/3"
    assert tokens[tokens.index("--accelerator-type") + 1] == "A100_80GB"
    assert tokens[tokens.index("--instance-count") + 1] == "2"
    # Verify this hint is accepted by the actual deploy command, without an ARM write.
    provider = FoundryProvider("sub", "rg", "account")
    provider.client = Mock()
    provider.client.scope.return_value = {"account_name": "account", "resource_group_name": "rg"}
    provider.client.resource_id.return_value = "/test/resource"
    provider.deploy = Mock()
    provider.catalog = catalog[0]
    monkeypatch.setattr("hf_cloud.cli.foundry.FoundryProvider", lambda *_: provider)
    deployed = CliRunner().invoke(app, [*tokens[1:], "--dry-run"])
    assert deployed.exit_code == 0, deployed.stdout
    provider.deploy.assert_not_called()


def test_context_cap_still_excludes_incompatible_published_template(comparison, catalog, monkeypatch):
    _, templates = catalog
    templates["a100"].environment_variables["AZ_VLLM_MAX_MODEL_LEN"] = "512"
    monkeypatch.setattr("hf_cloud.cli.estimate.check_catalog_availability", check_catalog_availability)
    result = CliRunner().invoke(app, ["foundry", "estimate", "org/model", "--json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["cheapest"]["acceleratorType"] == "H200_141GB"
    assert any("supports at most 512" in warning for warning in data["warnings"])
