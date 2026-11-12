"""Foundry control-plane lifecycle and safe inference without Azure resources."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.deployment import DeploymentStatus
from hf_cloud.core.exceptions import ConfigurationError, DeploymentError, DeploymentNotFoundError, ProviderError
from hf_cloud.core.state import StateManager
from hf_cloud.providers.foundry import FoundryProvider
from hf_cloud.providers.foundry.client import FoundryClient
from hf_cloud.providers.foundry.profiles import load_profile, profile_catalog

SUB = "00000000-0000-0000-0000-000000000000"
MODEL = "azureml://registries/azure-huggingface/models/example/versions/1"
TEMPLATE = "azureml://registries/azure-huggingface/deploymenttemplates/example-h100/versions/1"
CONFIG = {"model_asset_id": MODEL, "deployment_template": TEMPLATE, "accelerator_type": "H100_80GB"}
SCOPE = ["--subscription-id", SUB, "--resource-group", "rg", "--account", "hf-foundry"]


class AzureError(Exception):
    def __init__(self, status_code):
        super().__init__(f"Azure error {status_code}")
        self.status_code = status_code


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr("hf_cloud.core.config.Config.DEFAULT_CONFIG_FILE", tmp_path / "config.json")
    for key in (
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP",
        "AZURE_FOUNDRY_ACCOUNT",
        "AZURE_FOUNDRY_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    provider = FoundryProvider(SUB, "rg", "hf-foundry")
    provider.client._management = Mock()
    provider.client._monitor = Mock()
    provider.client._credential = Mock()
    provider.client._credential.get_token.return_value.token = "test-token"
    provider.client._account = SimpleNamespace(
        kind="AIServices",
        location="eastus",
        properties=SimpleNamespace(custom_sub_domain_name="hf-foundry"),
    )
    return provider


def resource(provider, *, state="Succeeded", count=1, name="test", message=None):
    return SimpleNamespace(
        id=provider.client.resource_id(name),
        name=name,
        sku=SimpleNamespace(capacity=count),
        properties=SimpleNamespace(
            model=MODEL,
            deployment_template=TEMPLATE,
            accelerator_type="H100_80GB",
            version_upgrade_option="NoAutoUpgrade",
            accelerators_per_instance=1,
            provisioning_state=state,
            provisioning_details=SimpleNamespace(as_dict=lambda: {"message": message}) if message else None,
        ),
    )


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "modelId": "org/model",
                "modelAssetId": MODEL,
                "deploymentTemplateId": TEMPLATE,
                "acceleratorType": "H100_80GB",
                "acceleratorsPerInstance": 1,
                "maxModelLen": 2048,
                "pricePerInstanceHourUsd": 0.5,
                "pricingSource": "https://example.com/pricing",
                "pricingObservedAt": "2026-11-11",
            }
        )
    )
    return path


def test_deploy_uses_template_sku_and_verifies_succeeded(provider):
    operations = provider.client.management.managed_compute_deployments
    operations.get.side_effect = [AzureError(404), resource(provider)]
    deployment = provider.deploy("org/model", "test", CONFIG)
    body = operations.begin_create_or_update.call_args.kwargs["resource"]
    assert body == {
        "sku": {"name": "GlobalManagedCompute", "capacity": 1},
        "properties": {
            "model": MODEL,
            "deploymentTemplate": TEMPLATE,
            "acceleratorType": "H100_80GB",
            "versionUpgradeOption": "NoAutoUpgrade",
        },
    }
    assert "tags" not in body  # Preview SDK's ProxyResource doesn't expose tags.
    assert deployment.status == DeploymentStatus.RUNNING
    assert deployment.model_id == "org/model"
    assert deployment.deployment_id == provider.client.resource_id("test")
    assert deployment.instance_type == "H100_80GB" and deployment.instance_count == 1
    assert operations.get.call_count == 2
    operations.begin_create_or_update.return_value.result.assert_called_once()


@pytest.mark.parametrize("state", ["Failed", "Canceled", "Creating", "Unexpected"])
def test_pollers_do_not_imply_success(provider, state):
    operations = provider.client.management.managed_compute_deployments
    operations.get.side_effect = [AzureError(404), resource(provider, state=state, message="Quota unavailable")]
    with pytest.raises(DeploymentError, match="Quota unavailable"):
        provider.deploy("org/model", "test", CONFIG)


def test_existing_deployment_is_never_overwritten_by_deploy(provider):
    operations = provider.client.management.managed_compute_deployments
    operations.get.return_value = resource(provider)
    with pytest.raises(DeploymentError, match="already exists"):
        provider.deploy("org/model", "test", CONFIG)
    operations.begin_create_or_update.assert_not_called()


def test_failed_read_does_not_start_billable_creation(provider):
    operations = provider.client.management.managed_compute_deployments
    operations.get.side_effect = AzureError(403)
    with pytest.raises(ProviderError):
        provider.deploy("org/model", "test", CONFIG)
    operations.begin_create_or_update.assert_not_called()


@pytest.mark.parametrize(
    "update",
    [
        {"model_asset_id": "not-a-published-model"},
        {"deployment_template": "guess"},
        {"accelerator_type": "Standard_NC24"},
        {"instance_count": 0},
        {"instance_count": True},
        {"version_upgrade_option": "guess"},
    ],
)
def test_invalid_specs_fail_before_azure(provider, update):
    with pytest.raises(ConfigurationError):
        provider.deploy("org/model", "test", {**CONFIG, **update})
    assert not provider.client.management.managed_compute_deployments.mock_calls


def test_hub_token_is_not_sent_or_persisted(provider):
    with pytest.raises(ConfigurationError, match="Hub tokens"):
        provider.deploy("org/model", "test", CONFIG, token="must-not-leave-process")
    assert not provider.client.management.managed_compute_deployments.mock_calls


def test_invalid_account_kind_fails_before_creating(provider):
    provider.client._account.kind = "OpenAI"
    with pytest.raises(ConfigurationError, match="AIServices"):
        provider.deploy("org/model", "test", CONFIG)
    provider.client.management.managed_compute_deployments.begin_create_or_update.assert_not_called()


def test_scaling_patches_only_capacity_and_waits(provider):
    operations = provider.client.management.managed_compute_deployments
    operations.get.return_value = resource(provider, count=3)
    assert provider.update_deployment("test", {"instance_count": 3}).instance_count == 3
    assert operations.begin_update.call_args.kwargs["properties"] == {
        "sku": {"name": "GlobalManagedCompute", "capacity": 3}
    }
    operations.begin_create_or_update.assert_not_called()
    operations.begin_update.return_value.result.assert_called_once()


def test_delete_waits_and_does_not_delete_account(provider):
    assert provider.delete_deployment("test")
    operations = provider.client.management.managed_compute_deployments
    operations.begin_delete.assert_called_once_with(
        resource_group_name="rg", account_name="hf-foundry", deployment_name="test"
    )
    operations.begin_delete.return_value.result.assert_called_once()
    provider.client.management.accounts.begin_delete.assert_not_called()


def test_list_and_404_mapping(provider):
    operations = provider.client.management.managed_compute_deployments
    operations.list.return_value = [resource(provider), resource(provider, state="Failed", name="bad")]
    assert len(provider.list_deployments({"status": "running"})) == 1
    operations.get.side_effect = AzureError(404)
    with pytest.raises(DeploymentNotFoundError):
        provider.get_deployment("missing")


def test_qualified_ids_prevent_account_state_collisions(provider, tmp_path):
    state = StateManager(tmp_path / "state.json")
    first = provider._deployment(resource(provider))
    provider.client.account_name = "second-account"
    second = provider._deployment(resource(provider))
    state.add_deployments([first, second])
    assert len(state.list_deployments("foundry")) == 2
    with pytest.raises(ConfigurationError, match="different Foundry account"):
        provider.client.scope(first.deployment_id)


def test_invoke_uses_entra_and_forces_deployment_name_without_following_redirects(provider, monkeypatch):
    provider.client.management.managed_compute_deployments.get.return_value = resource(provider)
    http = Mock()
    http.post.return_value = httpx.Response(200, json={"choices": [{"message": {"content": "Paris"}}]})
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=http)
    constructor.return_value.__exit__ = Mock()
    monkeypatch.setattr(httpx, "Client", constructor)
    assert "choices" in provider.invoke(provider.client.resource_id("test"), {"messages": [], "model": "wrong"})
    args = http.post.call_args
    assert args.args[0] == "https://hf-foundry.services.ai.azure.com/openai/v1/chat/completions"
    assert args.kwargs["json"]["model"] == "test"
    assert args.kwargs["headers"] == {"Authorization": "Bearer test-token"}
    provider.client.credential.get_token.assert_called_once_with("https://cognitiveservices.azure.com/.default")
    assert constructor.call_args.kwargs["follow_redirects"] is False


@pytest.mark.parametrize(
    "route", ["https://elsewhere.example", "../secrets", "/chat/completions", "chat?secret"]
)
def test_inference_rejects_arbitrary_routes_before_request(provider, route):
    with pytest.raises(ConfigurationError, match="relative path"):
        provider.invoke_route("test", {}, route)
    provider.client.credential.get_token.assert_not_called()


def test_activity_logs_are_scoped_and_bounded(provider):
    provider.client.monitor.activity_logs.list.return_value = [
        SimpleNamespace(as_dict=lambda: {"operation": "create"}) for _ in range(10)
    ]
    assert len(provider.get_logs("test", 2).splitlines()) == 2
    assert (
        provider.client.resource_id("test")
        in provider.client.monitor.activity_logs.list.call_args.kwargs["filter"]
    )


def test_cli_dry_run_needs_no_azure_client_or_saved_state(provider, profile, monkeypatch):
    monkeypatch.setattr("hf_cloud.cli.foundry.StateManager", Mock(side_effect=AssertionError("No state write")))
    monkeypatch.setattr("hf_cloud.cli.foundry.FoundryProvider", Mock(return_value=provider))
    result = CliRunner().invoke(
        app,
        ["foundry", *SCOPE, "deploy", "org/model", "--name", "test", "--profile", str(profile), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["resource"]["properties"]["deploymentTemplate"] == TEMPLATE
    assert data["scope"]["account_name"] == "hf-foundry"
    assert not provider.client.management.managed_compute_deployments.mock_calls


def test_profile_model_and_price_validation(profile):
    with pytest.raises(ConfigurationError, match="modelId"):
        load_profile(profile, "other/model")
    data = json.loads(profile.read_text())
    data["pricePerInstanceHourUsd"] = True
    profile.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="nonnegative"):
        load_profile(profile, "org/model")


def test_profile_gpu_count_is_per_model_instance(profile):
    data = json.loads(profile.read_text())
    data["acceleratorsPerInstance"] = 2
    data["pricePerInstanceHourUsd"] = 15.82
    profile.write_text(json.dumps(data))
    instance = profile_catalog(profile, "org/model").instances[0]
    assert instance["numGpu"] == 2
    assert instance["pricePerHourUsd"] == 15.82


def test_foundry_estimate_shares_comparison_and_cheapest_hint(comparison, profile):
    run, _ = comparison
    result = CliRunner().invoke(app, ["estimate", "org/model", "--foundry-profile", str(profile), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["providers"] == ["sagemaker", "vertex", "foundry"]
    assert data["cheapest"]["provider"] == "foundry"
    assert data["cheapest"]["pricingSource"] == "https://example.com/pricing"
    assert data["pricing"]["source"] == "catalogs-and-profiles"
    run.assert_called_once()
    filtered = CliRunner().invoke(app, ["foundry", "estimate", "org/model", "--profile", str(profile)])
    assert filtered.exit_code == 0, filtered.output
    assert "hf-cloud foundry deploy org/model" in filtered.stdout
    assert "--profile" in filtered.stdout


def test_template_context_limit_excludes_incompatible_foundry_match(comparison, profile):
    data = json.loads(profile.read_text())
    data["maxModelLen"] = 512
    profile.write_text(json.dumps(data))
    result = CliRunner().invoke(app, ["estimate", "org/model", "--profile", str(profile), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert all(row["provider"] != "foundry" for row in data["instances"])
    assert any("supports at most 512" in warning for warning in data["warnings"])


def test_foundry_estimate_does_not_guess_templates(comparison):
    run, _ = comparison
    result = CliRunner().invoke(app, ["foundry", "estimate", "org/model", "--json"])
    assert result.exit_code == 1
    assert "require --foundry-profile" in json.loads(result.stdout)["error"]
    run.assert_not_called()


@pytest.mark.parametrize(
    "selector", [["--format", "gguf"], ["--revision", "dev"], ["--file", "one.safetensors"]]
)
def test_foundry_profiles_do_not_claim_compatibility_for_different_weights(comparison, profile, selector):
    run, _ = comparison
    result = CliRunner().invoke(
        app, ["foundry", "estimate", "org/model", "--profile", str(profile), "--json", *selector]
    )
    assert result.exit_code == 1
    assert "cannot follow" in json.loads(result.stdout)["error"]
    run.assert_not_called()


def test_cli_failed_deploy_never_writes_success_state(provider, profile, monkeypatch):
    provider.deploy = Mock(side_effect=DeploymentError("Azure quota exceeded"))
    monkeypatch.setattr("hf_cloud.cli.foundry.FoundryProvider", Mock(return_value=provider))
    monkeypatch.setattr("hf_cloud.cli.foundry.SdkUpdateCheck", Mock())
    state = Mock()
    monkeypatch.setattr("hf_cloud.cli.foundry.StateManager", state)
    result = CliRunner().invoke(
        app, ["foundry", "deploy", "org/model", "--name", "test", "--profile", str(profile)]
    )
    assert result.exit_code == 1
    assert "quota exceeded" in result.stdout and "successfully" not in result.stdout
    state.assert_not_called()


def test_endpoint_configuration_is_bound_to_its_account(provider, monkeypatch):
    defaults = {
        "subscription_id": SUB,
        "resource_group": "rg",
        "account_name": "one",
        "endpoint": "https://one.example.com",
    }
    monkeypatch.setattr("hf_cloud.core.config.Config.get_provider_config", lambda *args: defaults)
    assert FoundryClient().endpoint_override == "https://one.example.com"
    assert FoundryClient(account_name="two").endpoint_override is None


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://account.example",
        "https://user:password@example.com",
        "https://example.com/?key=secret",
        "https://example.com/openai/v1",
    ],
)
def test_endpoint_rejects_credentials_query_and_nonbase_urls(provider, endpoint):
    provider.client.endpoint_override = endpoint
    with pytest.raises(ConfigurationError):
        provider.client.endpoint


def test_default_profile_is_valid_for_documented_dry_run(provider):
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples/foundry/qwen3-32b-h100.json"
    data = load_profile(path, "Qwen/Qwen3-32B")
    assert data["acceleratorType"] == "H100_80GB"
    assert profile_catalog(path, "Qwen/Qwen3-32B").instances[0]["numGpu"] == 1


def test_hub_only_deploy_resolves_publisher_defaults(provider, monkeypatch):
    from hf_cloud.providers.azure_catalog import PublishedModel

    published = PublishedModel(
        "org/model", "azure-huggingface", "example", "1", "text-generation", TEMPLATE, (TEMPLATE,)
    )
    provider.catalog = Mock()
    provider.catalog.resolve.return_value = published
    provider.catalog.foundry_template.return_value = {**CONFIG, "accelerators_per_instance": 1}
    operations = provider.client.management.managed_compute_deployments
    operations.get.side_effect = [AzureError(404), resource(provider)]
    result = provider.deploy("org/model", "test", {})
    assert result.status == DeploymentStatus.RUNNING
    provider.catalog.resolve.assert_called_once_with("org/model", "foundry", None)
    assert operations.begin_create_or_update.call_args.kwargs["resource"]["properties"]["model"] == MODEL


def test_cli_hub_only_dry_run_resolves_assets_without_provisioning(provider, monkeypatch):
    from hf_cloud.providers.azure_catalog import PublishedModel

    provider.catalog = Mock()
    provider.catalog.resolve.return_value = PublishedModel("org/model", "azure-huggingface", "example", "1")
    provider.catalog.foundry_template.return_value = {**CONFIG, "accelerators_per_instance": 1}
    monkeypatch.setattr("hf_cloud.cli.foundry.FoundryProvider", Mock(return_value=provider))
    result = CliRunner().invoke(app, ["foundry", *SCOPE, "deploy", "org/model", "--name", "test", "--dry-run"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["resource"]["properties"]["deploymentTemplate"] == TEMPLATE
    provider.client.management.managed_compute_deployments.begin_create_or_update.assert_not_called()


def test_discovery_failure_stops_before_billable_operations(provider):
    provider.catalog = Mock()
    provider.catalog.resolve.side_effect = ConfigurationError("Model is not published")
    with pytest.raises(ConfigurationError, match="not published"):
        provider.deploy("org/model", "test", {})
    provider.client.management.managed_compute_deployments.begin_create_or_update.assert_not_called()
