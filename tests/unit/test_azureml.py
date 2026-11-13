"""Catalog-only Azure ML deployment plans and endpoint lifecycle."""

import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.deployment import DeploymentStatus
from hf_cloud.core.exceptions import ConfigurationError, DeploymentError, ProviderError
from hf_cloud.core.state import StateManager
from hf_cloud.providers.azure_catalog import PublishedModel
from hf_cloud.providers.azureml import AzureMLProvider
from hf_cloud.providers.azureml.client import AzureMLClient
from hf_cloud.providers.azureml.serving import DEFAULT_INSTANCE, serving_plan
from hf_cloud.utils.hf_mem import load_instance_data

SUB = "00000000-0000-0000-0000-000000000000"
SCOPE = ["--subscription-id", SUB, "--resource-group", "rg", "--workspace", "workspace"]
PUBLISHED = PublishedModel("org/model", "HuggingFace", "org-model", "7", "text-generation")


class AzureError(Exception):
    def __init__(self, status):
        self.status_code = status
        super().__init__(f"Azure error {status}")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr("hf_cloud.core.config.Config.DEFAULT_CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(StateManager, "DEFAULT_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setenv("HF_CLOUD_NO_UPDATE_CHECK", "1")
    for key in ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZUREML_WORKSPACE_NAME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hf_cloud.providers.azure_catalog.AzureCatalog.resolve", Mock(return_value=PUBLISHED))


@pytest.fixture
def provider():
    provider = AzureMLProvider(SUB, "rg", "workspace")
    provider.client._management = Mock()
    provider.client._workspace = NS(location="eastus")
    return provider


def records(provider, config=None, state="Succeeded"):
    plan = serving_plan(PUBLISHED, "test", config or {})
    endpoint = NS(
        **plan["endpoint"],
        id=provider.client.resource_id("test"),
        traffic=plan["traffic"],
        provisioning_state="Succeeded",
        scoring_uri="https://example.test/score",
    )
    component = NS(**plan["deployment"], provisioning_state=state)
    return endpoint, component


def ready(provider, config=None, state="Succeeded"):
    endpoint, component = records(provider, config, state)
    client = provider.client.management
    client.online_endpoints.get.return_value = endpoint
    client.online_deployments.get.return_value = component
    return endpoint, component


def test_plan_uses_only_published_model_and_secret_reference():
    plan = serving_plan(PUBLISHED, "test", {"hf_token_connection": "hub-token"})
    assert plan["deployment"]["model"] == PUBLISHED.asset_id
    assert (
        plan["deployment"]["environment_variables"]["HF_TOKEN"]
        == "${{azureml://connections/hub-token/credentials/HF_TOKEN}}"
    )
    assert plan["endpoint"]["properties"]["enforce_access_to_default_secret_stores"] == "enabled"
    assert plan["endpoint"]["auth_mode"] == "aad_token"
    assert "environment" not in plan and "environment" not in plan["deployment"]
    assert "code_configuration" not in plan["deployment"]
    with pytest.raises(ConfigurationError, match="raw Hub tokens"):
        serving_plan(PUBLISHED, "test", {}, token="hf_secret")


@pytest.mark.parametrize(
    "config",
    [
        {"image": "vllm/vllm-openai:latest"},
        {"revision": "main"},
        {"max_model_len": 1024},
        {"task": "text-generation"},
        {"gpu_count": 1},
        {"hf_token": "hf_secret"},
        {"safety_margin": 0.1},
        {"instance_count": 0},
        {"instance_count": True},
        {"instance_type": "invalid"},
        {"hf_token_connection": "../../invalid"},
    ],
)
def test_invalid_or_custom_settings_fail_before_catalog_read(provider, config):
    provider.catalog.resolve.reset_mock()
    with pytest.raises(ConfigurationError):
        provider.prepare_deployment("org/model", "test", config)
    provider.catalog.resolve.assert_not_called()
    provider.client.management.online_endpoints.begin_create_or_update.assert_not_called()


def test_workspace_scope_rejects_cross_workspace_ids(provider):
    with pytest.raises(ConfigurationError, match="different Azure ML workspace"):
        provider.client.endpoint_name(AzureMLClient(SUB, "rg", "other").resource_id("test"))
    assert provider.client.endpoint_name(provider.client.resource_id("test")) == "test"
    with pytest.raises(ConfigurationError, match="Missing Azure ML"):
        AzureMLClient().require_scope()


def test_state_keeps_equal_names_in_different_workspaces(provider):
    state = StateManager()
    endpoint, component = records(provider)
    state.add_deployment(provider._deployment(endpoint, component))
    provider.client.workspace_name = "other"
    endpoint.id = provider.client.resource_id("test")
    state.add_deployment(provider._deployment(endpoint, component))
    assert len(state.list_deployments("azureml")) == 2


def test_scale_logs_invoke_and_delete_preserve_catalog_contract(provider):
    endpoint, component = ready(provider)
    assert provider.update_deployment("test", {"instance_count": 2}).instance_count == 2
    assert endpoint.traffic == {"main": 100} and component.model == PUBLISHED.asset_id
    provider.get_logs("test", 42, container_type="storage-initializer")
    provider.client.management.online_deployments.get_logs.assert_called_once_with(
        name="main", endpoint_name="test", lines=42, container_type="storage-initializer"
    )
    files = []
    payload = {"inputs": "Hello", "parameters": {"max_new_tokens": 10}}

    def invoke(**kwargs):
        file = Path(kwargs["request_file"])
        files.append(file)
        assert file.stat().st_mode & 0o077 == 0
        assert json.loads(file.read_text()) == payload
        assert kwargs["deployment_name"] == "main"
        return '{"generated_text": "Hello world"}'

    provider.client.management.online_endpoints.invoke.side_effect = invoke
    assert provider.invoke("test", payload) == {"generated_text": "Hello world"}
    assert not files[0].exists()
    assert provider.delete_deployment("test")
    provider.client.management.online_endpoints.begin_delete.assert_called_once_with(name="test")
    provider.client.management.workspaces.begin_delete.assert_not_called()


def test_failed_or_ambiguous_endpoint_cannot_invoke(provider):
    endpoint, component = ready(provider, state="Failed")
    assert provider.get_status("test") == DeploymentStatus.FAILED
    with pytest.raises(ProviderError, match="not running"):
        provider.invoke("test", {"inputs": "Hello"})
    provider.client.management.online_endpoints.invoke.assert_not_called()
    endpoint.properties = {}
    endpoint.traffic = {"blue": 50, "green": 50}
    provider.client.management.online_deployments.list.return_value = [component, component]
    with pytest.raises(ConfigurationError, match="multiple"):
        provider.get_deployment("test")
    provider.client.deployment_name = "main"
    assert provider.get_deployment("test").status == DeploymentStatus.FAILED


def test_dry_run_reads_catalog_but_never_provisions_and_can_export(monkeypatch, tmp_path):
    deploy = Mock(side_effect=AssertionError("No provisioning"))
    monkeypatch.setattr(AzureMLProvider, "deploy", deploy)
    directory = tmp_path / "export"
    result = CliRunner().invoke(
        app,
        [
            "azureml",
            *SCOPE,
            "deploy",
            "org/model",
            "--name",
            "test",
            "--dry-run",
            "--export-dir",
            str(directory),
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["deployment"]["model"] == PUBLISHED.asset_id
    assert data["resourceId"].endswith("/workspaces/workspace/onlineEndpoints/test")
    assert json.loads((directory / "deployment.json").read_text())["model"] == PUBLISHED.asset_id
    assert json.loads((directory / "endpoint.json").read_text())["auth_mode"] == "aad_token"
    deploy.assert_not_called()
    assert not StateManager.DEFAULT_STATE_FILE.exists()
    again = CliRunner().invoke(
        app,
        [
            "azureml",
            *SCOPE,
            "deploy",
            "org/model",
            "--name",
            "test",
            "--dry-run",
            "--export-dir",
            str(directory),
        ],
    )
    assert again.exit_code == 1 and "overwrite" in again.stdout


def test_failed_cli_deploy_never_records_or_prints_success(monkeypatch):
    monkeypatch.setattr(AzureMLProvider, "deploy", Mock(side_effect=DeploymentError("GPU allocation failed")))
    result = CliRunner().invoke(app, ["azureml", *SCOPE, "deploy", "org/model", "--name", "test"])
    assert result.exit_code == 1
    assert "GPU allocation failed" in result.stdout and "successfully" not in result.stdout
    assert not StateManager.DEFAULT_STATE_FILE.exists()


def test_unsupported_model_fails_before_deployment_or_export(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hf_cloud.providers.azure_catalog.AzureCatalog.resolve",
        Mock(side_effect=ConfigurationError("Not in the published catalog")),
    )
    deploy = Mock()
    monkeypatch.setattr(AzureMLProvider, "deploy", deploy)
    directory = tmp_path / "export"
    result = CliRunner().invoke(
        app, ["azureml", *SCOPE, "deploy", "org/model", "--name", "test", "--export-dir", str(directory)]
    )
    assert result.exit_code == 1 and "published catalog" in result.stdout
    deploy.assert_not_called()
    assert not directory.exists()


def test_estimates_verify_publication_and_preserve_catalog_runtime(comparison):
    run, catalogs = comparison
    catalogs["azureml"] = load_instance_data("azureml")
    result = CliRunner().invoke(app, ["estimate", "org/model", "--provider", "azureml", "--json"])
    data = json.loads(result.stdout)
    assert result.exit_code == 0 and data["cheapest"]["provider"] == "azureml"
    assert data["catalogAvailability"]["azureml"]["status"] == "available"
    result = CliRunner().invoke(app, ["azureml", "estimate", "org/model", "--max-model-len", "1024"])
    assert result.exit_code == 0
    output = result.stdout.replace("\\\n  ", " ")
    assert "hf-cloud azureml deploy org/model" in output
    assert "--max-model-len 1024" not in output
    assert "catalog model availability verified" in output.lower()
    assert run.call_count == 2


def test_incomplete_endpoint_remains_visible_and_deletable(provider):
    endpoint, _ = ready(provider)
    client = provider.client.management
    client.online_deployments.get.side_effect = AzureError(404)
    client.online_endpoints.list.return_value = [endpoint]
    deployment = provider.get_deployment("test")
    assert deployment.status == DeploymentStatus.FAILED and deployment.model_id == "org/model"
    assert "missing" in deployment.config["failure_reason"]
    assert len(provider.list_deployments()) == 1
    with pytest.raises(DeploymentError, match="missing"):
        provider.update_deployment("test", {"instance_count": 2})
    assert provider.delete_deployment("test")


def test_refresh_visits_saved_workspaces_and_default(provider, monkeypatch):
    state = StateManager()
    endpoint, component = records(provider)
    state.add_deployment(provider._deployment(endpoint, component))
    monkeypatch.setattr(
        "hf_cloud.providers.azureml.client.AzureMLClient",
        lambda: NS(subscription_id=SUB, resource_group="rg", workspace_name="default"),
    )
    factory = Mock(return_value=Mock(list_deployments=Mock(return_value=[])))
    monkeypatch.setattr("hf_cloud.providers.azureml.AzureMLProvider", factory)
    result = CliRunner().invoke(app, ["ls", "--refresh", "--provider", "azureml"])
    assert result.exit_code == 0, result.stdout
    assert {call.kwargs["workspace_name"] for call in factory.call_args_list} == {"workspace", "default"}


def test_configure_saves_workspace_and_instance():
    result = CliRunner().invoke(app, ["providers", "configure", "azureml"], input=f"{SUB}\nrg\nworkspace\n\n")
    assert result.exit_code == 0, result.stdout
    from hf_cloud.core.config import Config

    assert Config().get_provider_config("azureml")["workspace_name"] == "workspace"


def test_invoke_convenience_uses_published_scoring_schema(provider, monkeypatch):
    ready(provider)
    provider.invoke = Mock(return_value={"generated_text": "ok"})
    monkeypatch.setattr("hf_cloud.cli.azureml.AzureMLProvider", Mock(return_value=provider))
    result = CliRunner().invoke(app, ["azureml", "invoke", "test", "--input", "Hello"])
    assert result.exit_code == 0, result.stdout
    provider.invoke.assert_called_once_with("test", {"inputs": "Hello", "parameters": {"max_new_tokens": 100}})
