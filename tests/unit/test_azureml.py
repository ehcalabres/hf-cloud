"""Hub-first plans, workspace isolation and Azure ML endpoint lifecycle."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.deployment import DeploymentStatus
from hf_cloud.core.exceptions import ConfigurationError, DeploymentError, HFCloudError, ProviderError
from hf_cloud.core.state import StateManager
from hf_cloud.providers.azureml import AzureMLProvider
from hf_cloud.providers.azureml.client import AzureMLClient
from hf_cloud.providers.azureml.runtime.start import command
from hf_cloud.providers.azureml.serving import DEFAULT_INSTANCE, serving_plan
from hf_cloud.utils.hf_mem import load_instance_data

SUB = "00000000-0000-0000-0000-000000000000"
SCOPE = ["--subscription-id", SUB, "--resource-group", "rg", "--workspace", "workspace"]


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


@pytest.fixture
def provider():
    provider = AzureMLProvider(SUB, "rg", "workspace")
    provider.client._management = Mock()
    provider.client._workspace = SimpleNamespace(location="eastus")
    return provider


def records(provider, config=None, state="Succeeded"):
    plan = serving_plan("org/model", "test", config or {})
    endpoint = SimpleNamespace(
        **plan["endpoint"],
        id=provider.client.resource_id("test"),
        traffic=plan["traffic"],
        provisioning_state="Succeeded",
        scoring_uri="https://example.test/score",
    )
    component = SimpleNamespace(**plan["deployment"], provisioning_state=state)
    return endpoint, component


def ready(provider, config=None, state="Succeeded"):
    endpoint, component = records(provider, config, state)
    client = provider.client.management
    client.online_endpoints.get.return_value = endpoint
    client.online_deployments.get.return_value = component
    return endpoint, component


def test_plan_loads_hub_id_directly_and_secret_reference():
    plan = serving_plan(
        "org/model", "test", {"hf_token_connection": "hub-token", "revision": "branch;echo oops"}
    )
    env = plan["deployment"]["environment_variables"]
    assert env["HF_TOKEN"] == "${{azureml://connections/hub-token/credentials/HF_TOKEN}}"
    assert plan["endpoint"]["properties"]["enforce_access_to_default_secret_stores"] == "enabled"
    assert plan["endpoint"]["auth_mode"] == "aad_token"
    args = command(env)
    assert args[args.index("--model") + 1] == "org/model"
    assert args[args.index("--revision") + 1] == "branch;echo oops"
    assert not any("HF_TOKEN" in arg for arg in args)
    assert "--trust-remote-code" not in args
    with pytest.raises(ConfigurationError, match="raw Hub tokens"):
        serving_plan("org/model", "test", {}, token="hf_secret")
    with pytest.raises(ConfigurationError, match="Unsupported"):
        serving_plan("org/model", "test", {"hf_token": "hf_secret"})


@pytest.mark.parametrize(
    "config",
    [
        {"gpu_count": 0},
        {"gpu_count": True},
        {"gpu_count": 2},
        {"instance_count": 0},
        {"instance_type": "unknown"},
        {"task": "invalid"},
        {"dtype": "invalid"},
        {"kv_cache_dtype": "invalid"},
        {"safety_margin": 1},
        {"safety_margin": float("nan")},
        {"image": "vllm/vllm-openai"},
        {"image": "image:tag\nRUN echo unsafe"},
        {"max_model_len": 0},
        {"hf_token_connection": "../../invalid"},
    ],
)
def test_invalid_settings_fail_before_authentication(config):
    with pytest.raises(HFCloudError):
        serving_plan("org/model", "test", config)


def test_task_routes_have_distinct_reusable_environment_versions():
    plans = [
        serving_plan("org/model", "test", {"task": task})
        for task in ("text-generation", "chat-completion", "feature-extraction")
    ]
    assert len({p["environment"]["version"] for p in plans}) == 3
    assert plans[-1]["environment"]["inference_config"]["scoring_route"]["path"] == "/v1/embeddings"
    assert "--runner" in command(plans[-1]["deployment"]["environment_variables"])
    assert (
        serving_plan("other/model", "other", {})["environment"]["version"] == plans[0]["environment"]["version"]
    )
    t4 = serving_plan("org/model", "test", {"instance_type": "Standard_NC4as_T4_v3"})
    args = command(t4["deployment"]["environment_variables"])
    assert args[args.index("--dtype") + 1] == "half"


def test_workspace_scope_rejects_cross_workspace_ids(provider):
    other = AzureMLClient(SUB, "rg", "other")
    with pytest.raises(ConfigurationError, match="different Azure ML workspace"):
        provider.client.endpoint_name(other.resource_id("test"))
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


def test_scale_logs_invoke_and_delete_preserve_selected_component(provider):
    endpoint, component = ready(provider)
    assert provider.update_deployment("test", {"instance_count": 2}).instance_count == 2
    assert endpoint.traffic == {"main": 100}
    assert component.environment_variables["HF_MODEL_ID"] == "org/model"
    provider.get_logs("test", 42, container_type="storage-initializer")
    provider.client.management.online_deployments.get_logs.assert_called_once_with(
        name="main", endpoint_name="test", lines=42, container_type="storage-initializer"
    )
    files = []

    def invoke(**kwargs):
        file = Path(kwargs["request_file"])
        files.append(file)
        assert file.stat().st_mode & 0o077 == 0
        assert json.loads(file.read_text()) == {"model": "org/model", "prompt": "Hello"}
        assert kwargs["deployment_name"] == "main"
        return '{"choices": []}'

    provider.client.management.online_endpoints.invoke.side_effect = invoke
    assert provider.invoke("test", {"model": "wrong", "prompt": "Hello"}) == {"choices": []}
    assert not files[0].exists()
    assert provider.delete_deployment("test")
    provider.client.management.online_endpoints.begin_delete.assert_called_once_with(name="test")
    provider.client.management.workspaces.begin_delete.assert_not_called()


def test_failed_or_ambiguous_endpoint_cannot_invoke(provider):
    endpoint, component = ready(provider, state="Failed")
    assert provider.get_status("test") == DeploymentStatus.FAILED
    with pytest.raises(ProviderError, match="not running"):
        provider.invoke("test", {"prompt": "Hello"})
    provider.client.management.online_endpoints.invoke.assert_not_called()
    endpoint.properties = {}
    endpoint.traffic = {"blue": 50, "green": 50}
    provider.client.management.online_deployments.list.return_value = [component, component]
    with pytest.raises(ConfigurationError, match="multiple"):
        provider.get_deployment("test")
    provider.client.deployment_name = "main"
    assert provider.get_deployment("test").status == DeploymentStatus.FAILED


def test_dry_run_no_cloud_or_state(monkeypatch):
    monkeypatch.setattr(
        "hf_cloud.providers.azureml.AzureMLProvider.deploy", Mock(side_effect=AssertionError("No cloud"))
    )
    result = CliRunner().invoke(app, ["azureml", *SCOPE, "deploy", "org/model", "--name", "test", "--dry-run"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["deployment"]["instance_type"] == DEFAULT_INSTANCE
    assert data["resourceId"].endswith("/workspaces/workspace/onlineEndpoints/test")
    assert not StateManager.DEFAULT_STATE_FILE.exists()


def test_failed_cli_deploy_never_records_or_prints_success(monkeypatch):
    monkeypatch.setattr(AzureMLProvider, "deploy", Mock(side_effect=DeploymentError("GPU allocation failed")))
    result = CliRunner().invoke(app, ["azureml", *SCOPE, "deploy", "org/model", "--name", "test"])
    assert result.exit_code == 1
    assert "GPU allocation failed" in result.stdout
    assert "successfully" not in result.stdout
    assert not StateManager.DEFAULT_STATE_FILE.exists()


def test_catalog_in_comparison_and_provider_alias(comparison):
    run, catalogs = comparison
    catalogs["azureml"] = load_instance_data("azureml")
    result = CliRunner().invoke(app, ["estimate", "org/model", "--provider", "azureml", "--json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["cheapest"]["provider"] == "azureml"
    assert data["pricing"]["catalogs"]["azureml"]["region"] == "eastus"
    result = CliRunner().invoke(
        app, ["azureml", "estimate", "org/model", "--max-model-len", "1024", "--revision", "main"]
    )
    assert result.exit_code == 0, result.stdout
    output = result.stdout.replace("\\\n  ", " ")
    assert "hf-cloud azureml deploy org/model" in output
    assert "--max-model-len 1024" in output and "--revision main" in output
    assert run.call_count == 2


def test_incomplete_endpoint_remains_visible_and_deletable(provider):
    endpoint, _ = ready(provider)
    client = provider.client.management
    client.online_deployments.get.side_effect = AzureError(404)
    client.online_endpoints.list.return_value = [endpoint]
    deployment = provider.get_deployment("test")
    assert deployment.status == DeploymentStatus.FAILED
    assert deployment.model_id == "org/model"
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
        lambda: SimpleNamespace(subscription_id=SUB, resource_group="rg", workspace_name="default"),
    )
    factory = Mock(return_value=Mock(list_deployments=Mock(return_value=[])))
    monkeypatch.setattr("hf_cloud.providers.azureml.AzureMLProvider", factory)
    result = CliRunner().invoke(app, ["ls", "--refresh", "--provider", "azureml"])
    assert result.exit_code == 0, result.stdout
    assert {call.kwargs["workspace_name"] for call in factory.call_args_list} == {"workspace", "default"}
    assert all(call.kwargs["subscription_id"] == SUB for call in factory.call_args_list)


def test_configure_saves_workspace_and_instance():
    result = CliRunner().invoke(app, ["providers", "configure", "azureml"], input=f"{SUB}\nrg\nworkspace\n\n")
    assert result.exit_code == 0, result.stdout
    from hf_cloud.core.config import Config

    assert Config().get_provider_config("azureml") == {
        "subscription_id": SUB,
        "resource_group": "rg",
        "workspace_name": "workspace",
        "default_instance_type": DEFAULT_INSTANCE,
    }


@pytest.mark.parametrize(
    "task,expected",
    [
        ("text-generation", {"prompt": "Hello", "max_tokens": 100}),
        ("chat-completion", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}),
        ("feature-extraction", {"input": "Hello"}),
    ],
)
def test_cli_invoke_uses_the_selected_task(provider, monkeypatch, task, expected):
    ready(provider, {"task": task})
    provider.invoke = Mock(return_value={"result": "ok"})
    monkeypatch.setattr("hf_cloud.cli.azureml.AzureMLProvider", Mock(return_value=provider))
    result = CliRunner().invoke(app, ["azureml", "invoke", "test", "--input", "Hello"])
    assert result.exit_code == 0, result.stdout
    provider.invoke.assert_called_once_with("test", expected)


def test_untracked_endpoint_does_not_invoke_an_unknown_model(provider):
    endpoint, component = ready(provider)
    component.properties = {}
    endpoint.properties = {}
    with pytest.raises(ConfigurationError, match="served model name"):
        provider.invoke("test", {"prompt": "Hello"})
    provider.client.management.online_endpoints.invoke.assert_not_called()
