"""Validate real Azure ML SDK entities and lifecycle without contacting Azure."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("azure.ai.ml", reason="Install azure-ai-ml to check SDK compatibility")
from azure.ai.ml.entities import Environment, ManagedOnlineDeployment, ManagedOnlineEndpoint

from hf_cloud.core.deployment import DeploymentStatus
from hf_cloud.core.exceptions import ConfigurationError, DeploymentError, ProviderError
from hf_cloud.providers.azureml import AzureMLProvider
from hf_cloud.providers.azureml.serving import DEFAULT_INSTANCE, serving_plan
from tests.unit.test_azureml import SUB, AzureError, ready


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr("hf_cloud.core.config.Config.DEFAULT_CONFIG_FILE", tmp_path / "config.json")
    provider = AzureMLProvider(SUB, "rg", "workspace")
    provider.client._workspace = SimpleNamespace(location="eastus")
    provider.client._management = Mock()
    client = provider.client.management
    endpoint, component = ready(provider)
    client.online_endpoints.get.side_effect = [AzureError(404), endpoint, endpoint]
    client.compute.list_sizes.return_value = [SimpleNamespace(name=DEFAULT_INSTANCE, gpus=1)]
    client.environments.get.side_effect = AzureError(404)

    def register(environment):
        assert isinstance(environment, Environment)
        directory = Path(environment.build.path)
        assert (directory / "start.py").is_file()
        assert "vllm/vllm-openai:v0.32.0" in (directory / "Dockerfile").read_text()
        environment.validate()
        wire = environment._to_rest_object().as_dict()
        assert wire["properties"]["inferenceConfig"]["scoringRoute"] == {
            "path": "/v1/completions",
            "port": 8000,
        }
        return environment

    client.environments.create_or_update.side_effect = register
    return provider


def test_deploy_sdk_entities_and_serialized_managed_contract(provider):
    result = provider.deploy("org/model", "test", {"hf_token_connection": "hub-token", "max_model_len": 2048})
    assert result.status == DeploymentStatus.RUNNING
    client = provider.client.management
    client.connections.get.assert_called_once_with("hub-token", populate_secrets=False)
    endpoint = client.online_endpoints.begin_create_or_update.call_args_list[0].args[0]
    assert isinstance(endpoint, ManagedOnlineEndpoint)
    wire = endpoint._to_rest_online_endpoint(location="eastus").as_dict()
    assert wire["properties"]["authMode"] == "AADToken"
    assert wire["identity"]["type"] == "SystemAssigned"
    deployment = client.online_deployments.begin_create_or_update.call_args.args[0]
    assert isinstance(deployment, ManagedOnlineDeployment)
    # The SDK resolves azureml:name:version into an ARM environment ID before serializing.
    deployment.environment = provider.client.workspace_id + "/environments/hf-cloud-vllm/versions/1"
    wire = deployment._to_rest_object(location="eastus").as_dict()
    assert wire["properties"]["endpointComputeType"] == "Managed"
    assert wire["properties"]["instanceType"] == DEFAULT_INSTANCE
    assert wire["sku"]["capacity"] == 1
    assert wire["properties"]["environmentVariables"]["HF_TOKEN"].startswith("${{azureml://connections/")
    args = json.loads(wire["properties"]["environmentVariables"]["HF_CLOUD_VLLM_ARGS"])
    assert args[args.index("--max-model-len") + 1] == "2048"
    assert wire["properties"]["livenessProbe"]["initialDelay"] == "PT10M"
    assert client.online_endpoints.begin_create_or_update.call_args.args[0].traffic == {"main": 100}


@pytest.mark.parametrize("status", [403, 409])
def test_lookup_errors_do_not_create_or_build_anything(provider, status):
    client = provider.client.management
    client.online_endpoints.get.side_effect = AzureError(status)
    with pytest.raises(ProviderError, match=str(status)):
        provider.deploy("org/model", "test", {})
    client.environments.create_or_update.assert_not_called()
    client.online_endpoints.begin_create_or_update.assert_not_called()


def test_existing_endpoint_is_not_overwritten(provider):
    client = provider.client.management
    client.online_endpoints.get.side_effect = None
    with pytest.raises(DeploymentError, match="already exists"):
        provider.deploy("org/model", "test", {})
    client.online_endpoints.begin_create_or_update.assert_not_called()


def test_sdk_poller_success_does_not_override_failed_component(provider):
    client = provider.client.management
    endpoint, component = ready(provider, state="Failed")
    client.online_endpoints.get.side_effect = [AzureError(404), endpoint, endpoint]
    with pytest.raises(DeploymentError, match="did not become ready"):
        provider.deploy("org/model", "test", {})


def test_regional_gpu_layout_is_checked_before_building(provider):
    client = provider.client.management
    client.compute.list_sizes.return_value = [SimpleNamespace(name=DEFAULT_INSTANCE, gpus=0)]
    with pytest.raises(ConfigurationError, match="GPU layout"):
        provider.deploy("org/model", "test", {})
    client.environments.create_or_update.assert_not_called()


def test_reuses_versioned_environment_without_uploading_context(provider):
    client = provider.client.management
    client.environments.get.side_effect = None
    client.environments.get.return_value = SimpleNamespace(name="hf-cloud-vllm", version="cached")
    provider.deploy("org/model", "test", {})
    client.environments.create_or_update.assert_not_called()
