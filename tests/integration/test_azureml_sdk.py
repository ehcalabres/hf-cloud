"""Real SDK serialization for published catalog models; never contact Azure."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

pytest.importorskip("azure.ai.ml", reason="Install azure-ai-ml for SDK compatibility checks")
from azure.ai.ml import load_online_deployment, load_online_endpoint
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint

from hf_cloud.cli.azureml import _export_plan
from hf_cloud.core.deployment import DeploymentStatus
from hf_cloud.core.exceptions import ConfigurationError, DeploymentError, ProviderError
from hf_cloud.providers.azureml import AzureMLProvider
from hf_cloud.providers.azureml.serving import DEFAULT_INSTANCE, serving_plan
from tests.unit.test_azureml import PUBLISHED, SUB, AzureError, ready


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr("hf_cloud.core.config.Config.DEFAULT_CONFIG_FILE", tmp_path / "config.json")
    provider = AzureMLProvider(SUB, "rg", "workspace")
    provider.catalog = Mock(resolve=Mock(return_value=PUBLISHED))
    provider.client._workspace = NS(location="eastus")
    provider.client._management = Mock()
    endpoint, component = ready(provider)
    provider.client.management.online_endpoints.get.side_effect = [AzureError(404), endpoint, endpoint]
    provider.client.management.compute.list_sizes.return_value = [NS(name=DEFAULT_INSTANCE, gpus=1)]
    return provider


def test_catalog_deployment_serializes_without_custom_image_or_code(provider):
    result = provider.deploy("org/model", "test", {"hf_token_connection": "hub-token"})
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
    wire = deployment._to_rest_object(location="eastus").as_dict()
    assert wire["properties"]["model"] == PUBLISHED.asset_id
    assert wire["properties"]["endpointComputeType"] == "Managed"
    assert wire["properties"]["instanceType"] == DEFAULT_INSTANCE
    assert wire["sku"]["capacity"] == 1
    assert wire["properties"]["environmentVariables"]["HF_TOKEN"].startswith("${{azureml://connections/")
    assert "environmentId" not in wire["properties"]
    assert "codeConfiguration" not in wire["properties"]
    client.environments.create_or_update.assert_not_called()
    client.models.create_or_update.assert_not_called()
    assert client.online_endpoints.begin_create_or_update.call_args.args[0].traffic == {"main": 100}


def test_exported_catalog_definitions_load_with_real_sdk(tmp_path):
    plan = serving_plan(PUBLISHED, "test", {"hf_token_connection": "hub-token"})
    _export_plan(tmp_path, plan)
    endpoint = load_online_endpoint(tmp_path / "endpoint.json")
    component = load_online_deployment(tmp_path / "deployment.json")
    assert endpoint.name == "test" and endpoint.auth_mode == "aad_token"
    assert component.model == PUBLISHED.asset_id
    assert component.environment is None and component.code_configuration is None
    assert component.environment_variables["HF_TOKEN"].startswith("${{azureml://connections/")


@pytest.mark.parametrize("status", [403, 409])
def test_lookup_errors_never_create_any_resources(provider, status):
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
    endpoint, _ = ready(provider, state="Failed")
    provider.client.management.online_endpoints.get.side_effect = [AzureError(404), endpoint, endpoint]
    with pytest.raises(DeploymentError, match="did not become ready"):
        provider.deploy("org/model", "test", {})


def test_regional_vm_size_is_checked_before_provisioning(provider):
    client = provider.client.management
    client.compute.list_sizes.return_value = []
    with pytest.raises(ConfigurationError, match="VM size"):
        provider.deploy("org/model", "test", {})
    client.online_endpoints.begin_create_or_update.assert_not_called()


def test_real_model_metadata_exposes_publisher_template_references(monkeypatch):
    from azure.ai.ml._restclient.arm_ml_service.models import ModelVersion
    from azure.ai.ml.entities import AcceleratorMap, DeploymentTemplate, Model

    from hf_cloud.providers.azure_catalog import AzureCatalog

    model_id = "azureml://registries/azure-huggingface/models/org--model/versions/3"
    template_id = "azureml://registries/azure-huggingface/deploymenttemplates/org-model-h100/versions/5"
    wire = ModelVersion(
        {
            "id": model_id,
            "name": "3",
            "properties": {
                "modelType": "custom_model",
                "flavors": {},
                "tags": {"task": "text-generation"},
                "defaultDeploymentTemplate": {"assetId": template_id},
                "allowedDeploymentTemplates": [{"assetId": template_id}],
            },
        }
    )
    entity = Model._from_rest_object(wire)
    catalog = AzureCatalog()
    registry = Mock()
    catalog._clients["azure-huggingface"] = registry
    registry.models.get.return_value = entity
    registry.deployment_templates.get.return_value = DeploymentTemplate(
        name="org-model-h100",
        version="5",
        accelerator_maps=[
            AcceleratorMap(
                accelerator_type="H100_80GB", number_of_accelerators_per_model_instance=1, default=True
            )
        ],
    )
    monkeypatch.setattr(
        "hf_cloud.providers.azure_catalog.lookup_foundry_model",
        lambda _: ("azure-huggingface", "org--model", "3"),
    )
    published = catalog.resolve("org/model", "foundry")
    assert published.asset_id == model_id and published.default_template == template_id
    assert catalog.foundry_template(published)["deployment_template"] == template_id


def test_sdk_latest_label_absence_is_not_an_authentication_failure():
    from azure.ai.ml.exceptions import ErrorTarget, ValidationErrorType, ValidationException

    from hf_cloud.providers.azure_catalog import AzureCatalog, CatalogModelUnavailable

    catalog = AzureCatalog()
    registry = Mock()
    catalog._clients["HuggingFace"] = registry
    registry.models.get.side_effect = ValidationException(
        message="Asset is not published",
        no_personal_data_message="Asset is not published",
        target=ErrorTarget.ASSET,
        error_type=ValidationErrorType.RESOURCE_NOT_FOUND,
    )
    with pytest.raises(CatalogModelUnavailable):
        catalog.resolve("org/model", "azureml")
