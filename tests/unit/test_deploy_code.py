"""Generated deployment code is standalone, executable and never deployed while printing."""

import ast
import sys
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.config import Config
from hf_cloud.providers.azure_catalog import PublishedModel
from hf_cloud.providers.azureml.serving import serving_plan
from hf_cloud.providers.foundry import FoundryProvider

SUB = "00000000-0000-0000-0000-000000000000"
MODEL = "azureml://registries/azure-huggingface/models/org--model/versions/3"
TEMPLATE = "azureml://registries/azure-huggingface/deploymenttemplates/model-h100/versions/7"
runner = CliRunner()


@pytest.fixture
def generation(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "DEFAULT_CONFIG_FILE", tmp_path / "config.json")
    for provider in ("sagemaker", "vertex", "azureml", "foundry"):
        monkeypatch.setattr(
            f"hf_cloud.cli.{provider}.SdkUpdateCheck", Mock(side_effect=AssertionError("SDK update started"))
        )
        monkeypatch.setattr(
            f"hf_cloud.cli.{provider}.StateManager", Mock(side_effect=AssertionError("State changed"))
        )
    monkeypatch.setattr("hf_cloud.cli.sagemaker.get_model_task", lambda *_: "question-answering")
    monkeypatch.setattr("hf_cloud.cli.vertex.get_model_task", lambda *_: "text-generation")
    aws = Mock(side_effect=AssertionError("AWS provider initialized"))
    vertex = Mock(side_effect=AssertionError("Vertex provider initialized"))
    monkeypatch.setattr("hf_cloud.cli.sagemaker.SageMakerProvider", aws)
    monkeypatch.setattr("hf_cloud.cli.vertex.ProviderRegistry.get_provider", vertex)
    azure = Mock()
    azure.client = NS(
        deployment_name=None,
        subscription_id=SUB,
        resource_group="rg",
        workspace_name="workspace",
        resource_id=lambda _: "/endpoint",
    )
    azure.prepare_deployment.side_effect = lambda model, name, config: serving_plan(
        PublishedModel(model, "HuggingFace", "org-model", "3"),
        name,
        config,
    )
    foundry = Mock()
    foundry.client = NS(
        subscription_id=SUB,
        scope=lambda name: {"resource_group_name": "rg", "account_name": "account", "deployment_name": name},
    )
    foundry.prepare_config.side_effect = lambda _, config: {
        **config,
        "model_asset_id": MODEL,
        "deployment_template": TEMPLATE,
        "accelerator_type": "H100_80GB",
    }
    foundry.deployment_resource.side_effect = FoundryProvider.deployment_resource
    monkeypatch.setattr("hf_cloud.cli.azureml.AzureMLProvider", lambda *_: azure)
    monkeypatch.setattr("hf_cloud.cli.foundry.FoundryProvider", lambda *_: foundry)
    return azure, foundry


def generate(provider, *args):
    result = runner.invoke(
        app, [provider, "deploy", "org/model", "--name", "my-endpoint", "--print-code", *args]
    )
    assert result.exit_code == 0, result.stdout
    ast.parse(result.stdout)
    assert "from hf_cloud" not in result.stdout
    return result.stdout


def module(monkeypatch, name, **members):
    value = ModuleType(name)
    value.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, value)


def test_sagemaker_code_preserves_role_schema_and_token_reference(generation, monkeypatch):
    source = generate(
        "sagemaker",
        "--region",
        "us-east-2",
        "--role-name",
        "my-role",
        "--instance-count",
        "2",
        "--token",
        "secret-value",
    )
    assert "secret-value" not in source
    monkeypatch.setenv("HF_TOKEN", "runtime-secret")
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    boto = Mock()
    iam, sm = Mock(), Mock()
    boto.client.side_effect = lambda service: iam if service == "iam" else sm
    iam.get_role.return_value = {"Role": {"Arn": "arn:aws:iam::123456789012:role/my-role"}}
    sm.describe_endpoint.return_value = {"EndpointStatus": "InService", "EndpointArn": "endpoint-arn"}
    monkeypatch.setattr("boto3.Session", Mock(return_value=boto))
    builder, schema = Mock(), Mock()
    factory, schema_factory = Mock(return_value=builder), Mock(return_value=schema)
    module(monkeypatch, "sagemaker.core.helper.session_helper", Session=Mock(), get_execution_role=Mock())
    module(monkeypatch, "sagemaker.serve.model_builder", ModelBuilder=factory, SchemaBuilder=schema_factory)
    exec(compile(source, "snippet.py", "exec"), {})
    iam.get_role.assert_called_once_with(RoleName="my-role")
    assert "question" in schema_factory.call_args.kwargs["sample_input"]
    assert factory.call_args.kwargs["env_vars"]["HF_TOKEN"] == "runtime-secret"
    builder.deploy.assert_called_once_with(endpoint_name="my-endpoint", initial_instance_count=2, wait=True)
    sm.describe_endpoint.return_value = {"EndpointStatus": "Failed", "FailureReason": "capacity"}
    with pytest.raises(RuntimeError, match="capacity"):
        exec(compile(source, "snippet.py", "exec"), {})


def test_vertex_code_uses_gpu_container_and_selected_scaling(generation, monkeypatch):
    source = generate(
        "vertex",
        "--machine-type",
        "g2-standard-4",
        "--accelerator-type",
        "NVIDIA_L4",
        "--accelerator-count",
        "1",
        "--min-replicas",
        "2",
        "--max-replicas",
        "3",
        "--project",
        "project",
        "--token",
        "secret-value",
    )
    assert "secret-value" not in source
    api = Mock()
    module(
        monkeypatch,
        "google.cloud.aiplatform",
        **{name: getattr(api, name) for name in ("init", "Model", "Endpoint")},
    )
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    exec(compile(source, "snippet.py", "exec"), {})
    kwargs = api.Model.upload.call_args.kwargs
    assert "inference-cu121" in kwargs["serving_container_image_uri"]
    request = api.Model.upload.return_value.deploy.call_args.kwargs
    assert request["accelerator_type"] == "NVIDIA_L4" and request["accelerator_count"] == 1
    assert request["min_replica_count"] == 2 and request["max_replica_count"] == 3


def test_direct_vertex_deployment_also_selects_gpu_image(monkeypatch):
    from hf_cloud.providers.vertex.deployer import VertexDeployer

    api = Mock()
    api.Model.upload.return_value.resource_name = "projects/p/locations/us-central1/models/m"
    api.Endpoint.create.return_value.resource_name = "projects/p/locations/us-central1/endpoints/e"
    module(
        monkeypatch,
        "google.cloud.aiplatform",
        **{name: getattr(api, name) for name in ("init", "Model", "Endpoint")},
    )
    monkeypatch.setattr("hf_cloud.providers.vertex.deployer.get_model_task", lambda **_: "text-generation")
    VertexDeployer(Mock()).deploy(
        "org/model",
        "my-endpoint",
        {
            "project": "p",
            "machine_type": "g2-standard-4",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
            "location": "us-central1",
        },
    )
    assert "inference-cu121" in api.Model.upload.call_args.kwargs["serving_container_image_uri"]


def test_azureml_code_uses_pinned_plan_and_preserves_secret_connection(generation, monkeypatch):
    azure_provider, _ = generation
    source = generate("azureml", "--instance-count", "2", "--hf-token-connection", "hub-token")
    assert "HuggingFace/models/org-model/versions/3" in source
    assert "connections/hub-token/credentials/HF_TOKEN" in source
    import azure.ai.ml
    from azure.core.exceptions import ResourceNotFoundError

    client = Mock()
    endpoint = NS(name="my-endpoint", scoring_uri="https://example.com", traffic={})
    client.online_endpoints.get.side_effect = [ResourceNotFoundError(), endpoint]
    client.online_deployments.get.return_value = NS(provisioning_state="Succeeded")
    monkeypatch.setattr(azure.ai.ml, "MLClient", Mock(return_value=client))
    exec(compile(source, "snippet.py", "exec"), {})
    deployment = client.online_deployments.begin_create_or_update.call_args.args[0]
    assert deployment.model.endswith("/versions/3") and deployment.instance_count == 2
    assert endpoint.traffic == {"main": 100}
    azure_provider.deploy.assert_not_called()
    client.reset_mock()
    client.online_endpoints.get.side_effect = None
    client.online_endpoints.get.return_value = endpoint
    with pytest.raises(RuntimeError, match="already exists"):
        exec(compile(source, "snippet.py", "exec"), {})
    client.online_endpoints.begin_create_or_update.assert_not_called()


def test_foundry_code_uses_resolved_template_and_cannot_overwrite(generation, monkeypatch):
    _, foundry = generation
    source = generate("foundry", "--instance-count", "2")
    from azure.core.exceptions import ResourceNotFoundError

    client = Mock()
    operations = client.managed_compute_deployments
    operations.get.side_effect = [
        ResourceNotFoundError(),
        NS(properties=NS(provisioning_state="Succeeded"), id="deployment-id"),
    ]
    module(
        monkeypatch, "azure.mgmt.cognitiveservices", CognitiveServicesManagementClient=Mock(return_value=client)
    )
    exec(compile(source, "snippet.py", "exec"), {})
    body = operations.begin_create_or_update.call_args.kwargs["resource"]
    assert body["properties"]["model"] == MODEL
    assert body["properties"]["deploymentTemplate"] == TEMPLATE
    assert body["sku"]["capacity"] == 2
    foundry.deploy.assert_not_called()
    operations.reset_mock()
    operations.get.side_effect = None
    with pytest.raises(RuntimeError, match="already exists"):
        exec(compile(source, "snippet.py", "exec"), {})
    operations.begin_create_or_update.assert_not_called()


@pytest.mark.parametrize(
    "provider,flags",
    [("azureml", ["--dry-run"]), ("azureml", ["--export-dir", "/unused"]), ("foundry", ["--dry-run"])],
)
def test_code_output_conflicts_with_other_output_modes_before_discovery(generation, provider, flags):
    result = runner.invoke(app, [provider, "deploy", "org/model", "--name", "test", "--print-code", *flags])
    assert result.exit_code == 2
    generation[0].prepare_deployment.assert_not_called()
    generation[1].prepare_config.assert_not_called()
