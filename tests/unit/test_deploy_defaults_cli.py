"""All provider deploy modes share automatic hardware defaults and explicit overrides."""

import ast
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.config import Config
from hf_cloud.core.deployment import Deployment, DeploymentStatus
from hf_cloud.providers.azure_catalog import PublishedModel
from hf_cloud.providers.azureml.serving import serving_plan
from hf_cloud.providers.foundry import FoundryProvider
from hf_cloud.utils.hf_mem import HfMemError

MODEL = "azureml://registries/azure-huggingface/models/org--model/versions/7"
TEMPLATE = "azureml://registries/azure-huggingface/deploymenttemplates/model-a100/versions/3"
PROVIDERS = ["sagemaker", "vertex", "azureml", "foundry"]
HARDWARE = {
    "sagemaker": "ml.g6.12xlarge",
    "vertex": "g2-standard-48",
    "azureml": "Standard_NC48ads_A100_v4",
    "foundry": "model-a100",
}


@pytest.fixture
def flow(provider, monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "DEFAULT_CONFIG_FILE", tmp_path / "config.json")
    Config().set_provider_config(
        provider,
        {
            "default_instance_type": "legacy-instance",
            "default_machine_type": "legacy-machine",
        },
    )
    instance = {
        "instanceType": HARDWARE[provider],
        "pricePerHourUsd": 2,
        "numGpu": 4,
        "acceleratorType": "NVIDIA_L4" if provider == "vertex" else "A100_80GB",
        "modelAssetId": MODEL,
        "deploymentTemplateId": TEMPLATE,
    }
    choose = Mock(return_value=NS(instance=instance, estimate=NS(warnings=())))
    monkeypatch.setattr(f"hf_cloud.cli.{provider}.choose_instance", choose)
    client = NS(
        region="us-east-1",
        deployment_name=None,
        subscription_id="subscription",
        resource_group="rg",
        workspace_name="workspace",
        resource_id=lambda _: "/endpoint",
        scope=lambda _: {
            "resource_group_name": "rg",
            "account_name": "account",
            "deployment_name": "test-endpoint",
        },
    )
    cloud = Mock()
    cloud.client = client
    cloud.deploy.return_value = Deployment(
        deployment_id="test-endpoint",
        deployment_name="test-endpoint",
        provider=provider,
        model_id="org/model",
        status=DeploymentStatus.RUNNING,
        config={},
        instance_type=instance["instanceType"],
        instance_count=1,
        region="us-east-1",
    )
    cloud.prepare_deployment.side_effect = lambda model, name, config: serving_plan(
        PublishedModel(model, "HuggingFace", "org-model", "7"),
        name,
        config,
    )
    cloud.prepare_config.side_effect = lambda _, config: config
    cloud.deployment_resource.side_effect = FoundryProvider.deployment_resource
    factory = Mock(return_value=cloud)
    if provider == "vertex":
        monkeypatch.setattr("hf_cloud.cli.vertex.ProviderRegistry.get_provider", factory)
    else:
        cls = {"sagemaker": "SageMakerProvider", "azureml": "AzureMLProvider", "foundry": "FoundryProvider"}[
            provider
        ]
        monkeypatch.setattr(f"hf_cloud.cli.{provider}.{cls}", factory)
    if provider in ("sagemaker", "vertex"):
        monkeypatch.setattr(f"hf_cloud.cli.{provider}.get_model_task", lambda *_: "text-generation")
    monkeypatch.setattr("hf_cloud.cli.sagemaker.SageMakerClient.resolve_region", lambda _: "us-east-1")
    state, check = Mock(), Mock()
    monkeypatch.setattr(f"hf_cloud.cli.{provider}.StateManager", Mock(return_value=state))
    monkeypatch.setattr(f"hf_cloud.cli.{provider}.SdkUpdateCheck", check)
    return choose, cloud, state, check


def run(provider, flags):
    return CliRunner().invoke(app, [provider, "deploy", "org/model", "--name", "test-endpoint", *flags])


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("print_code", [False, True])
def test_deploy_and_code_select_same_hardware_and_ignore_old_defaults(provider, flow, print_code):
    choose, cloud, state, check = flow
    result = run(provider, ["--print-code"] if print_code else [])
    assert result.exit_code == 0, result.output
    choose.assert_called_once()
    assert choose.call_args.args == ("org/model", provider)
    assert "Auto-selected" in result.stderr
    if print_code:
        ast.parse(result.stdout)
        assert HARDWARE[provider] in result.stdout
        cloud.deploy.assert_not_called()
        state.add_deployment.assert_not_called()
        check.assert_not_called()
    else:
        cloud.deploy.assert_called_once()
        state.add_deployment.assert_called_once()
        if provider in ("sagemaker", "vertex"):
            config = cloud.deploy.call_args.kwargs["config"]
            assert config.get("instance_type", config.get("machine_type")) == HARDWARE[provider]
            if provider == "vertex":
                assert config["accelerator_type"] == "NVIDIA_L4"
                assert config["accelerator_count"] == 4
        elif provider == "azureml":
            config = cloud.prepare_deployment.call_args.args[2]
            assert config["instance_type"] == HARDWARE[provider]
            assert config["catalog_version"] == "7"
        else:
            config = cloud.prepare_config.call_args.args[1]
            assert config["deployment_template"] == TEMPLATE
            assert config["model_asset_id"] == MODEL


@pytest.mark.parametrize("provider", PROVIDERS)
def test_explicit_hardware_bypasses_estimation(provider, flow):
    choose, _, _, _ = flow
    flags = {
        "sagemaker": ["--instance-type", "ml.g6.2xlarge"],
        "vertex": ["--machine-type", "g2-standard-4"],
        "azureml": ["--instance-type", "Standard_NC24ads_A100_v4"],
        "foundry": ["--template", TEMPLATE, "--model-asset-id", MODEL, "--accelerator-type", "A100_80GB"],
    }[provider]
    result = run(provider, [*flags, "--print-code"])
    assert result.exit_code == 0, result.output
    ast.parse(result.stdout)
    choose.assert_not_called()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_estimation_failure_keeps_stdout_clean_and_never_deploys(provider, flow):
    choose, cloud, state, check = flow
    choose.side_effect = HfMemError("Cannot estimate model; check Hub access")
    result = run(provider, ["--print-code"])
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert "Cannot estimate" in result.stderr
    cloud.deploy.assert_not_called()
    state.add_deployment.assert_not_called()
    check.assert_not_called()
