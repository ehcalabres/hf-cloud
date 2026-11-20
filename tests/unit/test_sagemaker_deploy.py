"""Regression coverage for SDK returns that hide a failed SageMaker endpoint."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.cli import sagemaker as cli
from hf_cloud.core.deployment import DeploymentStatus
from hf_cloud.core.exceptions import DeploymentError
from hf_cloud.providers.sagemaker import SageMakerProvider, deployer


@pytest.fixture
def deployment_flow(monkeypatch):
    builder = Mock()
    # The affected SDK logged a failure but returned an endpoint object anyway.
    builder.deploy.return_value = Mock()
    factory = Mock(return_value=builder)
    module = ModuleType("sagemaker.serve")
    module.ModelServer = SimpleNamespace(VLLM="vllm-server")
    module.ModelBuilder = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    provider = SageMakerProvider(region="us-east-1")
    provider.client.get_sagemaker_session = Mock()
    provider.client.get_execution_role = Mock(return_value="arn:aws:iam::123456789012:role/ChosenRole")
    provider.client._sagemaker_client = Mock()
    provider.client.sagemaker.describe_endpoint.return_value = {"EndpointStatus": "InService"}
    monkeypatch.setattr(deployer, "get_model_task", Mock(return_value="text-generation"))
    monkeypatch.setattr(deployer, "get_schema_builder_from_model", Mock())
    monkeypatch.setattr(cli, "SageMakerProvider", Mock(return_value=provider))
    monkeypatch.setattr(cli, "_get_provider_defaults", lambda: {"role": "configured-role"})
    monkeypatch.setattr(cli, "SdkUpdateCheck", Mock())
    monkeypatch.setattr(
        cli,
        "choose_instance",
        Mock(
            return_value=SimpleNamespace(
                instance={"instanceType": "ml.g6.2xlarge", "pricePerHourUsd": 1.222},
                estimate=SimpleNamespace(warnings=()),
            )
        ),
    )
    state = Mock()
    monkeypatch.setattr(cli, "StateManager", Mock(return_value=state))
    return provider, builder, factory, state


def deploy_command(*flags):
    return CliRunner().invoke(app, ["sagemaker", "deploy", "org/model", "--name", "test-endpoint", *flags])


def assert_no_deployment_success(output):
    assert "Model deployed successfully" not in output
    assert "Deployment completed successfully" not in output
    assert "Deployment initiated successfully" not in output


@pytest.mark.parametrize("status", ["Failed", "Creating", "Updating", "Deleting", "OutOfService", None])
def test_sdk_return_without_inservice_never_reports_success(deployment_flow, status):
    provider, builder, _, state = deployment_flow
    response = {"EndpointStatus": status} if status else {}
    response["FailureReason"] = "Container failed its startup health check"
    provider.client.sagemaker.describe_endpoint.return_value = response
    result = deploy_command()
    assert result.exit_code == 1, result.output
    assert "Container failed its startup health check" in result.stdout
    assert (status or "Unknown") in result.stdout
    assert_no_deployment_success(result.stdout)
    state.add_deployment.assert_not_called()
    builder.deploy.assert_called_once_with(
        endpoint_name="test-endpoint",
        initial_instance_count=1,
        instance_type="ml.g6.2xlarge",
        container_timeout_in_seconds=900,
        wait=True,
    )
    provider.client.sagemaker.describe_endpoint.assert_called_once_with(EndpointName="test-endpoint")


def test_missing_failure_reason_is_explicit(deployment_flow):
    provider, _, _, _ = deployment_flow
    provider.client.sagemaker.describe_endpoint.return_value = {"EndpointStatus": "Failed"}
    result = deploy_command()
    assert result.exit_code == 1
    assert "No failure reason returned by SageMaker" in " ".join(result.stdout.split())
    assert_no_deployment_success(result.stdout)


def test_status_verification_failure_is_not_success(deployment_flow):
    provider, _, _, state = deployment_flow
    provider.client.sagemaker.describe_endpoint.side_effect = RuntimeError("DescribeEndpoint access denied")
    result = deploy_command()
    assert result.exit_code == 1
    assert "Unable to verify SageMaker endpoint status" in result.stdout
    assert "DescribeEndpoint access denied" in " ".join(result.stdout.split())
    assert_no_deployment_success(result.stdout)
    state.add_deployment.assert_not_called()


def test_inservice_is_saved_and_reported_running(deployment_flow):
    _, builder, _, state = deployment_flow
    result = deploy_command()
    assert result.exit_code == 0, result.output
    assert "Deployment completed successfully!" in result.stdout
    assert "Status: running" in result.stdout
    assert "5-10 minutes" not in result.stdout
    assert state.add_deployment.call_args.args[0].status == DeploymentStatus.RUNNING
    builder.deploy.assert_called_once_with(
        endpoint_name="test-endpoint",
        initial_instance_count=1,
        instance_type="ml.g6.2xlarge",
        container_timeout_in_seconds=900,
        wait=True,
    )


def test_explicit_arn_overrides_configured_role(deployment_flow):
    provider, _, factory, _ = deployment_flow
    arn = "arn:aws:iam::123456789012:role/ChosenRole"
    result = deploy_command("--role", arn)
    assert result.exit_code == 0, result.output
    provider.client.get_execution_role.assert_called_once_with(role_name=arn)
    assert factory.call_args.kwargs["role_arn"] == arn


def test_explicit_role_name_overrides_configured_role(deployment_flow):
    provider, _, _, _ = deployment_flow
    assert deploy_command("--role-name", "ChosenRole").exit_code == 0
    provider.client.get_execution_role.assert_called_once_with(role_name="ChosenRole")


@pytest.mark.parametrize(
    "flags",
    [
        ["--role", "configured-role", "--role-name", "ChosenRole"],
        ["--role-name", "arn:aws:iam::123456789012:role/ChosenRole"],
        ["--role-name", "path/ChosenRole"],
    ],
)
def test_ambiguous_or_invalid_role_name_options_fail_before_provider_work(deployment_flow, flags):
    provider, builder, _, state = deployment_flow
    result = deploy_command(*flags)
    assert result.exit_code == 2, result.output
    provider.client.get_execution_role.assert_not_called()
    builder.build.assert_not_called()
    state.add_deployment.assert_not_called()


def test_invalid_role_trust_stops_before_session_model_build_and_deploy(deployment_flow):
    provider, builder, factory, state = deployment_flow
    # Use the real resolver and trust check rather than the fixture's role mock.
    del provider.client.get_execution_role
    provider.client._iam_client = Mock()
    provider.client.iam.get_role.return_value = {
        "Role": {
            "Arn": "arn:aws:iam::123456789012:role/configured-role",
            "AssumeRolePolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "sts:AssumeRoleWithSAML",
                        "Principal": {"Federated": "arn:aws:iam::123456789012:saml-provider/SSO"},
                    }
                ]
            },
        }
    }
    result = deploy_command()
    assert result.exit_code == 1
    assert "does not allow SageMaker to assume it" in result.stdout
    provider.client.get_sagemaker_session.assert_not_called()
    deployer.get_model_task.assert_not_called()
    factory.assert_not_called()
    builder.build.assert_not_called()
    builder.deploy.assert_not_called()
    state.add_deployment.assert_not_called()


def test_configured_role_is_used_without_override(deployment_flow):
    provider, _, _, _ = deployment_flow
    assert deploy_command().exit_code == 0
    provider.client.get_execution_role.assert_called_once_with(role_name="configured-role")


def test_sdk_exception_fails_without_success_state(deployment_flow):
    provider, builder, _, state = deployment_flow
    builder.deploy.side_effect = RuntimeError("SDK deployment error")
    result = deploy_command()
    assert result.exit_code == 1
    assert "SDK deployment error" in result.stdout
    assert_no_deployment_success(result.stdout)
    state.add_deployment.assert_not_called()
    provider.client.sagemaker.describe_endpoint.assert_not_called()


@pytest.mark.parametrize("status", [DeploymentStatus.FAILED, DeploymentStatus.CREATING])
def test_cli_rejects_a_nonrunning_deployment_returned_by_provider(deployment_flow, status):
    provider, _, _, state = deployment_flow
    provider.deploy = Mock(return_value=Mock(deployment_id="test-endpoint", status=status))
    result = deploy_command()
    assert result.exit_code == 1
    assert f"status '{status.value}'" in result.stdout
    assert_no_deployment_success(result.stdout)
    state.add_deployment.assert_not_called()


def test_provider_preserves_failure_details(deployment_flow):
    provider, _, _, _ = deployment_flow
    provider.client.sagemaker.describe_endpoint.return_value = {
        "EndpointStatus": "Failed",
        "FailureReason": "Insufficient capacity",
    }
    with pytest.raises(DeploymentError, match="Insufficient capacity") as error:
        provider.deploy("org/model", "test-endpoint", {"instance_type": "ml.g6.2xlarge"})
    assert error.value.deployment_id == "test-endpoint"


@pytest.mark.parametrize("parallel_size", [None, 4])
def test_vllm_recipe_uses_instance_gpus_per_replica(deployment_flow, parallel_size):
    provider, builder, factory, _ = deployment_flow
    deployment = provider.deploy(
        "org/model",
        "test-endpoint",
        {
            "instance_type": "ml.p5.48xlarge",
            "instance_count": 2,
            "tensor_parallel_size": parallel_size,
        },
    )
    args = factory.call_args.kwargs
    assert args["model_server"] == "vllm-server"
    assert args["model_metadata"] == {"HF_TASK": "text-generation"}
    assert args["env_vars"] == {
        "SM_VLLM_MODEL": "org/model",
        "HF_TASK": "text-generation",
        "SM_VLLM_TENSOR_PARALLEL_SIZE": str(parallel_size or 8),
    }
    builder.deploy.assert_called_once_with(
        endpoint_name="test-endpoint",
        initial_instance_count=2,
        instance_type="ml.p5.48xlarge",
        container_timeout_in_seconds=3600,
        wait=True,
    )
    assert deployment.config["model_server"] == "vllm"
    assert deployment.config["tensor_parallel_size"] == (parallel_size or 8)


def test_non_generation_keeps_task_specific_serving(deployment_flow):
    provider, _, factory, _ = deployment_flow
    deployer.get_model_task.return_value = "question-answering"
    provider.deploy("org/model", "test-endpoint", {"instance_type": "ml.g6.2xlarge"})
    args = factory.call_args.kwargs
    assert "model_server" not in args
    assert args["env_vars"] == {"HF_MODEL_ID": "org/model", "HF_TASK": "question-answering"}


@pytest.mark.parametrize("size", [0, 2])
def test_invalid_parallel_size_fails_before_provider_work(deployment_flow, size):
    provider, builder, _, _ = deployment_flow
    result = deploy_command("--instance-type", "ml.g6.2xlarge", "--tensor-parallel-size", str(size))
    assert result.exit_code == 2, result.stdout
    provider.client.get_execution_role.assert_not_called()
    builder.build.assert_not_called()


@pytest.mark.parametrize(
    "saved_server,api,expected",
    [
        ("vllm", "auto", {"model": "org/model", "prompt": "Hello", "max_tokens": 7}),
        (None, "vllm", {"prompt": "Hello", "max_tokens": 7}),
        ("vllm", "huggingface", {"inputs": "Hello", "parameters": {"max_new_tokens": 7}}),
        (None, "auto", {"inputs": "Hello", "parameters": {"max_new_tokens": 7}}),
    ],
)
def test_invoke_matches_saved_serving_protocol(deployment_flow, saved_server, api, expected):
    provider, _, _, state = deployment_flow
    provider.get_deployment = Mock(return_value=Mock(to_dict=lambda: {}))
    provider.invoke = Mock(return_value={"choices": [{"text": "Hi"}]})
    state.get_deployment.return_value = (
        {"model_id": "org/model", "config": {"model_server": saved_server}} if saved_server else None
    )
    result = CliRunner().invoke(
        app,
        [
            "sagemaker",
            "invoke",
            "test-endpoint",
            "--input",
            "Hello",
            "--max-tokens",
            "7",
            "--api",
            api,
        ],
    )
    assert result.exit_code == 0, result.stdout
    provider.invoke.assert_called_once_with("test-endpoint", expected)


def test_vllm_environment_is_discovered_for_external_or_refreshed_endpoints(deployment_flow):
    provider, _, _, state = deployment_flow
    provider.client.sagemaker.describe_endpoint.return_value = {
        "EndpointName": "test-endpoint",
        "EndpointStatus": "InService",
        "EndpointConfigName": "config",
    }
    provider.client.sagemaker.describe_endpoint_config.return_value = {
        "ProductionVariants": [
            {"ModelName": "model", "InstanceType": "ml.g6.2xlarge", "InitialInstanceCount": 1}
        ],
    }
    provider.client.sagemaker.describe_model.return_value = {
        "PrimaryContainer": {"Environment": {"SM_VLLM_MODEL": "org/model", "HF_TOKEN": "secret"}},
    }
    deployment = provider.get_deployment("test-endpoint")
    assert deployment.model_id == "org/model"
    assert deployment.config["model_server"] == "vllm"
    assert "secret" not in str(deployment.to_dict())
    state.get_deployment.return_value = None
    provider.invoke = Mock(return_value={})
    result = CliRunner().invoke(app, ["sagemaker", "invoke", "test-endpoint", "--input", "Hello"])
    assert result.exit_code == 0, result.stdout
    provider.invoke.assert_called_once_with(
        "test-endpoint", {"model": "org/model", "prompt": "Hello", "max_tokens": 100}
    )
