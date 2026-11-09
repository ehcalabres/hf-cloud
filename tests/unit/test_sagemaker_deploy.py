"""Regression coverage for SDK returns that hide a failed SageMaker endpoint."""

import sys
from types import ModuleType
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
    module = ModuleType("sagemaker.serve.model_builder")
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
    builder.deploy.assert_called_once_with(endpoint_name="test-endpoint", initial_instance_count=1, wait=True)
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
    builder.deploy.assert_called_once_with(endpoint_name="test-endpoint", initial_instance_count=1, wait=True)


def test_explicit_arn_overrides_configured_role(deployment_flow):
    provider, _, factory, _ = deployment_flow
    arn = "arn:aws:iam::123456789012:role/ChosenRole"
    result = deploy_command("--role", arn)
    assert result.exit_code == 0, result.output
    provider.client.get_execution_role.assert_called_once_with(role_name=arn)
    assert factory.call_args.kwargs["role_arn"] == arn


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
