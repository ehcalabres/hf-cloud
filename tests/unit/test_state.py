"""Tests for state management."""

import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from hf_cloud.core.deployment import Deployment, DeploymentStatus
from hf_cloud.core.state import StateManager


@pytest.fixture
def temp_state_file():
    """Create a temporary state file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def state_manager(temp_state_file):
    """Create a state manager with temporary file."""
    return StateManager(state_file=temp_state_file)


@pytest.fixture
def sample_deployment():
    """Create a sample deployment."""
    return Deployment(
        deployment_id="test-endpoint-123",
        deployment_name="test-endpoint",
        provider="sagemaker",
        model_id="gpt2",
        status=DeploymentStatus.RUNNING,
        config={"instance_type": "ml.g5.xlarge"},
        region="us-east-1",
    )


class TestStateManager:
    """Tests for StateManager."""

    def test_add_deployment(self, state_manager, sample_deployment):
        """Test adding a deployment."""
        state_manager.add_deployment(sample_deployment)

        stored = state_manager.get_deployment(
            sample_deployment.deployment_id,
            provider=sample_deployment.provider,
            region=sample_deployment.region,
        )
        assert stored is not None
        assert stored["deployment_id"] == sample_deployment.deployment_id
        assert stored["model_id"] == "gpt2"

    def test_get_nonexistent_deployment(self, state_manager):
        """Test getting a deployment that doesn't exist."""
        result = state_manager.get_deployment("nonexistent-id", provider="sagemaker", region="us-east-1")
        assert result is None

    def test_remove_deployment(self, state_manager, sample_deployment):
        """Test removing a deployment."""
        state_manager.add_deployment(sample_deployment)

        result = state_manager.remove_deployment(
            sample_deployment.deployment_id,
            provider=sample_deployment.provider,
            region=sample_deployment.region,
        )
        assert result is True

        stored = state_manager.get_deployment(
            sample_deployment.deployment_id,
            provider=sample_deployment.provider,
            region=sample_deployment.region,
        )
        assert stored is None

    def test_remove_nonexistent_deployment(self, state_manager):
        """Test removing a deployment that doesn't exist."""
        result = state_manager.remove_deployment("nonexistent-id", provider="sagemaker", region="us-east-1")
        assert result is False

    def test_list_deployments(self, state_manager):
        """Test listing deployments."""
        deployment1 = Deployment(
            deployment_id="deploy-1",
            deployment_name="deploy-1",
            provider="sagemaker",
            model_id="gpt2",
            status=DeploymentStatus.RUNNING,
            config={},
        )
        deployment2 = Deployment(
            deployment_id="deploy-2",
            deployment_name="deploy-2",
            provider="azure",
            model_id="bert",
            status=DeploymentStatus.CREATING,
            config={},
        )

        state_manager.add_deployment(deployment1)
        state_manager.add_deployment(deployment2)

        all_deployments = state_manager.list_deployments()
        assert len(all_deployments) == 2

    def test_list_deployments_filter_by_provider(self, state_manager):
        """Test filtering deployments by provider."""
        deployment1 = Deployment(
            deployment_id="deploy-1",
            deployment_name="deploy-1",
            provider="sagemaker",
            model_id="gpt2",
            status=DeploymentStatus.RUNNING,
            config={},
        )
        deployment2 = Deployment(
            deployment_id="deploy-2",
            deployment_name="deploy-2",
            provider="azure",
            model_id="bert",
            status=DeploymentStatus.CREATING,
            config={},
        )

        state_manager.add_deployment(deployment1)
        state_manager.add_deployment(deployment2)

        sagemaker_deployments = state_manager.list_deployments(provider="sagemaker")
        assert len(sagemaker_deployments) == 1
        assert sagemaker_deployments[0]["provider"] == "sagemaker"

    def test_update_deployment_status(self, state_manager, sample_deployment):
        """Test updating deployment status."""
        state_manager.add_deployment(sample_deployment)

        result = state_manager.update_deployment_status(
            sample_deployment.deployment_id,
            "stopped",
            provider=sample_deployment.provider,
            region=sample_deployment.region,
        )
        assert result is True

        stored = state_manager.get_deployment(
            sample_deployment.deployment_id,
            provider=sample_deployment.provider,
            region=sample_deployment.region,
        )
        assert stored["status"] == "stopped"

    def test_persistence(self, temp_state_file, sample_deployment):
        """Test that state persists across manager instances."""
        # Create manager and add deployment
        manager1 = StateManager(state_file=temp_state_file)
        manager1.add_deployment(sample_deployment)

        # Create new manager instance
        manager2 = StateManager(state_file=temp_state_file)

        # Should see the same deployment
        stored = manager2.get_deployment(
            sample_deployment.deployment_id,
            provider=sample_deployment.provider,
            region=sample_deployment.region,
        )
        assert stored is not None
        assert stored["deployment_id"] == sample_deployment.deployment_id

    def test_clear(self, state_manager, sample_deployment):
        """Test clearing all deployments."""
        state_manager.add_deployment(sample_deployment)
        state_manager.clear()

        deployments = state_manager.list_deployments()
        assert len(deployments) == 0

    def test_get_deployment_object(self, state_manager, sample_deployment):
        """Test getting deployment as Deployment object."""
        state_manager.add_deployment(sample_deployment)

        deployment = state_manager.get_deployment_object(
            sample_deployment.deployment_id,
            provider=sample_deployment.provider,
            region=sample_deployment.region,
        )
        assert deployment is not None
        assert isinstance(deployment, Deployment)
        assert deployment.deployment_id == sample_deployment.deployment_id


def test_batch_updates_save_once_and_preserve_other_scopes(state_manager, sample_deployment, monkeypatch):
    state_manager.add_deployment(sample_deployment)
    save = Mock(wraps=state_manager._save_state)
    monkeypatch.setattr(state_manager, "_save_state", save)
    deployments = [
        replace(sample_deployment, status=DeploymentStatus.STOPPED),
        replace(sample_deployment, region="eu-west-1"),
        replace(sample_deployment, provider="vertex", region="us-central1"),
    ]
    state_manager.add_deployments(deployment for deployment in deployments)
    save.assert_called_once_with()
    reloaded = StateManager(state_manager.state_file)
    assert len(reloaded.list_deployments()) == 3
    for deployment in deployments:
        assert reloaded.get_deployment_object(
            deployment.deployment_id, provider=deployment.provider, region=deployment.region
        ) == deployment

    # A later partial refresh retains deployments from other regions/providers.
    state_manager.add_deployments([sample_deployment])
    assert len(state_manager.list_deployments()) == 3


def test_empty_batch_does_not_write(state_manager, monkeypatch):
    save = Mock()
    monkeypatch.setattr(state_manager, "_save_state", save)
    state_manager.add_deployments(iter([]))
    save.assert_not_called()


def test_failing_batch_iteration_does_not_partially_update(state_manager, sample_deployment, monkeypatch):
    def deployments():
        yield sample_deployment
        raise RuntimeError("Listing failed")

    save = Mock()
    monkeypatch.setattr(state_manager, "_save_state", save)
    with pytest.raises(RuntimeError, match="Listing failed"):
        state_manager.add_deployments(deployments())
    assert state_manager.list_deployments() == []
    save.assert_not_called()
