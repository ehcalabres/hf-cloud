"""Tests for deployment data model."""

from datetime import datetime

import pytest

from hf_cloud.core.deployment import Deployment, DeploymentStatus


class TestDeploymentStatus:
    """Tests for DeploymentStatus enum."""

    def test_status_values(self) -> None:
        """Test that all status values are strings."""
        assert DeploymentStatus.CREATING.value == "creating"
        assert DeploymentStatus.RUNNING.value == "running"
        assert DeploymentStatus.UPDATING.value == "updating"
        assert DeploymentStatus.DELETING.value == "deleting"
        assert DeploymentStatus.FAILED.value == "failed"
        assert DeploymentStatus.STOPPED.value == "stopped"

    def test_status_from_string(self) -> None:
        """Test creating status from string."""
        assert DeploymentStatus("running") == DeploymentStatus.RUNNING
        assert DeploymentStatus("creating") == DeploymentStatus.CREATING


class TestDeployment:
    """Tests for Deployment dataclass."""

    def test_create_deployment(self) -> None:
        """Test creating a deployment."""
        deployment = Deployment(
            deployment_id="test-123",
            deployment_name="my-deployment",
            provider="sagemaker",
            model_id="gpt2",
            status=DeploymentStatus.RUNNING,
            config={"instance_type": "ml.g5.xlarge"},
        )

        assert deployment.deployment_id == "test-123"
        assert deployment.deployment_name == "my-deployment"
        assert deployment.provider == "sagemaker"
        assert deployment.model_id == "gpt2"
        assert deployment.status == DeploymentStatus.RUNNING

    def test_deployment_to_dict(self) -> None:
        """Test converting deployment to dictionary."""
        now = datetime.utcnow()
        deployment = Deployment(
            deployment_id="test-123",
            deployment_name="my-deployment",
            provider="sagemaker",
            model_id="gpt2",
            status=DeploymentStatus.RUNNING,
            config={"instance_type": "ml.g5.xlarge"},
            created_at=now,
            region="us-east-1",
            instance_type="ml.g5.xlarge",
        )

        data = deployment.to_dict()

        assert data["deployment_id"] == "test-123"
        assert data["status"] == "running"
        assert data["created_at"] == now.isoformat()
        assert data["config"]["instance_type"] == "ml.g5.xlarge"

    def test_deployment_from_dict(self) -> None:
        """Test creating deployment from dictionary."""
        data = {
            "deployment_id": "test-456",
            "deployment_name": "my-deployment",
            "provider": "sagemaker",
            "model_id": "bert-base",
            "status": "creating",
            "config": {},
            "created_at": "2024-01-01T12:00:00",
        }

        deployment = Deployment.from_dict(data)

        assert deployment.deployment_id == "test-456"
        assert deployment.status == DeploymentStatus.CREATING
        assert deployment.created_at == datetime.fromisoformat("2024-01-01T12:00:00")

    def test_deployment_roundtrip(self) -> None:
        """Test converting to dict and back."""
        original = Deployment(
            deployment_id="test-789",
            deployment_name="roundtrip-test",
            provider="azure",
            model_id="gpt2",
            status=DeploymentStatus.RUNNING,
            config={"vm_size": "Standard_DS3_v2"},
            region="westus2",
            instance_type="Standard_DS3_v2",
            instance_count=2,
            tags={"env": "test"},
        )

        data = original.to_dict()
        restored = Deployment.from_dict(data)

        assert restored.deployment_id == original.deployment_id
        assert restored.status == original.status
        assert restored.config == original.config
        assert restored.tags == original.tags
