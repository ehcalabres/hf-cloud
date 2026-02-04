"""Base provider interface for cloud providers."""

from abc import ABC, abstractmethod
from typing import Any

from hf_cloud.core.deployment import Deployment, DeploymentStatus


class CloudProvider(ABC):
    """Abstract base class for cloud providers."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name (e.g., 'sagemaker', 'azure', 'vertex')."""
        pass

    @abstractmethod
    def deploy(
        self,
        model_id: str,
        deployment_name: str,
        config: dict[str, Any],
        token: str | None = None,
    ) -> Deployment:
        """Deploy a model and return deployment information.

        Args:
            model_id: HuggingFace model ID (e.g., 'gpt2', 'bert-base-uncased')
            deployment_name: Name for the deployment/endpoint
            config: Provider-specific configuration
            token: Optional HuggingFace Hub token for private models

        Returns:
            Deployment object with deployment details
        """
        pass

    @abstractmethod
    def list_deployments(self, filters: dict[str, Any] | None = None) -> list[Deployment]:
        """List all deployments for this provider.

        Args:
            filters: Optional filters (e.g., {'status': 'running', 'region': 'us-east-1'})

        Returns:
            List of Deployment objects
        """
        pass

    @abstractmethod
    def get_deployment(self, deployment_id: str) -> Deployment:
        """Get detailed information about a specific deployment.

        Args:
            deployment_id: Unique deployment identifier

        Returns:
            Deployment object with full details
        """
        pass

    @abstractmethod
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete a deployment.

        Args:
            deployment_id: Unique deployment identifier

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    def get_logs(self, deployment_id: str, tail: int = 100) -> str:
        """Get deployment logs.

        Args:
            deployment_id: Unique deployment identifier
            tail: Number of log lines to retrieve

        Returns:
            Log output as string
        """
        pass

    @abstractmethod
    def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Get current deployment status.

        Args:
            deployment_id: Unique deployment identifier

        Returns:
            DeploymentStatus enum value
        """
        pass

    @abstractmethod
    def update_deployment(
        self,
        deployment_id: str,
        config: dict[str, Any],
    ) -> Deployment:
        """Update deployment configuration.

        Args:
            deployment_id: Unique deployment identifier
            config: New configuration parameters

        Returns:
            Updated Deployment object
        """
        pass

    @abstractmethod
    def invoke(
        self,
        deployment_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke the deployed model for inference.

        Args:
            deployment_id: Unique deployment identifier
            payload: Input data for inference

        Returns:
            Inference response
        """
        pass
