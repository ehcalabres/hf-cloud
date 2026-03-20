
"""AWS SageMaker provider implementation."""

from typing import Any, Optional

from ...core.deployment import Deployment, DeploymentStatus
from ..base import CloudProvider
from .client import SageMakerClient
from .deployer import SageMakerDeployer
from .manager import SageMakerManager


class SageMakerProvider(CloudProvider):
    """AWS SageMaker provider implementation."""

    def __init__(self, region: Optional[str] = None):
        """Initialize SageMaker provider.

        Args:
            region: AWS region. Defaults to us-east-1.
        """
        self.default_region = region or "us-east-1"
        self.client = SageMakerClient(region=self.default_region)
        self.deployer = SageMakerDeployer(client=self.client, region=self.default_region)
        self.manager = SageMakerManager(self.client)

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "sagemaker"

    def deploy(
        self,
        model_id: str,
        deployment_name: str,
        config: dict[str, Any],
        token: Optional[str] = None,
    ) -> Deployment:
        """Deploy a model to SageMaker.

        Args:
            model_id: HuggingFace model ID
            deployment_name: Name for the endpoint
            config: SageMaker-specific configuration
            token: Optional HuggingFace Hub token

        Returns:
            Deployment object
        """
        return self.deployer.deploy(
            model_id=model_id,
            deployment_name=deployment_name,
            config=config,
            token=token,
        )

    def list_deployments(self, filters: Optional[dict[str, Any]] = None) -> list[Deployment]:
        """List SageMaker endpoints.

        Args:
            filters: Optional filters

        Returns:
            List of Deployment objects
        """
        return self.manager.list_deployments(filters)

    def get_deployment(self, deployment_id: str) -> Deployment:
        """Get endpoint details.

        Args:
            deployment_id: Endpoint name

        Returns:
            Deployment object
        """
        return self.manager.get_deployment(deployment_id)

    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete SageMaker endpoint.

        Args:
            deployment_id: Endpoint name

        Returns:
            True if successful
        """
        return self.manager.delete_deployment(deployment_id)

    def get_logs(self, deployment_id: str, tail: int = 100) -> str:
        """Get CloudWatch logs.

        Args:
            deployment_id: Endpoint name
            tail: Number of lines

        Returns:
            Log output
        """
        return self.manager.get_logs(deployment_id, tail)

    def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Get endpoint status.

        Args:
            deployment_id: Endpoint name

        Returns:
            DeploymentStatus
        """
        return self.manager.get_status(deployment_id)

    def update_deployment(
        self,
        deployment_id: str,
        config: dict[str, Any],
    ) -> Deployment:
        """Update endpoint configuration.

        Args:
            deployment_id: Endpoint name
            config: New configuration

        Returns:
            Updated Deployment object
        """
        return self.manager.update_deployment(deployment_id, config)

    def invoke(
        self,
        deployment_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke endpoint for inference.

        Args:
            deployment_id: Endpoint name
            payload: Input data

        Returns:
            Inference response
        """
        return self.manager.invoke(deployment_id, payload)
