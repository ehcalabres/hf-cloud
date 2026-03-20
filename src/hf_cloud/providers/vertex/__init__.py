
"""Google Cloud Vertex AI provider implementation."""

from typing import Any, Optional

from hf_cloud.core.deployment import Deployment, DeploymentStatus
from hf_cloud.providers.base import CloudProvider
from hf_cloud.providers.vertex.client import VertexClient
from hf_cloud.providers.vertex.deployer import VertexDeployer
from hf_cloud.providers.vertex.manager import VertexManager


class VertexProvider(CloudProvider):
    """Google Cloud Vertex AI provider implementation."""

    def __init__(self, project: Optional[str] = None, location: Optional[str] = None):
        """Initialize Vertex AI provider.

        Args:
            project: vertex project ID. If None, uses default from environment.
            location: GCP location/region. Defaults to us-central1.
        """
        self.project = project
        self.default_location = location or "us-central1"
        self.client = VertexClient(project=project, location=self.default_location)
        self.deployer = VertexDeployer(client=self.client, project=project, location=self.default_location)
        self.manager = VertexManager(client=self.client, project=project, location=self.default_location)

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "vertex"

    def deploy(
        self,
        model_id: str,
        deployment_name: str,
        config: dict[str, Any],
        token: Optional[str] = None,
    ) -> Deployment:
        """Deploy a model to Vertex AI.

        Args:
            model_id: HuggingFace model ID
            deployment_name: Name for the endpoint
            config: Vertex AI-specific configuration
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
        """List Vertex AI endpoints.

        Args:
            filters: Optional filters

        Returns:
            List of Deployment objects
        """
        return self.manager.list_deployments(filters)

    def get_deployment(self, deployment_id: str) -> Deployment:
        """Get endpoint details.

        Args:
            deployment_id: Endpoint resource name or display name

        Returns:
            Deployment object
        """
        return self.manager.get_deployment(deployment_id)

    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete Vertex AI endpoint.

        Args:
            deployment_id: Endpoint resource name or display name

        Returns:
            True if successful
        """
        return self.manager.delete_deployment(deployment_id)

    def get_logs(self, deployment_id: str, tail: int = 100) -> str:
        """Get Cloud Logging logs.

        Args:
            deployment_id: Endpoint resource name or display name
            tail: Number of lines

        Returns:
            Log output
        """
        return self.manager.get_logs(deployment_id, tail)

    def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Get endpoint status.

        Args:
            deployment_id: Endpoint resource name or display name

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
            deployment_id: Endpoint resource name or display name
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
            deployment_id: Endpoint resource name or display name
            payload: Input data

        Returns:
            Inference response
        """
        return self.manager.invoke(deployment_id, payload)


# Keep backward compatibility with old name
vertexProvider = VertexProvider
