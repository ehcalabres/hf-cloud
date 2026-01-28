"""Google Cloud Vertex AI provider implementation (stub for Phase 1)."""

from typing import Any

from ...core.deployment import Deployment, DeploymentStatus
from ...core.exceptions import ProviderError
from ..base import CloudProvider


class GCPProvider(CloudProvider):
    """Google Cloud Vertex AI provider implementation (stub)."""

    def __init__(self) -> None:
        """Initialize GCP provider."""
        # Check if GCP SDK is installed
        try:
            import google.cloud.aiplatform  # noqa: F401
        except ImportError:
            raise ProviderError(
                "gcp",
                "Google Cloud AI Platform SDK is not installed. Install with: pip install hf-cloud[gcp]",
            )

    def get_provider_name(self) -> str:
        return "gcp"

    def deploy(
        self,
        model_id: str,
        deployment_name: str,
        config: dict[str, Any],
        token: str | None = None,
    ) -> Deployment:
        raise NotImplementedError("GCP provider will be implemented in Phase 3")

    def list_deployments(self, filters: dict[str, Any] | None = None) -> list[Deployment]:
        raise NotImplementedError("GCP provider will be implemented in Phase 3")

    def get_deployment(self, deployment_id: str) -> Deployment:
        raise NotImplementedError("GCP provider will be implemented in Phase 3")

    def delete_deployment(self, deployment_id: str) -> bool:
        raise NotImplementedError("GCP provider will be implemented in Phase 3")

    def get_logs(self, deployment_id: str, tail: int = 100) -> str:
        raise NotImplementedError("GCP provider will be implemented in Phase 3")

    def get_status(self, deployment_id: str) -> DeploymentStatus:
        raise NotImplementedError("GCP provider will be implemented in Phase 3")

    def update_deployment(
        self,
        deployment_id: str,
        config: dict[str, Any],
    ) -> Deployment:
        raise NotImplementedError("GCP provider will be implemented in Phase 3")

    def invoke(
        self,
        deployment_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError("GCP provider will be implemented in Phase 3")
