
"""Azure ML provider implementation (stub for Phase 1)."""

from typing import Any, Optional

from ...core.deployment import Deployment, DeploymentStatus
from ...core.exceptions import ProviderError
from ..base import CloudProvider


class AzureProvider(CloudProvider):
    """Azure ML provider implementation (stub)."""

    def __init__(self) -> None:
        """Initialize Azure provider."""
        # Check if Azure SDK is installed
        try:
            import azure.ai.ml  # noqa: F401
        except ImportError:
            raise ProviderError(
                "azure",
                "Azure ML SDK is not installed. Install with: pip install hf-cloud[azure]",
            )

    def get_provider_name(self) -> str:
        return "azure"

    def deploy(
        self,
        model_id: str,
        deployment_name: str,
        config: dict[str, Any],
        token: Optional[str] = None,
    ) -> Deployment:
        raise NotImplementedError("Azure provider will be implemented in Phase 3")

    def list_deployments(self, filters: Optional[dict[str, Any]] = None) -> list[Deployment]:
        raise NotImplementedError("Azure provider will be implemented in Phase 3")

    def get_deployment(self, deployment_id: str) -> Deployment:
        raise NotImplementedError("Azure provider will be implemented in Phase 3")

    def delete_deployment(self, deployment_id: str) -> bool:
        raise NotImplementedError("Azure provider will be implemented in Phase 3")

    def get_logs(self, deployment_id: str, tail: int = 100) -> str:
        raise NotImplementedError("Azure provider will be implemented in Phase 3")

    def get_status(self, deployment_id: str) -> DeploymentStatus:
        raise NotImplementedError("Azure provider will be implemented in Phase 3")

    def update_deployment(
        self,
        deployment_id: str,
        config: dict[str, Any],
    ) -> Deployment:
        raise NotImplementedError("Azure provider will be implemented in Phase 3")

    def invoke(
        self,
        deployment_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError("Azure provider will be implemented in Phase 3")
