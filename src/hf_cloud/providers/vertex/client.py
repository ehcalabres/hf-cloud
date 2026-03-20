
"""Google Cloud Vertex AI client wrapper."""

from typing import Any, Optional

from rich.console import Console

from hf_cloud.core.exceptions import AuthenticationError, ProviderError

console = Console()


class VertexClient:
    """Wrapper for Google Cloud Vertex AI clients."""

    def __init__(self, project: Optional[str] = None, location: str = "us-central1"):
        """Initialize Vertex AI client.

        Args:
            project: vertex project ID. If None, uses default from environment.
            location: GCP region/location.
        """
        self.project = project
        self.location = location
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize Vertex AI SDK if not already done."""
        if self._initialized:
            return

        try:
            from google.cloud import aiplatform

            aiplatform.init(
                project=self.project,
                location=self.location,
            )
            self._initialized = True
        except ImportError:
            raise ProviderError(
                "vertex",
                "Google Cloud AI Platform SDK is not installed. Install with: pip install hf-cloud[vertex]",
            )

    def get_project(self) -> str:
        """Get the current vertex project ID.

        Returns:
            Project ID

        Raises:
            ProviderError: If project cannot be determined
        """
        if self.project:
            return self.project

        try:
            import google.auth

            _, project = google.auth.default()
            if project:
                self.project = project
                return project
        except Exception:
            pass

        raise ProviderError(
            "vertex",
            "Could not determine vertex project",
            details="Set --project or configure gcloud with: gcloud config set project PROJECT_ID",
        )

    def verify_credentials(self) -> bool:
        """Verify GCP credentials are configured.

        Returns:
            True if credentials are valid

        Raises:
            AuthenticationError: If credentials are invalid
        """
        try:
            import google.auth

            credentials, project = google.auth.default()
            if credentials is None:
                raise AuthenticationError(
                    "vertex",
                    "GCP credentials not configured",
                    details="Run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS",
                )
            return True
        except Exception as e:
            if "credentials" in str(e).lower():
                raise AuthenticationError(
                    "vertex",
                    "GCP credentials not configured or invalid",
                    details="Run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS",
                )
            raise ProviderError("vertex", f"Failed to verify credentials: {e}")

    def get_endpoint_client(self) -> Any:
        """Get the Vertex AI Endpoint client.

        Returns:
            EndpointServiceClient
        """
        self._ensure_initialized()
        from google.cloud.aiplatform_v1 import EndpointServiceClient

        return EndpointServiceClient()

    def get_model_client(self) -> Any:
        """Get the Vertex AI Model client.

        Returns:
            ModelServiceClient
        """
        self._ensure_initialized()
        from google.cloud.aiplatform_v1 import ModelServiceClient

        return ModelServiceClient()

    def get_prediction_client(self) -> Any:
        """Get the Vertex AI Prediction client.

        Returns:
            PredictionServiceClient
        """
        self._ensure_initialized()
        from google.cloud.aiplatform_v1 import PredictionServiceClient

        return PredictionServiceClient()
