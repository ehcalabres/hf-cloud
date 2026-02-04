"""Vertex AI management operations."""

import json
from datetime import datetime
from typing import Any

from rich.console import Console

from hf_cloud.core.deployment import Deployment, DeploymentStatus
from hf_cloud.core.exceptions import DeploymentError, DeploymentNotFoundError, ProviderError
from hf_cloud.providers.vertex.client import VertexClient

console = Console()

# Map Vertex AI endpoint states to DeploymentStatus
STATUS_MAP = {
    "JOB_STATE_PENDING": DeploymentStatus.CREATING,
    "JOB_STATE_RUNNING": DeploymentStatus.RUNNING,
    "JOB_STATE_SUCCEEDED": DeploymentStatus.RUNNING,
    "JOB_STATE_FAILED": DeploymentStatus.FAILED,
    "JOB_STATE_CANCELLING": DeploymentStatus.DELETING,
    "JOB_STATE_CANCELLED": DeploymentStatus.STOPPED,
    # For endpoints
    "DEPLOYED": DeploymentStatus.RUNNING,
    "DEPLOYING": DeploymentStatus.CREATING,
    "UNDEPLOYING": DeploymentStatus.DELETING,
}


class VertexManager:
    """Handles Vertex AI management operations."""

    def __init__(self, client: VertexClient, project: str | None = None, location: str = "us-central1"):
        """Initialize manager.

        Args:
            client: Vertex AI client instance
            project: vertex project ID
            location: GCP location/region
        """
        self.client = client
        self.project = project
        self.location = location

    def _get_project(self) -> str:
        """Get the project ID."""
        return self.project or self.client.get_project()

    def list_deployments(self, filters: dict[str, Any] | None = None) -> list[Deployment]:
        """List Vertex AI endpoints.

        Args:
            filters: Optional filters

        Returns:
            List of Deployment objects
        """
        try:
            from google.cloud import aiplatform

            project = self._get_project()
            aiplatform.init(project=project, location=self.location)

            # List endpoints with hf-cloud label
            endpoints = aiplatform.Endpoint.list(
                filter='labels.hf-cloud="true"',
                order_by="create_time desc",
            )

            deployments = []
            for endpoint in endpoints:
                try:
                    deployment = self._endpoint_to_deployment(endpoint)
                    deployments.append(deployment)
                except Exception:
                    continue

            return deployments

        except Exception as e:
            raise ProviderError("vertex", f"Failed to list endpoints: {e}")

    def get_deployment(self, deployment_id: str) -> Deployment:
        """Get details about a specific endpoint.

        Args:
            deployment_id: Endpoint resource name or display name

        Returns:
            Deployment object

        Raises:
            DeploymentNotFoundError: If endpoint not found
        """
        try:
            from google.cloud import aiplatform

            project = self._get_project()
            aiplatform.init(project=project, location=self.location)

            # Try to get endpoint by resource name first
            if deployment_id.startswith("projects/"):
                endpoint = aiplatform.Endpoint(endpoint_name=deployment_id)
            else:
                # Search by display name
                endpoints = aiplatform.Endpoint.list(
                    filter=f'display_name="{deployment_id}"',
                )
                if not endpoints:
                    raise DeploymentNotFoundError(deployment_id)
                endpoint = endpoints[0]

            return self._endpoint_to_deployment(endpoint)

        except DeploymentNotFoundError:
            raise
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                raise DeploymentNotFoundError(deployment_id)
            raise ProviderError("vertex", f"Failed to get endpoint: {e}")

    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete a Vertex AI endpoint and associated resources.

        Args:
            deployment_id: Endpoint resource name or display name

        Returns:
            True if successful

        Raises:
            DeploymentNotFoundError: If endpoint not found
        """
        try:
            from google.cloud import aiplatform

            project = self._get_project()
            aiplatform.init(project=project, location=self.location)

            # Get endpoint
            if deployment_id.startswith("projects/"):
                endpoint = aiplatform.Endpoint(endpoint_name=deployment_id)
            else:
                endpoints = aiplatform.Endpoint.list(
                    filter=f'display_name="{deployment_id}"',
                )
                if not endpoints:
                    raise DeploymentNotFoundError(deployment_id)
                endpoint = endpoints[0]

            # Get deployed models to clean up
            deployed_models = endpoint.gca_resource.deployed_models if endpoint.gca_resource else []
            model_names = [dm.model for dm in deployed_models]

            # Undeploy all models first
            for deployed_model in deployed_models:
                try:
                    endpoint.undeploy(deployed_model_id=deployed_model.id)
                except Exception:
                    pass

            # Delete endpoint
            endpoint.delete(force=True)

            # Delete associated models
            for model_name in model_names:
                try:
                    model = aiplatform.Model(model_name=model_name)
                    model.delete()
                except Exception:
                    pass

            return True

        except DeploymentNotFoundError:
            raise
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                raise DeploymentNotFoundError(deployment_id)
            raise ProviderError("vertex", f"Failed to delete endpoint: {e}")

    def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Get the current status of an endpoint.

        Args:
            deployment_id: Endpoint resource name or display name

        Returns:
            DeploymentStatus
        """
        deployment = self.get_deployment(deployment_id)
        return deployment.status

    def get_logs(self, deployment_id: str, tail: int = 100) -> str:
        """Get logs for an endpoint from Cloud Logging.

        Args:
            deployment_id: Endpoint resource name or display name
            tail: Number of log lines

        Returns:
            Log output string
        """
        try:
            from google.cloud import logging as cloud_logging

            project = self._get_project()

            # Get endpoint ID for filtering logs
            endpoint_id = deployment_id
            if deployment_id.startswith("projects/"):
                endpoint_id = deployment_id.split("/")[-1]

            # Initialize Cloud Logging client
            client = cloud_logging.Client(project=project)

            # Filter for Vertex AI endpoint logs
            filter_str = f'resource.type="aiplatform.googleapis.com/Endpoint" AND resource.labels.endpoint_id="{endpoint_id}"'

            # Get recent logs
            entries = client.list_entries(
                filter_=filter_str,
                order_by=cloud_logging.DESCENDING,
                max_results=tail,
            )

            log_lines = []
            for entry in entries:
                timestamp = entry.timestamp.isoformat() if entry.timestamp else ""
                payload = entry.payload if hasattr(entry, "payload") else str(entry)
                if isinstance(payload, dict):
                    payload = json.dumps(payload)
                log_lines.append(f"{timestamp} {payload}")

            if not log_lines:
                return "No logs available yet."

            # Reverse to show oldest first
            log_lines.reverse()
            return "\n".join(log_lines)

        except ImportError:
            return "Cloud Logging SDK not installed. Install with: pip install google-cloud-logging"
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                return "No logs available. The endpoint may still be starting."
            raise ProviderError("vertex", f"Failed to get logs: {e}")

    def update_deployment(self, deployment_id: str, config: dict[str, Any]) -> Deployment:
        """Update endpoint configuration.

        Args:
            deployment_id: Endpoint resource name or display name
            config: New configuration

        Returns:
            Updated Deployment object
        """
        try:
            from google.cloud import aiplatform

            project = self._get_project()
            aiplatform.init(project=project, location=self.location)

            # Get endpoint
            if deployment_id.startswith("projects/"):
                endpoint = aiplatform.Endpoint(endpoint_name=deployment_id)
            else:
                endpoints = aiplatform.Endpoint.list(
                    filter=f'display_name="{deployment_id}"',
                )
                if not endpoints:
                    raise DeploymentNotFoundError(deployment_id)
                endpoint = endpoints[0]

            # Get current deployed model
            deployed_models = endpoint.gca_resource.deployed_models if endpoint.gca_resource else []
            if not deployed_models:
                raise DeploymentError(
                    "No models deployed to endpoint",
                    deployment_id=deployment_id,
                )

            deployed_model = deployed_models[0]
            machine_type = config.get("machine_type")
            min_replica_count = config.get("min_replica_count")
            max_replica_count = config.get("max_replica_count")

            # Update deployed model traffic split or machine spec
            # Note: Vertex AI requires redeploying to change machine type
            if min_replica_count or max_replica_count:
                endpoint.update(
                    traffic_split={deployed_model.id: 100},
                )

            return self.get_deployment(deployment_id)

        except DeploymentNotFoundError:
            raise
        except Exception as e:
            raise DeploymentError(
                "Failed to update endpoint",
                deployment_id=deployment_id,
                details=str(e),
            )

    def invoke(self, deployment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke endpoint for inference.

        Args:
            deployment_id: Endpoint resource name or display name
            payload: Input data

        Returns:
            Inference response
        """
        try:
            from google.cloud import aiplatform

            project = self._get_project()
            aiplatform.init(project=project, location=self.location)

            # Get endpoint
            if deployment_id.startswith("projects/"):
                endpoint = aiplatform.Endpoint(endpoint_name=deployment_id)
            else:
                endpoints = aiplatform.Endpoint.list(
                    filter=f'display_name="{deployment_id}"',
                )
                if not endpoints:
                    raise DeploymentNotFoundError(deployment_id)
                endpoint = endpoints[0]

            # Format payload for prediction
            instances = [
                {
                    "text_inputs": [
                        {
                            "role": "user",
                            "content": payload.get("inputs", ""),
                        }
                    ]
                }
            ]

            # Make prediction
            response = endpoint.predict(instances=instances, parameters={"top_k": 2})

            # Return predictions
            return {
                "predictions": response.predictions,
                "deployed_model_id": response.deployed_model_id,
            }

        except DeploymentNotFoundError:
            raise
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                raise DeploymentNotFoundError(deployment_id)
            raise ProviderError("vertex", f"Failed to invoke endpoint: {e}")

    def _endpoint_to_deployment(self, endpoint: Any) -> Deployment:
        """Convert Vertex AI Endpoint to Deployment object.

        Args:
            endpoint: Vertex AI Endpoint object

        Returns:
            Deployment object
        """
        # Determine status
        deployed_models = endpoint.gca_resource.deployed_models if endpoint.gca_resource else []
        if deployed_models:
            status = DeploymentStatus.RUNNING
        else:
            status = DeploymentStatus.CREATING

        # Get model info from first deployed model
        model_id = "unknown"
        machine_type = None
        replica_count = None

        if deployed_models:
            dm = deployed_models[0]
            machine_type = getattr(dm.dedicated_resources, "machine_spec", {})
            if hasattr(machine_type, "machine_type"):
                machine_type = machine_type.machine_type
            else:
                machine_type = None

            replica_count = getattr(dm.dedicated_resources, "min_replica_count", None)

            # Try to get model info
            try:
                from google.cloud import aiplatform

                model = aiplatform.Model(model_name=dm.model)
                env_vars = model.gca_resource.container_spec.env if model.gca_resource.container_spec else []
                for env in env_vars:
                    if env.name == "HF_MODEL_ID":
                        model_id = env.value
                        break
            except Exception:
                pass

        # Get timestamps
        created_at = None
        updated_at = None
        if endpoint.gca_resource:
            if endpoint.gca_resource.create_time:
                created_at = endpoint.gca_resource.create_time
            if endpoint.gca_resource.update_time:
                updated_at = endpoint.gca_resource.update_time

        return Deployment(
            deployment_id=endpoint.resource_name,
            deployment_name=endpoint.display_name,
            provider="vertex",
            model_id=model_id,
            status=status,
            config={
                "machine_type": machine_type,
            },
            endpoint_url=f"https://{self.location}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict",
            created_at=created_at,
            updated_at=updated_at,
            region=self.location,
            instance_type=machine_type,
            instance_count=replica_count,
        )
