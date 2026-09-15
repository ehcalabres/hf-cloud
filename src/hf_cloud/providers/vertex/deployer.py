
"""Vertex AI deployment operations."""

from datetime import datetime
from typing import Any, Optional

from rich.console import Console

from hf_cloud.core.deployment import Deployment, DeploymentStatus
from hf_cloud.core.exceptions import DeploymentError
from hf_cloud.providers.vertex.client import VertexClient
from hf_cloud.providers.vertex.utils import (
    get_serving_container_uri,
    get_vertex_env_vars,
    sanitize_endpoint_name,
)
from hf_cloud.utils.hf_hub import get_model_task

console = Console()


class VertexDeployer:
    """Handles Vertex AI deployment operations."""

    def __init__(self, client: VertexClient, project: Optional[str] = None, location: str = "us-central1"):
        """Initialize deployer.

        Args:
            client: Vertex AI client instance
            project: vertex project ID
            location: GCP location/region
        """
        self.client = client
        self.project = project
        self.location = location

    def deploy(
        self,
        model_id: str,
        deployment_name: str,
        config: dict[str, Any],
        token: Optional[str] = None,
    ) -> Deployment:
        """Deploy a HuggingFace model to Vertex AI.

        Args:
            model_id: HuggingFace model ID
            deployment_name: Name for the endpoint
            config: Deployment configuration
            token: Optional HuggingFace Hub token

        Returns:
            Deployment object

        Raises:
            DeploymentError: If deployment fails
        """
        machine_type = config.get("machine_type", "n1-standard-4")
        accelerator_type = config.get("accelerator_type")
        accelerator_count = config.get("accelerator_count", 1)
        location = config.get("location", self.location)
        project = config.get("project", self.project) or self.client.get_project()
        min_replica_count = config.get("min_replica_count", 1)
        max_replica_count = config.get("max_replica_count", 1)

        # Sanitize endpoint name
        endpoint_name = (
            sanitize_endpoint_name(deployment_name) if deployment_name else sanitize_endpoint_name(model_id)
        )

        try:
            from google.cloud import aiplatform

            # Initialize Vertex AI
            aiplatform.init(project=project, location=location)

            console.print(
                f"[green]Deploying model '{model_id}' to Vertex AI endpoint '{endpoint_name}' in location '{location}'[/green]"
            )
            console.print(f"Using machine type '{machine_type}'")
            if accelerator_type:
                console.print(f"Using accelerator '{accelerator_type}' x {accelerator_count}")

            # Get task for the model from model info from HuggingFace Hub
            task = get_model_task(model_id=model_id, token=token)

            # Get the serving container URI for HuggingFace models
            serving_container_uri = get_serving_container_uri(location=location)

            # Build environment variables for the container
            env_vars = get_vertex_env_vars(model_id=model_id, token=token, task=task)

            console.print(f"Using serving container: {serving_container_uri}")

            # Upload model to Vertex AI Model Registry
            model = aiplatform.Model.upload(
                display_name=f"hf-{endpoint_name}",
                serving_container_image_uri=serving_container_uri,
                serving_container_environment_variables=env_vars,
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                labels={
                    "hf-cloud": "true",
                    "model-id": model_id.lower().replace("/", "--").replace(".", "-")[:63],
                },
            )

            console.print(f"[green]Model uploaded successfully: {model.resource_name}[/green]")

            # Create endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                labels={"hf-cloud": "true"},
            )

            console.print(f"[green]Endpoint created: {endpoint.resource_name}[/green]")

            # Deploy model to endpoint
            deployed_model_display_name = f"hf-{model_id.lower().replace('/', '--')[:50]}"

            # Build machine spec
            machine_spec = {"machine_type": machine_type}
            if accelerator_type:
                machine_spec["accelerator_type"] = accelerator_type
                machine_spec["accelerator_count"] = accelerator_count

            model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=deployed_model_display_name,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count if accelerator_type else 0,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
            )

            console.print(f"[green]Model deployed successfully to endpoint '{endpoint_name}'.[/green]")

            # Build deployment object
            deployment = Deployment(
                deployment_id=endpoint.resource_name,
                deployment_name=endpoint_name,
                provider="vertex",
                model_id=model_id,
                status=DeploymentStatus.CREATING,
                config=config,
                endpoint_url=f"https://{location}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict",
                region=location,
                instance_type=machine_type,
                instance_count=min_replica_count,
                created_at=datetime.utcnow(),
                tags={"hf-cloud": "true", "model_id": model_id},
            )

            return deployment

        except ImportError:
            raise DeploymentError(
                "Google Cloud AI Platform SDK is not installed",
                details="Install with: pip install --upgrade hf-cloud",
            )
        except Exception as e:
            raise DeploymentError(
                f"Failed to deploy model {model_id}",
                deployment_id=endpoint_name,
                details=str(e),
            )
