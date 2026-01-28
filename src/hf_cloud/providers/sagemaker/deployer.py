"""SageMaker deployment operations using ModelBuilder."""

from datetime import datetime
from typing import Any

from rich.console import Console

from ...core.deployment import Deployment, DeploymentStatus
from ...core.exceptions import DeploymentError
from .client import SageMakerClient
from .utils import (
    get_model_task,
    get_sagemaker_endpoint_name,
    get_sagemaker_env_vars,
    get_schema_builder_from_model,
    sanitize_endpoint_name,
)

console = Console()


class SageMakerDeployer:
    """Handles SageMaker deployment operations using ModelBuilder."""

    def __init__(self, client: SageMakerClient, region: str = "us-east-1"):
        """Initialize deployer.

        Args:
            client: SageMaker client instance
            region: Default AWS region
        """
        self.client = client
        self.default_region = region

    def deploy(
        self,
        model_id: str,
        deployment_name: str,
        config: dict[str, Any],
        token: str | None = None,
    ) -> Deployment:
        """Deploy a HuggingFace model to SageMaker using ModelBuilder.

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
        # Extract config
        instance_type = config.get("instance_type")
        region = config.get("region", self.default_region)
        role = config.get("role")
        instance_count = config.get("instance_count", 1)

        if not instance_type:
            raise DeploymentError(
                "Instance type is required",
                details="Specify --instance-type (e.g., ml.g5.xlarge)",
            )

        # Sanitize endpoint name or generate one
        endpoint_name = (
            sanitize_endpoint_name(deployment_name)
            if deployment_name
            else get_sagemaker_endpoint_name(model_id)
        )

        try:
            # Import ModelBuilder here to avoid import errors when SDK not installed
            from sagemaker.serve.model_builder import ModelBuilder

            # Get SageMaker session and role
            session = self.client.get_sagemaker_session(region=region)
            role_arn = self.client.get_execution_role(role_name=role)

            # Get model task and schema builder
            task = get_model_task(model_id=model_id, token=token)
            schema_builder = get_schema_builder_from_model(model_id=model_id, task=task)

            console.print(
                f"[green]Deploying model '{model_id}' to SageMaker endpoint '{endpoint_name}' in region '{region}'[/green]"
            )
            console.print(f"Using instance type '{instance_type}' with count {instance_count}")
            console.print(f"Model task detected as '{task}'")
            console.print(f"Using execution role '{role_arn}'")
            console.print(f"Schema input sample: {schema_builder.get_input_sample()}")

            # Create ModelBuilder
            model_builder = ModelBuilder(
                model=model_id,
                role_arn=role_arn,
                sagemaker_session=session,
                instance_type=instance_type,
                schema_builder=schema_builder,
                env_vars=get_sagemaker_env_vars(model_id=model_id, token=token),
            )

            # Build the model
            model_builder.build()

            console.print(
                f"[green]Model built successfully. Deploying to endpoint '{endpoint_name}'...[/green]"
            )

            # Deploy to endpoint
            model_builder.deploy(
                endpoint_name=endpoint_name,
                initial_instance_count=instance_count,
            )

            console.print(f"[green]Model deployed successfully to endpoint '{endpoint_name}'.[/green]")

            # Build deployment object
            deployment = Deployment(
                deployment_id=endpoint_name,
                deployment_name=deployment_name or endpoint_name,
                provider="sagemaker",
                model_id=model_id,
                status=DeploymentStatus.CREATING,
                config=config,
                endpoint_url=f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
                region=region,
                instance_type=instance_type,
                instance_count=instance_count,
                created_at=datetime.utcnow(),
                tags={"hf-cloud": "true", "model_id": model_id},
            )

            return deployment

        except ImportError:
            raise DeploymentError(
                "SageMaker SDK is not installed",
                details="Install with: pip install hf-cloud[sagemaker]",
            )
        except Exception as e:
            raise DeploymentError(
                f"Failed to deploy model {model_id}",
                deployment_id=endpoint_name,
                details=str(e),
            )
