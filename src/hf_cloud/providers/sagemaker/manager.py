"""SageMaker management operations."""

import json
from datetime import datetime
from typing import Any

from ...core.deployment import Deployment, DeploymentStatus
from ...core.exceptions import DeploymentError, DeploymentNotFoundError, ProviderError
from .client import SageMakerClient

# Map SageMaker endpoint statuses to DeploymentStatus
STATUS_MAP = {
    "Creating": DeploymentStatus.CREATING,
    "InService": DeploymentStatus.RUNNING,
    "Updating": DeploymentStatus.UPDATING,
    "Deleting": DeploymentStatus.DELETING,
    "Failed": DeploymentStatus.FAILED,
    "OutOfService": DeploymentStatus.STOPPED,
    "RollingBack": DeploymentStatus.UPDATING,
    "SystemUpdating": DeploymentStatus.UPDATING,
}


class SageMakerManager:
    """Handles SageMaker management operations."""

    def __init__(self, client: SageMakerClient):
        """Initialize manager.

        Args:
            client: SageMaker client instance
        """
        self.client = client

    def list_deployments(self, filters: dict[str, Any] | None = None) -> list[Deployment]:
        """List SageMaker endpoints.

        Args:
            filters: Optional filters (status, region)

        Returns:
            List of Deployment objects
        """
        try:
            # List all endpoints
            paginator = self.client.sagemaker.get_paginator("list_endpoints")
            endpoints = []

            for page in paginator.paginate():
                for endpoint in page["Endpoints"]:
                    # Apply status filter if specified
                    if filters and "status" in filters:
                        status = STATUS_MAP.get(endpoint["EndpointStatus"], DeploymentStatus.FAILED)
                        if status.value != filters["status"]:
                            continue

                    # Get deployment details
                    try:
                        deployment = self._endpoint_to_deployment(endpoint)
                        endpoints.append(deployment)
                    except Exception:
                        # Skip endpoints we can't parse
                        continue

            return endpoints

        except Exception as e:
            raise ProviderError("sagemaker", f"Failed to list endpoints: {e}")

    def get_deployment(self, deployment_id: str) -> Deployment:
        """Get details about a specific endpoint.

        Args:
            deployment_id: Endpoint name

        Returns:
            Deployment object

        Raises:
            DeploymentNotFoundError: If endpoint not found
        """
        try:
            response = self.client.sagemaker.describe_endpoint(EndpointName=deployment_id)
            return self._describe_response_to_deployment(response)
        except self.client.sagemaker.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                raise DeploymentNotFoundError(deployment_id)
            raise ProviderError("sagemaker", f"Failed to get endpoint: {e}")

    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete a SageMaker endpoint and associated resources.

        Args:
            deployment_id: Endpoint name

        Returns:
            True if successful

        Raises:
            DeploymentNotFoundError: If endpoint not found
        """
        try:
            # Get endpoint config name
            endpoint_desc = self.client.sagemaker.describe_endpoint(EndpointName=deployment_id)
            endpoint_config_name = endpoint_desc["EndpointConfigName"]

            # Get model name from endpoint config
            config_desc = self.client.sagemaker.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            model_name = config_desc["ProductionVariants"][0]["ModelName"]

            # Delete endpoint
            self.client.sagemaker.delete_endpoint(EndpointName=deployment_id)

            # Wait for endpoint deletion (optional, can be async)
            # Delete endpoint config
            try:
                self.client.sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
            except Exception:
                pass  # Config might be shared or already deleted

            # Delete model
            try:
                self.client.sagemaker.delete_model(ModelName=model_name)
            except Exception:
                pass  # Model might be shared or already deleted

            return True

        except self.client.sagemaker.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                raise DeploymentNotFoundError(deployment_id)
            raise ProviderError("sagemaker", f"Failed to delete endpoint: {e}")

    def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Get the current status of an endpoint.

        Args:
            deployment_id: Endpoint name

        Returns:
            DeploymentStatus
        """
        deployment = self.get_deployment(deployment_id)
        return deployment.status

    def get_logs(self, deployment_id: str, tail: int = 100) -> str:
        """Get CloudWatch logs for an endpoint.

        Args:
            deployment_id: Endpoint name
            tail: Number of log lines

        Returns:
            Log output string
        """
        try:
            # SageMaker logs are in /aws/sagemaker/Endpoints/{endpoint_name}
            log_group_name = f"/aws/sagemaker/Endpoints/{deployment_id}"

            # Get log streams
            streams_response = self.client.logs.describe_log_streams(
                logGroupName=log_group_name,
                orderBy="LastEventTime",
                descending=True,
                limit=5,
            )

            if not streams_response.get("logStreams"):
                return "No logs available yet."

            # Get logs from the most recent stream
            log_lines = []
            for stream in streams_response["logStreams"][:3]:
                events_response = self.client.logs.get_log_events(
                    logGroupName=log_group_name,
                    logStreamName=stream["logStreamName"],
                    limit=tail // 3,  # Distribute across streams
                    startFromHead=False,
                )
                for event in events_response.get("events", []):
                    log_lines.append(event["message"])

            return "\n".join(log_lines[-tail:])

        except Exception as e:
            if "ResourceNotFoundException" in str(e):
                return "No logs available. The endpoint may still be starting."
            raise ProviderError("sagemaker", f"Failed to get logs: {e}")

    def update_deployment(self, deployment_id: str, config: dict[str, Any]) -> Deployment:
        """Update endpoint configuration.

        Args:
            deployment_id: Endpoint name
            config: New configuration (instance_type, instance_count)

        Returns:
            Updated Deployment object
        """
        try:
            # Get current endpoint info
            current = self.get_deployment(deployment_id)

            instance_type = config.get("instance_type", current.instance_type)
            instance_count = config.get("instance_count", current.instance_count)

            # Get current endpoint config to find model
            endpoint_desc = self.client.sagemaker.describe_endpoint(EndpointName=deployment_id)
            config_desc = self.client.sagemaker.describe_endpoint_config(
                EndpointConfigName=endpoint_desc["EndpointConfigName"]
            )
            model_name = config_desc["ProductionVariants"][0]["ModelName"]

            # Create new endpoint config
            new_config_name = f"{deployment_id}-config-{int(datetime.utcnow().timestamp())}"
            self.client.sagemaker.create_endpoint_config(
                EndpointConfigName=new_config_name,
                ProductionVariants=[
                    {
                        "VariantName": "AllTraffic",
                        "ModelName": model_name,
                        "InstanceType": instance_type,
                        "InitialInstanceCount": instance_count,
                    }
                ],
            )

            # Update endpoint
            self.client.sagemaker.update_endpoint(
                EndpointName=deployment_id,
                EndpointConfigName=new_config_name,
            )

            # Return updated deployment
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
            deployment_id: Endpoint name
            payload: Input data

        Returns:
            Inference response
        """
        try:
            response = self.client.sagemaker_runtime.invoke_endpoint(
                EndpointName=deployment_id,
                ContentType="application/json",
                Body=json.dumps(payload),
            )

            result = json.loads(response["Body"].read().decode())
            return result

        except self.client.sagemaker_runtime.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                raise DeploymentNotFoundError(deployment_id)
            raise ProviderError("sagemaker", f"Failed to invoke endpoint: {e}")

    def _endpoint_to_deployment(self, endpoint_summary: dict[str, Any]) -> Deployment:
        """Convert endpoint list summary to Deployment object.

        Args:
            endpoint_summary: Endpoint summary from list_endpoints

        Returns:
            Deployment object
        """
        endpoint_name = endpoint_summary["EndpointName"]
        status = STATUS_MAP.get(endpoint_summary["EndpointStatus"], DeploymentStatus.FAILED)

        return Deployment(
            deployment_id=endpoint_name,
            deployment_name=endpoint_name,
            provider="sagemaker",
            model_id="unknown",  # Not available in list response
            status=status,
            config={},
            created_at=endpoint_summary.get("CreationTime"),
            updated_at=endpoint_summary.get("LastModifiedTime"),
            region=self.client.region,
        )

    def _describe_response_to_deployment(self, response: dict[str, Any]) -> Deployment:
        """Convert describe_endpoint response to Deployment object.

        Args:
            response: Response from describe_endpoint

        Returns:
            Deployment object
        """
        endpoint_name = response["EndpointName"]
        status = STATUS_MAP.get(response["EndpointStatus"], DeploymentStatus.FAILED)

        # Try to get more details from endpoint config
        instance_type = None
        instance_count = None
        model_id = "unknown"

        try:
            config_name = response.get("EndpointConfigName")
            if config_name:
                config_desc = self.client.sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
                variant = config_desc["ProductionVariants"][0]
                instance_type = variant.get("InstanceType")
                instance_count = variant.get("InitialInstanceCount")

                # Try to get model info
                model_name = variant.get("ModelName")
                if model_name:
                    model_desc = self.client.sagemaker.describe_model(ModelName=model_name)
                    env = model_desc.get("PrimaryContainer", {}).get("Environment", {})
                    model_id = env.get("HF_MODEL_ID", "unknown")
        except Exception:
            pass

        return Deployment(
            deployment_id=endpoint_name,
            deployment_name=endpoint_name,
            provider="sagemaker",
            model_id=model_id,
            status=status,
            config={
                "instance_type": instance_type,
                "instance_count": instance_count,
            },
            endpoint_url=f"https://runtime.sagemaker.{self.client.region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
            created_at=response.get("CreationTime"),
            updated_at=response.get("LastModifiedTime"),
            region=self.client.region,
            instance_type=instance_type,
            instance_count=instance_count,
        )
