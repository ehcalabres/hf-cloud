"""Hub model deployment on Azure ML managed online endpoints."""

import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, NoReturn, Optional

from hf_cloud.core.deployment import Deployment, DeploymentStatus
from hf_cloud.core.exceptions import ConfigurationError, DeploymentError, DeploymentNotFoundError, ProviderError
from hf_cloud.providers.azureml.client import AzureMLClient
from hf_cloud.providers.azureml.serving import RUNTIME_DIR, serving_plan
from hf_cloud.providers.base import CloudProvider


class AzureMLProvider(CloudProvider):
    def __init__(self, subscription_id=None, resource_group=None, workspace_name=None, deployment_name=None):
        self.client = AzureMLClient(subscription_id, resource_group, workspace_name, deployment_name)

    def get_provider_name(self) -> str:
        return "azureml"

    @staticmethod
    def _raise(error: Exception, endpoint_name=None) -> NoReturn:
        if isinstance(error, (ConfigurationError, DeploymentError, ProviderError)):
            raise error
        if getattr(error, "status_code", None) == 404 and endpoint_name:
            raise DeploymentNotFoundError(endpoint_name) from error
        raise ProviderError("azureml", str(error)) from error

    @staticmethod
    def _state(state: str) -> DeploymentStatus:
        return {
            "Succeeded": DeploymentStatus.RUNNING,
            "Creating": DeploymentStatus.CREATING,
            "Updating": DeploymentStatus.UPDATING,
            "Deleting": DeploymentStatus.DELETING,
            "Scaling": DeploymentStatus.UPDATING,
            "Canceled": DeploymentStatus.FAILED,
            "Failed": DeploymentStatus.FAILED,
        }.get(state, DeploymentStatus.FAILED)

    def _selected_deployment(self, endpoint):
        name = self.client.deployment_name or (endpoint.properties or {}).get("hf-cloud-deployment")
        if not name:
            active = [key for key, weight in (endpoint.traffic or {}).items() if weight > 0]
            if len(active) == 1:
                name = active[0]
            else:
                deployments = list(self.client.management.online_deployments.list(endpoint_name=endpoint.name))
                if len(deployments) != 1:
                    raise ConfigurationError(
                        "Select --deployment-name before the subcommand for an endpoint with multiple or no deployments"
                    )
                return deployments[0]
        try:
            return self.client.management.online_deployments.get(name=name, endpoint_name=endpoint.name)
        except Exception as error:
            if (
                getattr(error, "status_code", None) == 404
                and not self.client.deployment_name
                and (endpoint.tags or {}).get("hf-cloud") == "true"
            ):
                return None  # Endpoint creation succeeded but component creation has not completed.
            raise

    def _deployment(self, endpoint, component) -> Deployment:
        metadata = (component.properties or {}) if component else {}
        task_config = json.loads(metadata.get("hf-cloud-config", "{}"))
        allowed = {
            key: task_config[key]
            for key in (
                "task",
                "image",
                "gpu_count",
                "revision",
                "max_model_len",
                "batch_size",
                "kv_cache_dtype",
                "max_num_batched_tokens",
                "safety_margin",
                "dtype",
                "trust_remote_code",
                "hf_token_connection",
            )
            if key in task_config
        }
        state = self._state(component.provisioning_state) if component else DeploymentStatus.FAILED
        if endpoint.provisioning_state != "Succeeded":
            state = self._state(endpoint.provisioning_state)
        return Deployment(
            deployment_id=endpoint.id or self.client.resource_id(endpoint.name),
            deployment_name=endpoint.name,
            provider="azureml",
            model_id=metadata.get(
                "hf-cloud-model-id", (endpoint.properties or {}).get("hf-cloud-model-id", "unknown")
            ),
            status=state,
            region=self.client.workspace.location,
            instance_type=component.instance_type if component else None,
            instance_count=component.instance_count if component else 0,
            endpoint_url=endpoint.scoring_uri,
            config={
                **allowed,
                "subscription_id": self.client.subscription_id,
                "resource_group": self.client.resource_group,
                "workspace_name": self.client.workspace_name,
                "deployment_name": component.name
                if component
                else (endpoint.properties or {}).get("hf-cloud-deployment"),
                "task": metadata.get("hf-cloud-task", "text-generation"),
                "traffic": endpoint.traffic or {},
                **(
                    {
                        "failure_reason": "Serving deployment is missing; inspect Azure ML activity logs or delete the incomplete endpoint."
                    }
                    if not component
                    else {}
                ),
            },
            tags=endpoint.tags or {},
        )

    def get_deployment(self, deployment_id: str) -> Deployment:
        try:
            name = self.client.endpoint_name(deployment_id)
            endpoint = self.client.management.online_endpoints.get(name=name)
            return self._deployment(endpoint, self._selected_deployment(endpoint))
        except Exception as error:
            self._raise(error, deployment_id)

    def list_deployments(self, filters: Optional[dict[str, Any]] = None) -> list[Deployment]:
        try:
            output = []
            for endpoint in self.client.management.online_endpoints.list():
                if (endpoint.tags or {}).get("hf-cloud") != "true":
                    continue
                component = self._selected_deployment(endpoint)
                deployment = self._deployment(endpoint, component)
                if not filters or not filters.get("status") or filters["status"] == deployment.status.value:
                    output.append(deployment)
            return output
        except Exception as error:
            self._raise(error)

    def _environment(self, plan):
        from azure.ai.ml.entities import BuildContext, Environment

        settings = plan["environment"]
        operations = self.client.management.environments
        try:
            return operations.get(name=settings["name"], version=settings["version"])
        except Exception as error:
            if getattr(error, "status_code", None) != 404:
                raise
        with TemporaryDirectory(prefix="hf-cloud-azureml-") as directory:
            context = Path(directory)
            (context / "Dockerfile").write_text(settings["dockerfile"])
            shutil.copyfile(RUNTIME_DIR / "start.py", context / "start.py")
            return operations.create_or_update(
                Environment(
                    name=settings["name"],
                    version=settings["version"],
                    build=BuildContext(path=directory, dockerfile_path="Dockerfile"),
                    inference_config=settings["inference_config"],
                )
            )

    def deploy(
        self, model_id: str, deployment_name: str, config: dict[str, Any], token: Optional[str] = None
    ) -> Deployment:
        plan = serving_plan(model_id, deployment_name, config, token)
        self.client.require_scope()
        try:
            from azure.ai.ml.entities import (
                IdentityConfiguration,
                ManagedOnlineDeployment,
                ManagedOnlineEndpoint,
                OnlineRequestSettings,
                ProbeSettings,
            )

            workspace = self.client.workspace
            client = self.client.management
            try:
                client.online_endpoints.get(name=deployment_name)
            except Exception as error:
                if getattr(error, "status_code", None) != 404:
                    raise
            else:
                raise DeploymentError(
                    "Endpoint already exists; choose a new --name or use azureml update to scale it",
                    deployment_id=deployment_name,
                )
            sizes = list(client.compute.list_sizes(location=workspace.location))
            size = next((size for size in sizes if size.name == plan["deployment"]["instance_type"]), None)
            if size is None or size.gpus != plan_gpu_count(plan):
                raise ConfigurationError(
                    "Selected VM GPU layout is unavailable or differs from workspace regional size metadata"
                )
            if config.get("hf_token_connection"):
                client.connections.get(config["hf_token_connection"], populate_secrets=False)
            environment = self._environment(plan)
            endpoint = ManagedOnlineEndpoint(
                **plan["endpoint"], identity=IdentityConfiguration(type="system_assigned")
            )
            client.online_endpoints.begin_create_or_update(endpoint).result()
            component = ManagedOnlineDeployment(
                **plan["deployment"],
                environment=f"azureml:{environment.name}:{environment.version}",
                request_settings=OnlineRequestSettings(request_timeout_ms=180000),
                liveness_probe=ProbeSettings(initial_delay=600, period=10, failure_threshold=30),
                readiness_probe=ProbeSettings(initial_delay=600, period=10, failure_threshold=30),
            )
            client.online_deployments.begin_create_or_update(component).result()
            endpoint = client.online_endpoints.get(name=deployment_name)
            endpoint.traffic = plan["traffic"]
            client.online_endpoints.begin_create_or_update(endpoint).result()
            deployment = self.get_deployment(deployment_name)
            traffic = {name: weight for name, weight in deployment.config["traffic"].items() if weight}
            if (
                deployment.status != DeploymentStatus.RUNNING
                or traffic != plan["traffic"]
                or deployment.instance_type != plan["deployment"]["instance_type"]
                or deployment.instance_count != plan["deployment"]["instance_count"]
            ):
                raise DeploymentError(
                    "Azure ML endpoint/deployment did not become ready with the requested traffic",
                    deployment_id=deployment_name,
                )
            return deployment
        except ImportError as error:
            raise ProviderError(
                "azureml", "Azure ML SDK is missing; run uv sync or pip install --upgrade hf-cloud"
            ) from error
        except Exception as error:
            self._raise(error)

    def update_deployment(self, deployment_id: str, config: dict[str, Any]) -> Deployment:
        count = config.get("instance_count")
        if type(count) is not int or count < 1 or set(config) != {"instance_count"}:
            raise ConfigurationError("Azure ML scaling requires only a positive instance_count")
        try:
            name = self.client.endpoint_name(deployment_id)
            endpoint = self.client.management.online_endpoints.get(name=name)
            component = self._selected_deployment(endpoint)
            if component is None:
                raise DeploymentError(
                    "Serving deployment is missing; cannot scale an incomplete endpoint",
                    deployment_id=deployment_id,
                )
            component.instance_count = count
            self.client.management.online_deployments.begin_create_or_update(component).result()
            deployment = self.get_deployment(deployment_id)
            if deployment.status != DeploymentStatus.RUNNING or deployment.instance_count != count:
                raise DeploymentError(
                    "Azure ML scaling did not reach the requested running instance count",
                    deployment_id=deployment_id,
                )
            return deployment
        except Exception as error:
            self._raise(error, deployment_id)

    def delete_deployment(self, deployment_id: str) -> bool:
        try:
            self.client.management.online_endpoints.begin_delete(
                name=self.client.endpoint_name(deployment_id)
            ).result()
            return True
        except Exception as error:
            self._raise(error, deployment_id)

    def get_status(self, deployment_id: str) -> DeploymentStatus:
        return self.get_deployment(deployment_id).status

    def get_logs(self, deployment_id: str, tail: int = 100, *, container_type="inference-server") -> str:
        if (
            type(tail) is not int
            or not 1 <= tail <= 1000
            or container_type not in {"inference-server", "storage-initializer"}
        ):
            raise ConfigurationError("Select a valid log container and 1–1000 lines")
        try:
            name = self.client.endpoint_name(deployment_id)
            endpoint = self.client.management.online_endpoints.get(name=name)
            component = self._selected_deployment(endpoint)
            if component is None:
                raise DeploymentError(
                    "Serving deployment is missing; inspect Azure ML activity logs", deployment_id=deployment_id
                )
            return self.client.management.online_deployments.get_logs(
                name=component.name, endpoint_name=name, lines=tail, container_type=container_type
            )
        except Exception as error:
            self._raise(error, deployment_id)

    def invoke(self, deployment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stream"):
            raise ConfigurationError("Streaming is not supported by the JSON inference command")
        try:
            deployment = self.get_deployment(deployment_id)
            if deployment.status != DeploymentStatus.RUNNING:
                raise ProviderError("azureml", "Endpoint is not running")
            with NamedTemporaryFile(mode="w+", suffix=".json", prefix="hf-cloud-request-") as file:
                model_id = deployment.model_id if deployment.model_id != "unknown" else payload.get("model")
                if not model_id:
                    raise ConfigurationError("Untracked endpoint: supply its served model name in --json-file")
                json.dump({**payload, "model": model_id}, file)
                file.flush()
                result = self.client.management.online_endpoints.invoke(
                    endpoint_name=self.client.endpoint_name(deployment_id),
                    deployment_name=deployment.config["deployment_name"],
                    request_file=file.name,
                )
            return json.loads(result)
        except Exception as error:
            self._raise(error, deployment_id)


def plan_gpu_count(plan):
    return json.loads(plan["deployment"]["properties"]["hf-cloud-config"])["gpu_count"]
