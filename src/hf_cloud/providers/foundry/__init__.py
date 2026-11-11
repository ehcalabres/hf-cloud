"""Microsoft Foundry provider using published GlobalManagedCompute templates."""

import json
import re
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import Any, NoReturn, Optional

from hf_cloud.core.deployment import Deployment, DeploymentStatus
from hf_cloud.core.exceptions import ConfigurationError, DeploymentError, DeploymentNotFoundError, ProviderError
from hf_cloud.providers.base import CloudProvider
from hf_cloud.providers.foundry.client import FoundryClient
from hf_cloud.providers.foundry.profiles import ACCELERATOR_MEMORY_GIB, UPGRADE_OPTIONS, validate_asset_id


class FoundryProvider(CloudProvider):
    def __init__(self, subscription_id=None, resource_group=None, account_name=None, endpoint=None):
        self.client = FoundryClient(subscription_id, resource_group, account_name, endpoint)

    def get_provider_name(self) -> str:
        return "foundry"

    @staticmethod
    def _status(state: Optional[str]) -> DeploymentStatus:
        return {
            "Succeeded": DeploymentStatus.RUNNING,
            "Creating": DeploymentStatus.CREATING,
            "Accepted": DeploymentStatus.CREATING,
            "Updating": DeploymentStatus.UPDATING,
            "Deleting": DeploymentStatus.DELETING,
            "Failed": DeploymentStatus.FAILED,
            "Canceled": DeploymentStatus.FAILED,
            "Cancelled": DeploymentStatus.FAILED,
            "Disabled": DeploymentStatus.STOPPED,
        }.get(state, DeploymentStatus.FAILED)

    def _deployment(self, resource: Any) -> Deployment:
        properties = resource.properties
        tags = {}
        return Deployment(
            deployment_id=resource.id or self.client.resource_id(resource.name),
            deployment_name=resource.name,
            provider="foundry",
            model_id=properties.model,
            status=self._status(properties.provisioning_state),
            config={
                "subscription_id": self.client.subscription_id,
                "resource_group": self.client.resource_group,
                "account_name": self.client.account_name,
                "endpoint": self.client.endpoint,
                "model_asset_id": properties.model,
                "deployment_template": properties.deployment_template,
                "accelerator_type": properties.accelerator_type,
                "version_upgrade_option": properties.version_upgrade_option,
                "accelerators_per_instance": properties.accelerators_per_instance,
                "provisioning_details": properties.provisioning_details.as_dict()
                if properties.provisioning_details
                else None,
            },
            endpoint_url=f"{self.client.endpoint}/managed-deployments/{resource.name}/",
            region=self.client.account.location,
            instance_type=properties.accelerator_type,
            instance_count=resource.sku.capacity,
            tags=tags,
        )

    @staticmethod
    def _raise(error: Exception, name: Optional[str] = None) -> NoReturn:
        if isinstance(error, (ConfigurationError, ProviderError, DeploymentError)):
            raise error
        if getattr(error, "status_code", None) == 404 and name:
            raise DeploymentNotFoundError(name) from error
        raise ProviderError("foundry", str(error)) from error

    def get_deployment(self, deployment_id: str) -> Deployment:
        try:
            resource = self.client.management.managed_compute_deployments.get(
                **self.client.scope(deployment_id)
            )
            return self._deployment(resource)
        except Exception as error:
            self._raise(error, deployment_id)

    def list_deployments(self, filters: Optional[dict[str, Any]] = None) -> list[Deployment]:
        try:
            resources = self.client.management.managed_compute_deployments.list(**self.client.scope())
            deployments = [self._deployment(resource) for resource in resources]
            if filters and filters.get("status"):
                deployments = [d for d in deployments if d.status.value == filters["status"]]
            return deployments
        except Exception as error:
            self._raise(error)

    @staticmethod
    def deployment_resource(
        model_id: str, config: dict[str, Any], token: Optional[str] = None
    ) -> dict[str, Any]:
        """Prepare the documented SDK resource body, without authentication or provisioning."""
        if token:
            raise ConfigurationError(
                "Foundry managed compute uses pre-staged catalog weights; Hub tokens and gated/private Hub models are not supported"
            )
        if not model_id.startswith("azureml://"):
            from huggingface_hub.utils import HFValidationError, validate_repo_id

            try:
                validate_repo_id(model_id)
            except HFValidationError as error:
                raise ConfigurationError(
                    "Use a Hub repository ID or a published registry model URI", details=str(error)
                ) from error
            if model_id.count("/") != 1:
                raise ConfigurationError(
                    "Use a namespaced Hub model ID (org/model) or a published registry model URI"
                )
        model_asset = config.get("model_asset_id") or (model_id if model_id.startswith("azureml://") else None)
        if model_asset is None:
            raise ConfigurationError(
                "A Hub ID requires a published Foundry --model-asset-id or --profile; copy its registry asset reference from the Foundry catalog"
            )
        if model_id.startswith("azureml://") and model_id != model_asset:
            raise ConfigurationError("The positional model asset and --model-asset-id must match")
        validate_asset_id(model_asset, "models")
        template = validate_asset_id(config.get("deployment_template"), "deploymenttemplates")
        accelerator = config.get("accelerator_type")
        if accelerator not in ACCELERATOR_MEMORY_GIB:
            raise ConfigurationError(
                f"Choose a Foundry accelerator family: {', '.join(ACCELERATOR_MEMORY_GIB)}"
            )
        count = config.get("instance_count", 1)
        if type(count) is not int or count < 1:
            raise ConfigurationError("Foundry instance_count must be a positive integer")
        upgrade = config.get("version_upgrade_option", "NoAutoUpgrade")
        if upgrade not in UPGRADE_OPTIONS:
            raise ConfigurationError(
                f"Invalid version upgrade option; choose from {', '.join(sorted(UPGRADE_OPTIONS))}"
            )
        return {
            "sku": {"name": "GlobalManagedCompute", "capacity": count},
            "properties": {
                "model": model_asset,
                "deploymentTemplate": template,
                "acceleratorType": accelerator,
                "versionUpgradeOption": upgrade,
            },
        }

    def deploy(
        self, model_id: str, deployment_name: str, config: dict[str, Any], token: Optional[str] = None
    ) -> Deployment:
        scope = self.client.scope(deployment_name)
        resource = self.deployment_resource(model_id, config, token)
        try:
            # Validate the account and inference endpoint before starting a billable operation.
            self.client.endpoint
            operations = self.client.management.managed_compute_deployments
            try:
                operations.get(**scope)
            except Exception as error:
                if getattr(error, "status_code", None) != 404:
                    raise
            else:
                raise DeploymentError(
                    "A deployment with this name already exists; use foundry update to scale it",
                    deployment_id=deployment_name,
                )
            operations.begin_create_or_update(
                **scope,
                resource=resource,
            ).result()
            # Re-read instead of treating an SDK poller return as success.
            deployment = self.get_deployment(deployment_name)
            if deployment.status != DeploymentStatus.RUNNING:
                raise DeploymentError(
                    "Foundry deployment did not reach Succeeded",
                    deployment_id=deployment_name,
                    details=json.dumps(deployment.config.get("provisioning_details")),
                )
            if (
                deployment.instance_count != resource["sku"]["capacity"]
                or deployment.instance_type != resource["properties"]["acceleratorType"]
            ):
                raise DeploymentError(
                    "Provisioned Foundry accelerator or instance count differs from the request",
                    deployment_id=deployment_name,
                )
            deployment.model_id = model_id
            deployment.tags = {"hf-cloud": "true"}
            return deployment
        except Exception as error:
            self._raise(error)

    def delete_deployment(self, deployment_id: str) -> bool:
        try:
            self.client.management.managed_compute_deployments.begin_delete(
                **self.client.scope(deployment_id)
            ).result()
            return True
        except Exception as error:
            self._raise(error, deployment_id)

    def get_status(self, deployment_id: str) -> DeploymentStatus:
        return self.get_deployment(deployment_id).status

    def update_deployment(self, deployment_id: str, config: dict[str, Any]) -> Deployment:
        count = config.get("instance_count")
        if type(count) is not int or count < 1:
            raise ConfigurationError("Foundry updates require a positive instance_count")
        try:
            # Patch only capacity, preserving the published model, template, tags, and upgrade policy.
            self.client.management.managed_compute_deployments.begin_update(
                **self.client.scope(deployment_id),
                properties={"sku": {"name": "GlobalManagedCompute", "capacity": count}},
            ).result()
            deployment = self.get_deployment(deployment_id)
            if deployment.status != DeploymentStatus.RUNNING:
                raise DeploymentError(
                    "Foundry scaling did not reach Succeeded",
                    deployment_id=deployment_id,
                    details=json.dumps(deployment.config.get("provisioning_details")),
                )
            if deployment.instance_count != count:
                raise DeploymentError(
                    "Foundry capacity differs from the requested instance count", deployment_id=deployment_id
                )
            return deployment
        except Exception as error:
            self._raise(error, deployment_id)

    def invoke(self, deployment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.invoke_route(deployment_id, payload, "chat/completions")

    def invoke_route(self, deployment_id: str, payload: dict[str, Any], route: str) -> dict[str, Any]:
        import httpx

        name = self.client.scope(deployment_id)["deployment_name"]
        if not route or not all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in route.split("/")):
            raise ConfigurationError(
                "Inference route must be a relative path such as chat/completions or embeddings"
            )
        if payload.get("stream"):
            raise ConfigurationError("Streaming inference is not supported by this JSON command")
        try:
            if self.get_status(deployment_id) != DeploymentStatus.RUNNING:
                raise ProviderError("foundry", "Deployment is not running")
            access_token = self.client.credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            ).token
            # Dedicated managed route works beyond chat models; never accept an arbitrary URL.
            with httpx.Client(timeout=120, follow_redirects=False) as client:
                response = client.post(
                    f"{self.client.endpoint}/managed-deployments/{name}/{route}",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={**payload, "model": name}
                    if route.removeprefix("v1/")
                    in {"chat/completions", "completions", "embeddings", "responses"}
                    else payload,
                )
                if not response.is_success:
                    raise ProviderError(
                        "foundry", f"Inference failed (HTTP {response.status_code})", details=response.text
                    )
                return response.json()
        except Exception as error:
            self._raise(error, deployment_id)

    def get_logs(self, deployment_id: str, tail: int = 100) -> str:
        if type(tail) is not int or not 1 <= tail <= 1000:
            raise ConfigurationError("Log tail must be between 1 and 1000")
        scope = self.client.resource_id(deployment_id)
        start = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        try:
            events = self.client.monitor.activity_logs.list(
                filter=f"eventTimestamp ge '{start}' and resourceUri eq '{scope}'"
            )
            return "\n".join(json.dumps(event.as_dict(), default=str) for event in islice(events, tail))
        except Exception as error:
            self._raise(error, deployment_id)
