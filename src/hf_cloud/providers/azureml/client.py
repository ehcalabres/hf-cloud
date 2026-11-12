"""Workspace-scoped Azure ML clients with lazy SDK/authentication loading."""

import os
import re
from typing import Any

from hf_cloud.core.config import Config
from hf_cloud.core.exceptions import ConfigurationError, ProviderError
from hf_cloud.providers.azureml.serving import validate_name


class AzureMLClient:
    def __init__(self, subscription_id=None, resource_group=None, workspace_name=None, deployment_name=None):
        defaults = Config().get_provider_config("azureml")
        self.subscription_id = (
            subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID") or defaults.get("subscription_id")
        )
        self.resource_group = (
            resource_group or os.getenv("AZURE_RESOURCE_GROUP") or defaults.get("resource_group")
        )
        self.workspace_name = (
            workspace_name or os.getenv("AZUREML_WORKSPACE_NAME") or defaults.get("workspace_name")
        )
        self.deployment_name = deployment_name
        self._credential = None
        self._management = None
        self._workspace = None

    def require_scope(self):
        missing = [
            key for key in ("subscription_id", "resource_group", "workspace_name") if not getattr(self, key)
        ]
        if missing:
            raise ConfigurationError(
                f"Missing Azure ML configuration: {', '.join(missing)}",
                details="Run hf-cloud providers configure azureml, or select --subscription-id, --resource-group and --workspace",
            )
        for key in ("subscription_id", "resource_group", "workspace_name"):
            if not re.fullmatch(r"[A-Za-z0-9_.()-]+", getattr(self, key)):
                raise ConfigurationError(f"Invalid Azure ML {key}")

    @property
    def management(self) -> Any:
        self.require_scope()
        if self._management is None:
            try:
                from azure.ai.ml import MLClient
                from azure.identity import DefaultAzureCredential
            except ImportError as error:
                raise ProviderError(
                    "azureml", "Install Azure ML dependencies with uv sync or pip install --upgrade hf-cloud"
                ) from error
            self._credential = self._credential or DefaultAzureCredential()
            self._management = MLClient(
                self._credential, self.subscription_id, self.resource_group, self.workspace_name
            )
        return self._management

    @property
    def workspace(self) -> Any:
        if self._workspace is None:
            self._workspace = self.management.workspaces.get(self.workspace_name)
        return self._workspace

    def endpoint_name(self, value: str) -> str:
        self.require_scope()
        if value.lower().startswith("/subscriptions/"):
            prefix = self.workspace_id + "/onlineEndpoints/"
            if not value.lower().startswith(prefix.lower()):
                raise ConfigurationError(
                    "Endpoint ID belongs to a different Azure ML workspace; select its subscription, resource group and workspace"
                )
            value = value[len(prefix) :]
        return validate_name(value)

    @property
    def workspace_id(self) -> str:
        self.require_scope()
        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{self.workspace_name}"

    def resource_id(self, name: str) -> str:
        return self.workspace_id + "/onlineEndpoints/" + self.endpoint_name(name)
