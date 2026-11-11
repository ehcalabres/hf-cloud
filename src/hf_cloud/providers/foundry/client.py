"""Lazy clients for the Foundry account control plane and unified inference endpoint."""

import os
import re
from typing import Any, Optional
from urllib.parse import urlsplit

from hf_cloud.core.config import Config
from hf_cloud.core.exceptions import ConfigurationError, ProviderError

INSTALL_HELP = "Install Foundry support with: pip install 'hf-cloud[foundry]' (or uv sync --extra foundry)"


class FoundryClient:
    def __init__(
        self,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        account_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        defaults = Config().get_provider_config("foundry")
        self.subscription_id = (
            subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID") or defaults.get("subscription_id")
        )
        self.resource_group = (
            resource_group or os.getenv("AZURE_RESOURCE_GROUP") or defaults.get("resource_group")
        )
        self.account_name = account_name or os.getenv("AZURE_FOUNDRY_ACCOUNT") or defaults.get("account_name")
        configured_scope = all(
            getattr(self, key) == defaults.get(key)
            for key in ("subscription_id", "resource_group", "account_name")
        )
        self.endpoint_override = (
            endpoint
            or os.getenv("AZURE_FOUNDRY_ENDPOINT")
            or (defaults.get("endpoint") if configured_scope else None)
        )
        self._credential = None
        self._management = None
        self._monitor = None
        self._account = None

    def require_scope(self) -> None:
        missing = [
            key for key in ("subscription_id", "resource_group", "account_name") if not getattr(self, key)
        ]
        if missing:
            raise ConfigurationError(
                f"Missing Foundry configuration: {', '.join(missing)}",
                details="Run hf-cloud providers configure foundry, or pass --subscription-id, --resource-group and --account",
            )
        if not re.fullmatch(r"[A-Za-z0-9-]+", self.subscription_id):
            raise ConfigurationError("Invalid Azure subscription ID")
        if not re.fullmatch(r"[A-Za-z0-9_.()-]+", self.resource_group):
            raise ConfigurationError("Invalid Azure resource group name")
        if not re.fullmatch(r"[A-Za-z0-9-]+", self.account_name):
            raise ConfigurationError("Invalid Foundry account name")

    @property
    def credential(self) -> Any:
        if self._credential is None:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as error:
                raise ProviderError("foundry", INSTALL_HELP) from error
            self._credential = DefaultAzureCredential()
        return self._credential

    @property
    def management(self) -> Any:
        self.require_scope()
        if self._management is None:
            try:
                from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
            except ImportError as error:
                raise ProviderError("foundry", INSTALL_HELP) from error
            self._management = CognitiveServicesManagementClient(self.credential, self.subscription_id)
            if not hasattr(self._management, "managed_compute_deployments"):
                raise ProviderError(
                    "foundry", "Installed Azure SDK lacks managed compute support", details=INSTALL_HELP
                )
        return self._management

    @property
    def monitor(self) -> Any:
        self.require_scope()
        if self._monitor is None:
            try:
                from azure.mgmt.monitor import MonitorManagementClient
            except ImportError as error:
                raise ProviderError("foundry", INSTALL_HELP) from error
            self._monitor = MonitorManagementClient(self.credential, self.subscription_id)
        return self._monitor

    @property
    def account(self) -> Any:
        if self._account is None:
            self._account = self.management.accounts.get(self.resource_group, self.account_name)
        if self._account.kind != "AIServices":
            raise ConfigurationError(
                "Foundry managed compute requires an AIServices account, not an Azure ML workspace or OpenAI-only account"
            )
        return self._account

    @property
    def endpoint(self) -> str:
        # The managed compute routing endpoint uses the account's custom subdomain.
        domain = self.account.properties.custom_sub_domain_name
        value = self.endpoint_override or (f"https://{domain}.services.ai.azure.com" if domain else None)
        if not value:
            raise ConfigurationError(
                "No Foundry inference endpoint; configure endpoint or set AZURE_FOUNDRY_ENDPOINT"
            )
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigurationError(
                "Foundry endpoint must be an HTTPS account base URL without credentials or query parameters"
            )
        if parsed.path not in ("", "/"):
            raise ConfigurationError(
                "Foundry endpoint must be the account base URL, without /openai/v1 or project paths"
            )
        return value.rstrip("/")

    def scope(self, deployment_name: Optional[str] = None) -> dict[str, str]:
        self.require_scope()
        values = {"resource_group_name": self.resource_group, "account_name": self.account_name}
        if deployment_name is not None:
            if deployment_name.startswith("/subscriptions/"):
                prefix = (
                    f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}"
                    f"/providers/Microsoft.CognitiveServices/accounts/{self.account_name}/managedComputeDeployments/"
                )
                if not deployment_name.lower().startswith(prefix.lower()):
                    raise ConfigurationError(
                        "Deployment ID belongs to a different Foundry account; select its subscription, resource group and account"
                    )
                deployment_name = deployment_name[len(prefix) :]
            if not re.fullmatch(r"[A-Za-z0-9_-]{2,64}", deployment_name):
                raise ConfigurationError(
                    "Foundry deployment names must contain 2–64 letters, digits, underscores or hyphens"
                )
            values["deployment_name"] = deployment_name
        return values

    def resource_id(self, deployment_name: str) -> str:
        deployment_name = self.scope(deployment_name)["deployment_name"]
        return (
            f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.CognitiveServices/accounts/{self.account_name}"
            f"/managedComputeDeployments/{deployment_name}"
        )
