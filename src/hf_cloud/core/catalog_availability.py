"""Bounded, read-only publication checks shared by discovery and estimation."""

import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Optional

from hf_cloud.core.exceptions import ConfigurationError, HFCloudError, ProviderError
from hf_cloud.providers.azure_catalog import (
    AzureCatalog,
    CatalogModelUnavailable,
    PublishedModel,
    asset_parts,
    validate_hub_id,
)

MICROSOFT_PROVIDERS = ("azureml", "foundry")


@dataclass(frozen=True)
class CatalogAvailability:
    provider: str
    status: str
    reason: Optional[str] = None
    model: Optional[PublishedModel] = None
    template: Optional[dict] = None
    details: Optional[str] = None

    def to_dict(self):
        result = {"status": self.status}
        if self.reason:
            result["reason"] = self.reason
        if self.details:
            result["details"] = self.details
        if self.model:
            result["model"] = self.model.to_dict()
            registry = self.model.registry
            prefix = "nextgen/goto/discover" if self.provider == "foundry" else "explore"
            result["catalogUrl"] = (
                f"https://ai.azure.com/{prefix}/models/{self.model.name}/version/{self.model.version}/registry/{registry}"
            )
        if self.template:
            result["template"] = self.template
        return result


def _check(catalog: AzureCatalog, model_id: str, provider: str, profile: Optional[dict]) -> CatalogAvailability:
    version = None
    if provider == "foundry" and profile:
        try:
            _, _, kind, version = asset_parts(profile["modelAssetId"], "models", "azure-huggingface")
        except ConfigurationError as error:
            return CatalogAvailability(provider, "incompatible-profile", str(error))
        if kind != "versions":
            return CatalogAvailability(
                provider,
                "incompatible-profile",
                "Pin modelAssetId to a concrete catalog version to verify this Foundry profile",
            )
    try:
        model = catalog.resolve(model_id, provider, version)
    except CatalogModelUnavailable as error:
        return CatalogAvailability(provider, "unavailable", str(error))
    except HFCloudError as error:
        return CatalogAvailability(provider, "unknown", error.message, details=error.details)
    if provider == "foundry" and not model.templates:
        return CatalogAvailability(
            provider,
            "no-template",
            "Model is published but no managed-compute templates were returned for this version",
            model=model,
        )
    if provider != "foundry" or not profile:
        return CatalogAvailability(provider, "available", model=model)
    try:
        if profile["modelAssetId"].lower() != model.asset_id.lower():
            raise ConfigurationError(
                "Profile modelAssetId does not match this Hub model in the Foundry catalog"
            )
        template = catalog.foundry_template(model, profile["deploymentTemplateId"], profile["acceleratorType"])
        if profile.get("acceleratorsPerInstance") != template["accelerators_per_instance"]:
            raise ConfigurationError("Profile acceleratorsPerInstance differs from the published template")
    except ProviderError as error:
        return CatalogAvailability(provider, "unknown", error.message, model=model, details=error.details)
    except ConfigurationError as error:
        return CatalogAvailability(provider, "incompatible-profile", str(error), model=model)
    return CatalogAvailability(provider, "available", model=model, template=template)


def check_catalog_availability(
    model_id: str,
    providers=MICROSOFT_PROVIDERS,
    *,
    foundry_profile: Optional[dict] = None,
    timeout_s: float = 15,
) -> dict[str, CatalogAvailability]:
    """Check providers concurrently, excluding unknown results after a bounded wait.

    Daemon workers only read catalog metadata. SDK authentication and network calls
    cannot block command exit after the deadline. No cross-command negative cache
    is used, so login fixes and new publications are visible on the next invocation.
    """
    validate_hub_id(model_id)
    names = list(dict.fromkeys(providers))
    if any(name not in MICROSOFT_PROVIDERS for name in names):
        raise ConfigurationError("Catalog discovery supports azureml and foundry")
    if not names:
        return {}
    catalog = AzureCatalog()
    queue = Queue()

    def inspect(provider):
        try:
            result = _check(catalog, model_id, provider, foundry_profile)
        except Exception as error:
            result = CatalogAvailability(provider, "unknown", f"Catalog check failed: {error}")
        queue.put((provider, result))

    deadline = time.monotonic() + timeout_s
    for provider in names:
        Thread(target=inspect, args=(provider,), daemon=True).start()
    results = {}
    while len(results) < len(names):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            provider, result = queue.get(timeout=remaining)
        except Empty:
            break
        results[provider] = result
    return {
        provider: results.get(
            provider,
            CatalogAvailability(
                provider, "unknown", "Catalog check timed out; check az login/network access and retry"
            ),
        )
        for provider in names
    }
