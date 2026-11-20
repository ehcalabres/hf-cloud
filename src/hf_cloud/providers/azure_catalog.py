"""Read published Microsoft model assets; never import weights or build environments."""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional
from urllib.parse import urlsplit

from huggingface_hub.utils import HFValidationError, validate_repo_id

from hf_cloud.core.exceptions import ConfigurationError, ProviderError


class CatalogModelUnavailable(ConfigurationError):
    """A catalog lookup confirmed absence, rather than an authentication/service failure."""


LOOKUP_URL = "https://get-azure-ai-url.azurewebsites.net/api/get-azure-ai-url"
ASSET_PATTERN = re.compile(
    r"azureml://registries/([A-Za-z0-9_-]+)/(models|deploymenttemplates)/([A-Za-z0-9_.-]+)/(versions|labels)/([A-Za-z0-9_.-]+)"
)


def _is_not_found(error: Exception) -> bool:
    if getattr(error, "status_code", None) == 404:
        return True
    try:
        from azure.ai.ml.exceptions import ValidationErrorType, ValidationException
        from azure.core.exceptions import ResourceNotFoundError
    except ImportError:
        return False
    return isinstance(error, ResourceNotFoundError) or (
        isinstance(error, ValidationException) and error.error_type == ValidationErrorType.RESOURCE_NOT_FOUND
    )


def validate_hub_id(model_id: str) -> str:
    try:
        validate_repo_id(model_id)
    except HFValidationError as error:
        raise ConfigurationError(
            "Provide a Hugging Face Hub model ID (org/model)", details=str(error)
        ) from error
    if model_id.count("/") != 1:
        raise ConfigurationError("Provide a namespaced Hub model ID, such as Qwen/Qwen2.5-7B-Instruct")
    return model_id


def asset_parts(value: str, kind: str, registry: Optional[str] = None):
    match = ASSET_PATTERN.fullmatch(value) if isinstance(value, str) else None
    if not match or match[2] != kind or (registry and match[1].lower() != registry.lower()):
        raise ConfigurationError(f"Expected a published {kind} asset in the {registry or 'Azure'} registry")
    return match[1], match[3], match[4], match[5]


def lookup_foundry_model(model_id: str) -> tuple[str, str, str]:
    """Use the Hub/Microsoft documented lookup; never infer the new registry slug."""
    validate_hub_id(model_id)
    import httpx

    try:
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            response = client.get(
                LOOKUP_URL,
                params={"model_id": model_id},
                headers={"User-Agent": "hf-cloud", "Accept": "application/json"},
            )
        if response.status_code == 404:
            raise CatalogModelUnavailable(
                f"{model_id} is not published in the Foundry catalog. Check https://ai.azure.com or request its addition from the model's Hub Deploy menu."
            )
        if not response.is_success:
            raise ProviderError(
                "foundry", "Catalog lookup failed", details=f"HTTP {response.status_code}; retry later"
            )
        data = response.content
        if len(data) > 1024 * 1024:
            raise ValueError("Catalog lookup response exceeds size limit")
        payload = json.loads(data)
        parsed = urlsplit(payload["url"])
        match = re.fullmatch(
            r"/(?:explore|nextgen/goto/discover)/models/([A-Za-z0-9_.-]+)/version/([A-Za-z0-9_.-]+)/registry/([A-Za-z0-9_-]+)",
            parsed.path,
        )
        if (
            parsed.scheme != "https"
            or parsed.netloc != "ai.azure.com"
            or parsed.query
            or parsed.fragment
            or not match
        ):
            raise ValueError("Unexpected catalog URL")
        name, version, registry = match.groups()
        if registry.lower() == "huggingface":
            raise CatalogModelUnavailable(
                f"{model_id} resolves to the Azure ML / Foundry classic catalog. Use hf-cloud azureml deploy {model_id}, with an Azure ML workspace or classic hub-based project."
            )
        if registry != "azure-huggingface":
            raise ProviderError("foundry", "Catalog lookup returned an unrecognized registry")
        return registry, name, version
    except (ConfigurationError, ProviderError):
        raise
    except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as error:
        raise ProviderError(
            "foundry", "Cannot resolve the Hub ID in the Microsoft catalog", details=str(error)
        ) from error


@dataclass(frozen=True)
class PublishedModel:
    hub_id: str
    registry: str
    name: str
    version: str
    task: Optional[str] = None
    default_template: Optional[str] = None
    templates: tuple[str, ...] = ()

    @property
    def asset_id(self):
        return f"azureml://registries/{self.registry}/models/{self.name}/versions/{self.version}"

    def to_dict(self):
        return {**asdict(self), "asset_id": self.asset_id}


class AzureCatalog:
    def __init__(self, credential=None):
        self._credential = credential
        self._clients = {}
        self._models = {}
        self._templates = {}

    def registry(self, name: str):
        if name not in self._clients:
            try:
                from azure.ai.ml import MLClient
                from azure.identity import DefaultAzureCredential
            except ImportError as error:
                raise ProviderError("azure", "Run uv sync to install the Microsoft catalog SDK") from error
            self._credential = self._credential or DefaultAzureCredential()
            self._clients[name] = MLClient(self._credential, registry_name=name)
        return self._clients[name]

    def resolve(self, hub_id: str, backend: str, version: Optional[str] = None) -> PublishedModel:
        validate_hub_id(hub_id)
        cache_key = (hub_id, backend, version)
        if cache_key in self._models:
            return self._models[cache_key]
        if version and not re.fullmatch(r"[A-Za-z0-9_.-]+", version):
            raise ConfigurationError("Invalid catalog version")
        if backend == "azureml":
            registry = "HuggingFace"
            # Published mapping documented by the Hugging Face Azure ML integration.
            name = hub_id.replace("/", "-").replace("_", "-").lower()
        elif backend == "foundry":
            registry, name, published_version = lookup_foundry_model(hub_id)
            version = version or published_version
        else:
            raise ConfigurationError("Select the azureml or foundry catalog")
        try:
            model = self.registry(registry).models.get(
                name=name, **({"version": version} if version else {"label": "latest"})
            )
        except Exception as error:
            if isinstance(error, (ConfigurationError, ProviderError)):
                raise
            if _is_not_found(error):
                alternative = "foundry" if backend == "azureml" else "azureml"
                raise CatalogModelUnavailable(
                    f"{hub_id} is not published in the {backend} catalog"
                    + (f" at version {version}" if version else "")
                    + f". Check the model catalog or try hf-cloud {alternative} catalog {hub_id}. HF Cloud only deploys published assets; it does not build custom images."
                ) from error
            raise ProviderError(
                backend,
                "Cannot read the published model catalog; check az login and registry access",
                details=str(error),
            ) from error
        metadata = {**(model.tags or {}), **(model.properties or {})}
        for key in ("huggingface_model_id", "hf_model_id", "model_id"):
            upstream = metadata.get(key)
            if (
                isinstance(upstream, str)
                and upstream.count("/") == 1
                and ":" not in upstream
                and upstream.lower() != hub_id.lower()
            ):
                raise ConfigurationError("Published model metadata does not match the requested Hub ID")
        if not model.name or not model.version:
            raise ConfigurationError("Published model is missing its name or version")
        resolved_version = str(model.version)
        asset_parts(
            f"azureml://registries/{registry}/models/{model.name}/versions/{resolved_version}",
            "models",
            registry,
        )
        task = next(
            (
                metadata[key]
                for key in ("task", "pipeline_tag", "pipelineTag")
                if isinstance(metadata.get(key), str) and metadata[key]
            ),
            None,
        )
        default = getattr(getattr(model, "default_deployment_template", None), "asset_id", None)
        templates = tuple(
            dict.fromkeys(
                ([default] if default else [])
                + [
                    item.asset_id
                    for item in (getattr(model, "allowed_deployment_templates", None) or [])
                    if item.asset_id
                ]
            )
        )
        published = PublishedModel(hub_id, registry, model.name, resolved_version, task, default, templates)
        self._models[cache_key] = published
        self._models[(hub_id, backend, resolved_version)] = published
        return published

    def template(self, reference: str):
        """Resolve and cache an active published template, including label-to-version resolution."""
        if reference in self._templates:
            return self._templates[reference]
        registry, name, kind, identifier = asset_parts(reference, "deploymenttemplates", "azure-huggingface")
        try:
            template = self.registry(registry).deployment_templates.get(
                name=name, **({"version": identifier} if kind == "versions" else {"label": identifier})
            )
        except Exception as error:
            if _is_not_found(error):
                raise ConfigurationError("The selected published template no longer exists") from error
            raise ProviderError(
                "foundry", "Cannot read the published deployment template", details=str(error)
            ) from error
        if not template.name or not template.version:
            raise ConfigurationError("Published template is missing its name or version")
        if getattr(template, "stage", None) == "Archived":
            raise ConfigurationError("The published template is archived; select an active template")
        asset_parts(
            f"azureml://registries/{registry}/deploymenttemplates/{template.name}/versions/{template.version}",
            "deploymenttemplates",
            registry,
        )
        self._templates[reference] = template
        return template

    def foundry_template(
        self, model: PublishedModel, selected: Optional[str] = None, accelerator: Optional[str] = None
    ) -> dict[str, Any]:
        if model.registry != "azure-huggingface":
            raise ConfigurationError("Foundry managed compute requires an azure-huggingface model")
        candidates = list(model.templates)
        if selected:
            if selected.startswith("azureml://"):
                if selected not in candidates:
                    _, name, _, _ = asset_parts(selected, "deploymenttemplates", model.registry)
                    matching = [
                        uri
                        for uri in candidates
                        if asset_parts(uri, "deploymenttemplates", model.registry)[1] == name
                    ]
                    if not matching:
                        raise ConfigurationError(
                            "The selected template is not published for this model version; run foundry catalog MODEL_ID"
                        )
                    selected_template = self.template(selected)
                    if not any(
                        str(self.template(uri).version) == str(selected_template.version) for uri in matching
                    ):
                        raise ConfigurationError(
                            "The selected template version is not published for this model version"
                        )
                target = selected
            else:
                matching = [
                    uri
                    for uri in candidates
                    if asset_parts(uri, "deploymenttemplates", model.registry)[1] == selected
                ]
                if len(matching) != 1:
                    raise ConfigurationError("Select a template shown by hf-cloud foundry catalog MODEL_ID")
                target = matching[0]
        else:
            target = model.default_template or (candidates[0] if len(candidates) == 1 else None)
        if not target:
            raise ConfigurationError(
                "This model has no unambiguous published default template. Run hf-cloud foundry catalog MODEL_ID and select --template NAME."
            )
        registry, name, kind, identifier = asset_parts(target, "deploymenttemplates", model.registry)
        template = self.template(target)
        maps = template.accelerator_maps or []
        eligible = (
            [item for item in maps if item.accelerator_type == accelerator]
            if accelerator
            else [item for item in maps if item.default]
        )
        if not accelerator and not eligible and len(maps) == 1:
            eligible = maps
        if len(eligible) != 1:
            choices = ", ".join(item.accelerator_type for item in maps) or "none published"
            raise ConfigurationError(
                f"Select --accelerator-type from this template's supported families: {choices}"
            )
        family = eligible[0]
        if (
            type(family.number_of_accelerators_per_model_instance) is not int
            or family.number_of_accelerators_per_model_instance < 1
        ):
            raise ConfigurationError("Published template has no valid accelerator count")
        return {
            "model_asset_id": model.asset_id,
            "deployment_template": f"azureml://registries/{registry}/deploymenttemplates/{template.name}/versions/{template.version}",
            "accelerator_type": family.accelerator_type,
            "accelerators_per_instance": family.number_of_accelerators_per_model_instance,
        }
