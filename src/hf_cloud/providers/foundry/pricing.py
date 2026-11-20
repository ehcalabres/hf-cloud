"""Join live, model-specific deployment templates with shared accelerator rates."""

import json
from importlib.resources import files

from hf_cloud.core.exceptions import HFCloudError
from hf_cloud.providers.azure_catalog import AzureCatalog, PublishedModel, asset_parts
from hf_cloud.providers.foundry.profiles import ACCELERATOR_MEMORY_GIB
from hf_cloud.utils.hf_mem import InstanceData


def _context_limit(template):
    # Read machine-readable runtime settings, never infer limits from a template name.
    environment = template.environment_variables or {}
    limits = []
    for key in (
        "AZ_VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_MODEL_LEN",
        "MAX_MODEL_LEN",
        "AZ_SGLANG_CONTEXT_LENGTH",
        "SGLANG_CONTEXT_LENGTH",
    ):
        value = environment.get(key)
        if type(value) is int and value > 0:
            limits.append(value)
        elif isinstance(value, str) and value.isascii() and value.isdecimal() and int(value) > 0:
            limits.append(int(value))
    return min(limits) if limits else None


def template_catalog(catalog: AzureCatalog, model: PublishedModel) -> tuple[InstanceData, list[str]]:
    """Price only published GPU layouts; unknown rates stay unknown, never VM-derived."""
    prices = json.loads(files("hf_cloud.data").joinpath("pricing/foundry.json").read_text())
    instances, warnings, seen = [], [], set()
    for reference in model.templates:
        try:
            template = catalog.template(reference)
            concrete = f"azureml://registries/{model.registry}/deploymenttemplates/{template.name}/versions/{template.version}"
            asset_parts(concrete, "deploymenttemplates", "azure-huggingface")
            if concrete in seen:
                continue
            seen.add(concrete)
            context = _context_limit(template)
            if context is None:
                warnings.append(
                    f"Foundry template {template.name}: no machine-readable context limit; skipped."
                )
                continue
            maps = template.accelerator_maps or []
            if not maps:
                warnings.append(f"Foundry template {template.name}: no published accelerator layout; skipped.")
            families = set()
            for family in maps:
                accelerator = family.accelerator_type
                if accelerator in families:
                    continue
                families.add(accelerator)
                memory = ACCELERATOR_MEMORY_GIB.get(accelerator)
                if memory is None:
                    warnings.append(
                        f"Foundry template {template.name}: unknown accelerator memory for {accelerator}; skipped."
                    )
                    continue
                plan = catalog.foundry_template(model, reference, accelerator)
                count = plan["accelerators_per_instance"]
                rate = prices["rates"].get(accelerator)
                instances.append(
                    {
                        "instanceType": template.name,
                        "acceleratorType": accelerator,
                        "numGpu": count,
                        "gpuMemGbPerGpu": memory,
                        "pricePerHourUsd": round(rate * count, 6) if rate is not None else None,
                        "pricePerAcceleratorHourUsd": rate,
                        "maxContextLength": context,
                        "modelAssetId": model.asset_id,
                        "deploymentTemplateId": plan["deployment_template"],
                        "pricingSource": prices["pricingSource"],
                    }
                )
        except HFCloudError as error:
            warnings.append(f"Foundry template {reference}: {error}")
    return InstanceData(
        provider="foundry",
        instances=instances,
        safety_margin_default=0.1,
        last_updated=prices["last_updated"],
        pricing_region=prices["scope"],
        pricing_source=prices["pricingSource"],
    ), warnings
