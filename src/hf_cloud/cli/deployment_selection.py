"""Shared estimate-based hardware defaults for deployment and generated code."""

from dataclasses import replace
from typing import Optional

import typer

from hf_cloud.core.estimation import InstanceMatch, compare_instances, hourly_price
from hf_cloud.core.exceptions import ConfigurationError
from hf_cloud.providers.azure_catalog import AzureCatalog
from hf_cloud.providers.foundry.pricing import template_catalog
from hf_cloud.utils.hf_mem import EstimateOptions, load_instance_data


def choose_instance(
    model_id: str,
    provider: str,
    *,
    registry: Optional[AzureCatalog] = None,
    catalog_version: Optional[str] = None,
    token: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    accelerator_type: Optional[str] = None,
    accelerator_count: Optional[int] = None,
) -> InstanceMatch:
    """Use estimate's default workload, reserve and price ranking, within one provider."""
    model = registry.resolve(model_id, provider, catalog_version) if registry is not None else None
    catalog = template_catalog(registry, model)[0] if provider == "foundry" else load_instance_data(provider)
    instances = catalog.instances
    if accelerator_type is not None:
        instances = [item for item in instances if item.get("acceleratorType") == accelerator_type]
    if accelerator_count is not None:
        instances = [item for item in instances if item.get("numGpu") == accelerator_count]
    if tensor_parallel_size is not None:
        # Price the whole replica while assessing only the GPUs the server will use.
        instances = [
            dict(item, numGpu=tensor_parallel_size)
            for item in instances
            if item.get("numGpu", 0) >= tensor_parallel_size
        ]
    if model is not None and provider == "azureml":
        instances = [dict(item, modelAssetId=model.asset_id) for item in instances]
    selection = compare_instances(
        model_id,
        [replace(catalog, instances=instances)],
        0.1,
        EstimateOptions(),
        **({"token": token} if token is not None else {}),
    )
    chosen = next((match for match in selection.matches if hourly_price(match.instance) is not None), None)
    if chosen is None:
        raise ConfigurationError(
            f"No priced compatible {provider} instance found for {model_id}",
            details="Review estimate results or select hardware explicitly; no resources were created",
        )
    return chosen


def announce_selection(match: InstanceMatch) -> None:
    """Keep snippet and dry-run stdout machine readable."""
    item = match.instance
    typer.echo(
        f"Auto-selected {item['instanceType']} (${hourly_price(item):g}/instance-hour, catalog estimate).",
        err=True,
    )
    for warning in match.estimate.warnings:
        typer.echo(f"Estimate note: {warning}", err=True)
