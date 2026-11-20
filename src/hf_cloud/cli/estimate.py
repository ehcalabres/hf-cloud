"""Canonical estimate command and provider-filtered compatibility aliases."""

import json
import math
import shlex
from contextlib import nullcontext
from dataclasses import asdict
from enum import Enum
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Annotated, Any, Callable, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hf_cloud.cli.provider_availability import print_provider_availability
from hf_cloud.core.catalog_availability import (
    MICROSOFT_PROVIDERS,
    CatalogAvailability,
    check_catalog_availability,
)
from hf_cloud.core.estimation import InstanceMatch, compare_instances, compute_cost, hourly_price
from hf_cloud.core.exceptions import ConfigurationError
from hf_cloud.core.provider_availability import provider_availability
from hf_cloud.providers.azure_catalog import validate_hub_id
from hf_cloud.providers.foundry.profiles import load_profile, profile_catalog
from hf_cloud.utils.hf_mem import (
    HF_MEM_VERSION,
    EstimateOptions,
    HfMemError,
    _get_total_vram,
    list_instance_providers,
    load_instance_data,
    validate_safety_margin,
)

console = Console()


def _instance_output(match: InstanceMatch) -> dict[str, Any]:
    instance, estimate = match.instance, match.estimate
    fit = estimate.fit
    assert fit is not None  # select_instances only returns validated GPU assessments.
    required_vram = fit.required_bytes / 1024**3
    total_vram = _get_total_vram(instance)
    output = {
        "instanceType": instance.get("instanceType"),
        "numGpu": instance.get("numGpu"),
        "gpuMemGbPerGpu": instance.get("gpuMemGbPerGpu"),
        "totalVramGb": total_vram,
        "requiredVramGb": round(required_vram, 2),
        "usagePercent": round(required_vram / total_vram * 100, 1),
        "vramLeftGb": round(total_vram - required_vram, 1),
        "pricePerHourUsd": instance.get("pricePerHourUsd"),
        "neededGpuCount": fit.needed_gpu_count,
        "maxModelLen": fit.max_model_len,
        "cpuMemoryGb": round(estimate.cpu_memory_bytes / 1024**3, 2),
        "cpuMemoryBytes": estimate.cpu_memory_bytes,
        "warnings": list(estimate.warnings),
        "weightsFormat": estimate.raw.get("format"),
        "acceleratorType": instance.get("acceleratorType"),
    }
    if instance.get("provider") == "foundry":
        output.update(
            {
                key: instance.get(key)
                for key in (
                    "modelAssetId",
                    "deploymentTemplateId",
                    "profilePath",
                    "pricingSource",
                    "pricePerAcceleratorHourUsd",
                    "maxContextLength",
                )
            }
        )
    return output


def _price(value: Optional[float]) -> str:
    return f"${value:,.2f}" if value is not None else "—"


def _accelerator_label(instance: dict[str, Any]) -> str:
    name = instance.get("acceleratorType") or ""
    return (
        name.removeprefix("NVIDIA_")
        .removeprefix("TESLA_")
        .removesuffix(f"_{instance['gpuMemGbPerGpu']:g}GB")
        .replace("_", " ")
    )


def _print_workload(
    console: Console, diagnostics: dict[str, Any], options: EstimateOptions, margin: float
) -> None:
    context = diagnostics["estimatedContextLength"] or options.max_model_len
    context_label = f"{context:,} tokens" if context else "not applicable"
    console.print(
        Text(
            f"Context {context_label} · Batch {options.batch_size} · KV {options.kv_cache_dtype} · Reserve {margin * 100:g}%",
            style="dim",
        )
    )
    console.print()


def _print_diagnostics(console: Console, diagnostics: dict[str, Any]) -> None:
    if diagnostics["cpuMemoryBytes"]:
        console.print(
            Text(
                f"\nCPU-offloaded weights: {diagnostics['cpuMemoryGb']:,.2f} GiB · additional host RAM required.",
                style="yellow",
            )
        )
    if diagnostics["warnings"]:
        console.print("\nNotes", style="bold yellow")
        notes = Table.grid(padding=(0, 1))
        notes.add_column(width=1, style="yellow")
        notes.add_column()
        for warning in diagnostics["warnings"]:
            notes.add_row("•", Text(warning, style="dim"))
        console.print(notes)


class PriceSort(str, Enum):
    CHEAPEST = "cheapest"
    MOST_EXPENSIVE = "most-expensive"


def _print_cheapest(cheapest: Optional[dict], hours: float = 1, instance_count: int = 1) -> None:
    if cheapest is None:
        console.print("No catalog prices available for compatible instances.", style="yellow")
        return
    details = Text(
        f"{_price(cheapest['deploymentPricePerHourUsd'])}/h · {cheapest['provider']} · {cheapest['instanceType']}"
    )
    if hours != 1 or instance_count != 1:
        details.append(
            f"\n{_price(cheapest['estimatedCostUsd'])} for {hours:g} hours × {instance_count} instance(s)"
        )
    console.print(
        Panel(details, title="Cheapest compatible instance", title_align="left", border_style="green")
    )


def _print_comparison(rows: list[dict[str, Any]], total: int, order: PriceSort, hours: float = 1) -> None:
    table = Table(box=box.SIMPLE_HEAD, header_style="bold", padding=(0, 1))
    table.add_column("Provider", no_wrap=True)
    table.add_column("Instance", style="cyan", no_wrap=True)
    table.add_column("GPUs", no_wrap=True)
    wide = console.width >= 105
    if wide:
        table.add_column("VRAM used / total", justify="right", no_wrap=True)
    table.add_column("Est. $/h" if hours == 1 else f"Est. {hours:g}h", justify="right", no_wrap=True)
    for row in rows:
        values = [
            row["provider"],
            row["instanceType"],
            f"{row['numGpu']} × {row['gpuMemGbPerGpu']:g} {_accelerator_label(row)}",
        ]
        if wide:
            values.append(f"{row['requiredVramGb']:,.2f} / {row['totalVramGb']:g}")
        values.append(_price(row["deploymentPricePerHourUsd"] if hours == 1 else row["estimatedCostUsd"]))
        table.add_row(*values)
    console.print(table)
    console.print(
        f"{len(rows)} of {total} compatible instances · {order.value} first · memory in GiB", style="dim"
    )


def _print_deploy_command(
    model_id: str, instance: dict[str, Any], options: EstimateOptions, instance_count: int = 1
) -> None:
    # Only print executable hints for supported Hub weight selections.
    if (
        instance["weightsFormat"] != "safetensors"
        or options.revision
        or options.file
        or "@" in model_id
        or ":" in model_id
        or model_id.count("/") != 1
        or Path(model_id).exists()
    ):
        return
    provider = instance["provider"]
    if provider not in {"sagemaker", "vertex", "azureml", "foundry"}:
        return
    parts = [
        shlex.join(["hf-cloud", provider, "deploy", model_id]),
        "--name my-endpoint",
    ]
    if provider == "foundry":
        if instance.get("profilePath"):
            parts.append(shlex.join(["--profile", instance["profilePath"]]))
        else:
            parts.extend([
                shlex.join(["--model-asset-id", instance["modelAssetId"]]),
                shlex.join(["--template", instance["deploymentTemplateId"]]),
                shlex.join(["--accelerator-type", instance["acceleratorType"]]),
            ])
    else:
        instance_flag = "--machine-type" if provider == "vertex" else "--instance-type"
        parts.append(shlex.join([instance_flag, instance["instanceType"]]))
    if provider == "vertex" and instance.get("acceleratorType"):
        parts.extend(
            [
                shlex.join(["--accelerator-type", instance["acceleratorType"]]),
                shlex.join(["--accelerator-count", str(instance["numGpu"])]),
            ]
        )
    if provider == "azureml" and instance.get("catalogVersion"):
        parts.append(shlex.join(["--catalog-version", instance["catalogVersion"]]))
    if instance_count != 1:
        if provider == "vertex":
            parts.extend([f"--min-replicas {instance_count}", f"--max-replicas {instance_count}"])
        else:
            parts.append(f"--instance-count {instance_count}")
    command = " ".join(parts)
    if len(command) > console.width:
        command = " \\\n  ".join(parts)
    console.print("\nDeploy cheapest instance", style="bold")
    console.print(Text(command, style="cyan"), soft_wrap=True)
    note = (
        "The published template controls the runtime and context settings."
        if provider == "foundry"
        else "Catalog model availability verified; Azure ML uses the published runtime, whose settings may differ from this estimate."
        if provider == "azureml"
        else "Match the deployment engine, sharding and workload to this estimate."
    )
    console.print(note, style="dim")


def estimate(
    model_id: Annotated[str, typer.Argument(help="Namespaced Hub model ID, URL, or local path")],
    providers: Annotated[
        Optional[list[str]],
        typer.Option(
            "--provider",
            "-p",
            help="Provider to include; repeat to compare selected providers",
            rich_help_panel="Comparison",
        ),
    ] = None,
    foundry_profile: Annotated[
        Optional[Path],
        typer.Option(
            "--foundry-profile",
            "--profile",
            help="Optional override for Foundry template selection and pricing",
            rich_help_panel="Comparison",
        ),
    ] = None,
    sort: Annotated[
        PriceSort,
        typer.Option("--sort", help="Order compatible instances by hourly price", rich_help_panel="Comparison"),
    ] = PriceSort.CHEAPEST,
    limit: Annotated[
        Optional[int],
        typer.Option(
            "--limit",
            min=1,
            help="Maximum rows to show; cheapest option is selected from all matches",
            rich_help_panel="Comparison",
        ),
    ] = None,
    safety_margin: Annotated[
        float,
        typer.Option(
            "--safety-margin",
            "-s",
            help="Fraction of GPU memory to reserve across all providers",
            rich_help_panel="Comparison",
        ),
    ] = 0.1,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output comparison as JSON", rich_help_panel="Output")
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Show deployment-check details and price sources", rich_help_panel="Output"
        ),
    ] = False,
    hours: Annotated[
        float,
        typer.Option(
            "--hours",
            help="Continuous running hours to project compute costs",
            rich_help_panel="Cost projection",
        ),
    ] = 1,
    instance_count: Annotated[
        int,
        typer.Option(
            "--instance-count",
            "-c",
            min=1,
            help="Replicas for cost projection; does not change GPUs within an instance",
            rich_help_panel="Cost projection",
        ),
    ] = 1,
    show_all: Annotated[
        bool,
        typer.Option(
            "--all", "-a", hidden=True, help="Compatibility flag; all matching instances are listed by default"
        ),
    ] = False,
    revision: Annotated[
        Optional[str],
        typer.Option("--revision", help="Hub branch, tag, or commit", rich_help_panel="Model selection"),
    ] = None,
    file: Annotated[
        Optional[str],
        typer.Option(
            "--file", help="Weight file, directory, or GGUF quantization tag", rich_help_panel="Model selection"
        ),
    ] = None,
    format: Annotated[
        Optional[str],
        typer.Option("--format", help="Weights: safetensors, gguf, or onnx", rich_help_panel="Model selection"),
    ] = None,
    max_model_len: Annotated[
        Optional[int],
        typer.Option("--max-model-len", help="Context tokens per sequence", rich_help_panel="Workload"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Concurrent sequences in the KV cache", rich_help_panel="Workload"),
    ] = 1,
    kv_cache_dtype: Annotated[
        str,
        typer.Option("--kv-cache-dtype", help="KV cache dtype (e.g., auto, fp8)", rich_help_panel="Workload"),
    ] = "auto",
    max_num_batched_tokens: Annotated[
        Optional[int],
        typer.Option(
            "--max-num-batched-tokens",
            help="Engine tokens per batch; use 2048 for A100",
            rich_help_panel="Workload",
        ),
    ] = None,
) -> None:
    """Compare compatible instances across providers using bundled hourly price estimates."""
    try:
        if not math.isfinite(hours) or hours <= 0:
            raise HfMemError("--hours must be finite and greater than zero")
        options = EstimateOptions(
            revision=revision,
            file=file,
            format=format,
            max_model_len=max_model_len,
            batch_size=batch_size,
            kv_cache_dtype=kv_cache_dtype,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        options.validate()
        validate_safety_margin(safety_margin)
        available = list(dict.fromkeys([*list_instance_providers(), "foundry"]))
        selected = list(dict.fromkeys(name.lower() for name in providers)) if providers else available
        unsupported = [name for name in selected if name not in available]
        if unsupported or not selected:
            raise HfMemError(
                "No estimate catalog for requested providers",
                details=f"Unsupported: {', '.join(unsupported) or 'none available'}. Available: {', '.join(available)}",
            )
        microsoft = [provider for provider in selected if provider in MICROSOFT_PROVIDERS]
        profile = load_profile(foundry_profile, model_id) if foundry_profile and "foundry" in selected else None
        checks = {}
        excluded = {}
        selection_reason = None
        if microsoft:
            try:
                validate_hub_id(model_id)
            except ConfigurationError:
                selection_reason = "Published Microsoft catalog deployments require a namespaced Hub model ID; local paths and URLs cannot be deployed there"
            if (
                (not selection_reason and Path(model_id).exists())
                or options.revision
                or options.file
                or options.format not in (None, "safetensors")
            ):
                selection_reason = "Published Microsoft catalog models cannot follow --revision, --file, local paths, or non-Safetensors weight selection"
            if selection_reason:
                checks = {
                    provider: CatalogAvailability(provider, "unsupported-selection", selection_reason)
                    for provider in microsoft
                }
            else:
                with (
                    console.status("Checking Microsoft model catalogs...") if not json_output else nullcontext()
                ):
                    checks = check_catalog_availability(
                        model_id, microsoft, foundry_profile=profile,
                        **({"include_estimates": True} if "foundry" in microsoft and profile is None else {}),
                    )
            excluded.update(
                {
                    provider: result.reason or result.status
                    for provider, result in checks.items()
                    if result.status != "available"
                }
            )
            if "foundry" in microsoft and "foundry" not in excluded and profile is None:
                discovered = checks["foundry"].estimate_catalog
                if discovered is None or not discovered.instances:
                    excluded["foundry"] = (
                        "No verified Foundry template layouts and context limits available for estimates. "
                        "Use --verbose for details, or supply an explicit --foundry-profile."
                    )
        catalogs = [
            (profile_catalog(foundry_profile, model_id) if profile else checks["foundry"].estimate_catalog)
            if provider == "foundry"
            else load_instance_data(provider)
            for provider in selected
            if provider not in excluded
        ]
        if not json_output:
            console.print(
                "Instance estimate" if len(selected) == 1 else "Cross-provider estimate", style="bold"
            )
            console.print(model_id, style="cyan", markup=False)
        with console.status("Comparing compatible GPU layouts...") if not json_output else nullcontext():
            selection = compare_instances(
                model_id, catalogs, safety_margin, options, descending=sort == PriceSort.MOST_EXPENSIVE
            )
        rows = [
            {
                **_instance_output(match),
                "provider": match.instance["provider"],
                "pricePerHourUsd": hourly_price(match.instance),
            }
            for match in selection.matches
        ]
        for row in rows:
            row.update(compute_cost(row, hours, instance_count))
            publication = checks.get(row["provider"])
            if publication and publication.model:
                row["modelAssetId"] = publication.model.asset_id
                row["catalogVersion"] = publication.model.version
                if publication.template:
                    row["deploymentTemplateId"] = publication.template["deployment_template"]
        priced = [row for row in rows if row["pricePerHourUsd"] is not None]
        cheapest = min(priced, key=lambda row: row["pricePerHourUsd"]) if priced else None
        most_expensive = max(priced, key=lambda row: row["pricePerHourUsd"]) if priced else None
        reference = selection.reference
        availability = provider_availability(
            model_id, selected, options, checks, rows, excluded,
            reference.raw.get("format") if reference else None,
        )
        diagnostics = {
            "hfMemVersion": HF_MEM_VERSION,
            "memoryUnit": "GiB",
            "workload": asdict(options),
            "warnings": selection.warnings,
            "cpuMemoryGb": round(reference.cpu_memory_bytes / 1024**3, 2) if reference else 0,
            "cpuMemoryBytes": reference.cpu_memory_bytes if reference else 0,
            "estimatedContextLength": (reference.raw.get("kv_cache") or {}).get("max_model_len")
            if reference
            else None,
        }
        output = {
            "modelId": model_id,
            "providers": selected,
            "sort": sort.value,
            "safetyMargin": safety_margin,
            "costEstimate": {
                "hours": hours,
                "instanceCount": instance_count,
                "basis": "compute-only",
                "replicas": "constant",
            },
            "pricing": {
                "source": "catalogs-and-profiles"
                if profile is not None and any(catalog.provider == "foundry" for catalog in catalogs)
                else "bundled-catalogs",
                "currency": "USD",
                "unit": "hour",
                "catalogs": {
                    catalog.provider: {
                        "lastUpdated": catalog.last_updated,
                        "region": catalog.pricing_region,
                        **({"source": catalog.pricing_source} if catalog.pricing_source else {}),
                    }
                    for catalog in catalogs
                },
            },
            "totalCompatible": len(rows),
            "pricedCount": len(priced),
            "unpricedCount": len(rows) - len(priced),
            "cheapest": cheapest,
            "mostExpensive": most_expensive,
            "instances": rows[:limit],
            "providerAvailability": availability,
            **diagnostics,
        }
        if checks:
            output["catalogAvailability"] = {provider: result.to_dict() for provider, result in checks.items()}
        if excluded:
            output["excludedProviders"] = excluded
        if not catalogs and excluded:
            output["error"] = "; ".join(f"{provider}: {reason}" for provider, reason in excluded.items())
        elif not rows:
            output["error"] = "No compatible instances found"
        elif not priced:
            output["error"] = "No catalog prices available for compatible instances"
        if json_output:
            typer.echo(json.dumps(output, indent=2))
        else:
            _print_workload(console, diagnostics, options, safety_margin)
            if rows:
                _print_cheapest(cheapest, hours, instance_count)
                _print_comparison(output["instances"], len(rows), sort, hours)
            else:
                console.print("No compatible instances found.", style="yellow")
            if hours != 1 or instance_count != 1:
                console.print(
                    f"Compute projection: {hours:g} running hours × {instance_count} instance(s), held constant.",
                    style="dim",
                )
            console.print()
            print_provider_availability(availability, checks, console, verbose=verbose)
            if checks:
                if checks.get("foundry") and checks["foundry"].status == "available" and "foundry" in excluded:
                    console.print(
                        "Foundry: no verified template layouts; use --verbose or supply --foundry-profile FILE.",
                        style="dim",
                    )
                elif checks.get("foundry") and checks["foundry"].estimate_warnings and not verbose:
                    console.print("Some Foundry templates were skipped; use --verbose for details.", style="dim")
                if not verbose and any(result.status == "unknown" for result in checks.values()):
                    console.print(
                        "Some catalog checks failed. Run az login and retry; use --verbose for details.",
                        style="dim",
                    )
            if verbose:
                for catalog in catalogs:
                    console.print(
                        Text(
                            f"{catalog.provider} prices: {catalog.last_updated or 'date unknown'} · {catalog.pricing_region or 'region unspecified'}",
                            style="dim",
                        )
                    )
                    if catalog.pricing_source:
                        console.print(Text(f"  Source: {catalog.pricing_source}", style="dim"))
            dates = "; ".join(f"{catalog.provider} {catalog.last_updated or 'undated'}" for catalog in catalogs)
            console.print(Text(f"Pricing: compute-only USD estimates; not live quotes. {dates}", style="dim"))
            console.print("Storage, network, autoscaling and other service charges are excluded.", style="dim")
            if output["unpricedCount"]:
                console.print(
                    f"{output['unpricedCount']} compatible instances have no usable price; excluded from the price range.",
                    style="yellow",
                )
            _print_diagnostics(console, diagnostics)
            if cheapest is not None:
                _print_deploy_command(model_id, cheapest, options, instance_count)
        if "error" in output:
            raise typer.Exit(code=1)
    except (HfMemError, ConfigurationError) as error:
        if json_output:
            typer.echo(json.dumps({"error": error.message, "details": error.details}))
        else:
            console.print(error.message, markup=False, style="bold red")
            if error.details:
                console.print(error.details, markup=False)
        raise typer.Exit(code=1) from error


def provider_estimate(provider: str) -> Callable[..., None]:
    """Bind the canonical command to one provider while sharing its option definitions."""
    command_signature = signature(estimate)
    alias_signature = command_signature.replace(
        parameters=[
            parameter for name, parameter in command_signature.parameters.items() if name != "providers"
        ]
    )

    @wraps(estimate)
    def alias(*args: Any, **kwargs: Any) -> None:
        arguments = alias_signature.bind(*args, **kwargs).arguments
        estimate(**arguments, providers=[provider])

    # Typer uses this signature for parsing and help, so options cannot drift.
    alias.__signature__ = alias_signature
    alias.__doc__ = f"Shortcut for hf-cloud estimate --provider {provider}."
    return alias
