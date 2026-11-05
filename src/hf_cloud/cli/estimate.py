"""Shared estimation workflow and output for cloud provider commands."""

import json
import shlex
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from hf_cloud.core.estimation import InstanceMatch, select_instances
from hf_cloud.utils.hf_mem import (
    HF_MEM_VERSION,
    EstimateOptions,
    HfMemError,
    _get_total_vram,
    load_instance_data,
    validate_safety_margin,
)

_PROVIDER_LABELS = {
    "sagemaker": ("SageMaker", "Instance Type"),
    "vertex": ("Vertex AI", "Machine Type"),
}


def _instance_output(match: InstanceMatch, provider: str) -> dict[str, Any]:
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
    }
    if provider == "vertex":
        output["acceleratorType"] = instance.get("acceleratorType")
    return output


def _print_table(console: Console, provider: str, instances: list[dict[str, Any]]) -> None:
    provider_label, instance_label = _PROVIDER_LABELS[provider]
    table = Table(title=f"Compatible {provider_label} Instances")
    table.add_column(instance_label, style="cyan")
    if provider == "vertex":
        table.add_column("Accelerator", style="green")
    for label in (
        "GPUs",
        "GPU Mem (GiB)",
        "Total VRAM (GiB)",
        "Required (GiB)",
        "Usage %",
        "VRAM Left (GiB)",
        "Price/Hour",
    ):
        table.add_column(label, justify="right")

    for instance in instances:
        row = [instance.get("instanceType") or "N/A"]
        if provider == "vertex":
            row.append(instance.get("acceleratorType") or "N/A")
        row.extend(
            [
                str(instance.get("numGpu", "N/A")),
                str(instance.get("gpuMemGbPerGpu", "N/A")),
                f"{instance['totalVramGb']:.0f}",
                f"{instance['requiredVramGb']:.2f}",
                f"{instance['usagePercent']:.1f}%",
                f"{instance['vramLeftGb']:.1f}",
                f"${instance.get('pricePerHourUsd') or 0:.2f}",
            ]
        )
        table.add_row(*row)
    console.print(table)


def _print_recommendation(
    console: Console, provider: str, model_id: str, instance: dict[str, Any], options: EstimateOptions
) -> None:
    _, instance_label = _PROVIDER_LABELS[provider]
    console.print("[bold green]Recommended Instance:[/bold green]")
    console.print(f"  [cyan]{instance_label}:[/cyan] {instance['instanceType']}")
    if provider == "vertex":
        console.print(f"  [cyan]Accelerator:[/cyan] {instance.get('acceleratorType') or 'N/A'}")
    console.print(f"  [cyan]GPUs:[/cyan] {instance['numGpu']}x {instance['gpuMemGbPerGpu']} GiB")
    console.print(f"  [cyan]Total VRAM:[/cyan] {instance['totalVramGb']:.0f} GiB")
    console.print(
        f"  [cyan]Usage:[/cyan] {instance['usagePercent']:.1f}% ({instance['vramLeftGb']:.1f} GiB left)"
    )
    console.print(f"  [cyan]Required VRAM:[/cyan] {instance['requiredVramGb']:.2f} GiB")
    console.print(f"  [cyan]GPUs needed:[/cyan] {instance['neededGpuCount']}")
    if instance["maxModelLen"]:
        console.print(f"  [cyan]Maximum context that fits:[/cyan] {instance['maxModelLen']} tokens")
    console.print(f"  [cyan]Price/Hour:[/cyan] ${instance.get('pricePerHourUsd') or 0:.2f}")
    console.print()
    # Deployment commands do not yet expose local files, revision selection or
    # alternate serving engines. Do not suggest a command that deploys different weights.
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
    instance_flag = "--machine-type" if provider == "vertex" else "--instance-type"
    command = f"hf-cloud {provider} deploy {shlex.quote(model_id)} --name <name> {instance_flag} {instance['instanceType']}"
    if provider == "vertex" and instance.get("acceleratorType"):
        command += f" --accelerator-type {instance['acceleratorType']} --accelerator-count {instance['numGpu']}"
    console.print("[dim]Deploy with:[/dim]")
    console.print(f"[dim]  {command}[/dim]")


def estimate_instances(
    model_id: str,
    *,
    provider: str,
    safety_margin: Optional[float],
    show_all: bool,
    json_output: bool,
    console: Console,
    options: Optional[EstimateOptions] = None,
) -> None:
    """Select instances using v1 GPU fit and render results with workload assumptions."""
    options = options or EstimateOptions()
    try:
        options.validate()
        instance_data = load_instance_data(provider)
        margin = safety_margin if safety_margin is not None else instance_data.safety_margin_default
        validate_safety_margin(margin)
        if not json_output:
            console.print(f"[bold blue]Estimating instance for {model_id}...[/bold blue]")
        with console.status("Checking GPU layouts with hf-mem...") if not json_output else nullcontext():
            selection = select_instances(model_id, instance_data.instances, margin, options, show_all=show_all)

        instances = [_instance_output(match, provider) for match in selection.matches]
        reference = selection.matches[0].estimate if selection.matches else selection.reference
        diagnostics = {
            "hfMemVersion": HF_MEM_VERSION,
            "memoryUnit": "GiB",
            "workload": asdict(options),
            "warnings": selection.warnings,
            "cpuMemoryGb": round(reference.cpu_memory_bytes / 1024**3, 2) if reference else 0,
            "cpuMemoryBytes": reference.cpu_memory_bytes if reference else 0,
            "estimatedContextLength": (
                (reference.raw.get("kv_cache") or {}).get("max_model_len") if reference else None
            ),
        }
        if not instances:
            message = "No compatible instances found" if show_all else "No suitable instance found"
            if json_output:
                typer.echo(
                    json.dumps(
                        {
                            "error": message,
                            "instances" if show_all else "instance": [] if show_all else None,
                            **diagnostics,
                        }
                    )
                )
            else:
                console.print(f"[bold red]{message}.[/bold red]")
                console.print("[dim]No catalog GPU layout fits this workload and safety margin.[/dim]")
                _print_diagnostics(console, diagnostics)
            raise typer.Exit(code=1)

        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "modelId": model_id,
                        "requiredVramGb": instances[0]["requiredVramGb"],
                        "safetyMargin": margin,
                        "instances" if show_all else "instance": instances if show_all else instances[0],
                        **diagnostics,
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"  [cyan]Safety margin:[/cyan] {margin * 100:g}%")
            console.print(
                f"  [cyan]Context:[/cyan] {diagnostics['estimatedContextLength'] or options.max_model_len or 'not applicable'}; "
                f"concurrent sequences: {options.batch_size}; KV dtype: {options.kv_cache_dtype}"
            )
            _print_diagnostics(console, diagnostics)
            if show_all:
                _print_table(console, provider, instances)
            else:
                _print_recommendation(console, provider, model_id, instances[0], options)
            console.print(
                "[dim]Fit assumes hf-mem's inference engine and sharding settings; "
                "configure the deployment to match this workload.[/dim]"
            )
    except HfMemError as e:
        if json_output:
            typer.echo(json.dumps({"error": e.message, "details": e.details}))
        else:
            console.print(f"[bold red]Error:[/bold red] {e.message}")
            if e.details:
                console.print(e.details, markup=False)
        raise typer.Exit(code=1) from e


def _print_diagnostics(console: Console, diagnostics: dict[str, Any]) -> None:
    if diagnostics["cpuMemoryBytes"]:
        console.print(
            f"  CPU-offloaded weights: {diagnostics['cpuMemoryGb']:.2f} GiB "
            f"({diagnostics['cpuMemoryBytes']} bytes; check host RAM separately)"
        )
    for warning in diagnostics["warnings"]:
        console.print(f"Warning: {warning}", markup=False, style="yellow")
