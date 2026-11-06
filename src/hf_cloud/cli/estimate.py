"""Shared estimation workflow and output for cloud provider commands."""

import json
import shlex
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import typer
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

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


def _price(value: Optional[float]) -> str:
    return f"${value:.2f}" if value is not None else "—"


def _accelerator_label(instance: dict[str, Any]) -> str:
    name = instance.get("acceleratorType") or ""
    return (
        name.removeprefix("NVIDIA_")
        .removeprefix("TESLA_")
        .removesuffix(f"_{instance['gpuMemGbPerGpu']:g}GB")
        .replace("_", " ")
    )


def _print_table(console: Console, provider: str, instances: list[dict[str, Any]]) -> None:
    provider_label, _ = _PROVIDER_LABELS[provider]
    wide = console.width >= 105
    show_accelerator = provider == "vertex" and console.width >= 120
    table = Table(
        title=f"Compatible {provider_label} Instances", box=box.SIMPLE_HEAD, header_style="bold", padding=(0, 1)
    )
    table.add_column("Instance", style="cyan", no_wrap=True)
    if show_accelerator:
        table.add_column("Accelerator")
    table.add_column("GPUs", no_wrap=True)
    table.add_column("VRAM used / total", justify="right", no_wrap=True)
    if wide:
        table.add_column("Free", justify="right", no_wrap=True)
        table.add_column("Usage", justify="right", no_wrap=True)
    table.add_column("Est. $/h", justify="right", no_wrap=True)

    for index, instance in enumerate(instances):
        row = [instance.get("instanceType") or "N/A"]
        if show_accelerator:
            row.append(_accelerator_label(instance) or "N/A")
        layout = f"{instance['numGpu']} × {instance['gpuMemGbPerGpu']:g}"
        if provider == "vertex" and not show_accelerator:
            layout += f" {_accelerator_label(instance)}"
        row.extend(
            [
                layout,
                f"{instance['requiredVramGb']:,.2f} / {instance['totalVramGb']:g}",
            ]
        )
        if wide:
            row.extend([f"{instance['vramLeftGb']:,.1f}", f"{instance['usagePercent']:.1f}%"])
        row.append(_price(instance.get("pricePerHourUsd")))
        table.add_row(*row, style="bold" if index == 0 else None)
    console.print(table)
    console.print(
        f"{len(instances)} compatible instances · memory in GiB · first row is the minimum memory fit.",
        style="dim",
    )


def _print_recommendation(console: Console, provider: str, instance: dict[str, Any]) -> None:
    details = Table.grid(padding=(0, 2))
    details.add_column(style="dim")
    details.add_column()
    layout = f"{instance['numGpu']} × {instance['gpuMemGbPerGpu']:g} GiB"
    if provider == "vertex" and instance.get("acceleratorType"):
        layout += f" · {_accelerator_label(instance)}"
    details.add_row("GPUs", Text(layout))
    details.add_row(
        "GPU memory",
        f"{instance['requiredVramGb']:,.2f} / {instance['totalVramGb']:g} GiB · {instance['usagePercent']:.1f}% used",
    )
    if console.width >= 60:
        details.add_row(
            "", ProgressBar(total=100, completed=instance["usagePercent"], width=min(32, console.width - 25))
        )
    details.add_row("Free GPU memory", f"{instance['vramLeftGb']:,.1f} GiB")
    if instance["neededGpuCount"] != instance["numGpu"]:
        details.add_row("GPUs needed", str(instance["neededGpuCount"]))
    if instance["maxModelLen"]:
        details.add_row("Maximum context", f"{instance['maxModelLen']:,} tokens")
    details.add_row(
        "Estimated price",
        f"{_price(instance.get('pricePerHourUsd'))} / hour"
        if instance.get("pricePerHourUsd") is not None
        else "Not available",
    )
    console.print(
        Panel(
            Group(Text(instance["instanceType"], style="bold cyan"), Text(""), details),
            title="Recommended Instance",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )
    )


def _print_deploy_command(
    console: Console, provider: str, model_id: str, instance: dict[str, Any], options: EstimateOptions
) -> None:
    # These commands do not expose local files, revisions, or alternate engines.
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
    parts = [
        f"hf-cloud {provider} deploy {shlex.quote(model_id)}",
        "--name <name>",
        f"{instance_flag} {instance['instanceType']}",
    ]
    if provider == "vertex" and instance.get("acceleratorType"):
        parts.extend(
            [f"--accelerator-type {instance['acceleratorType']}", f"--accelerator-count {instance['numGpu']}"]
        )
    command = " ".join(parts)
    if len(command) > console.width:
        command = " \\\n  ".join(parts)
    console.print("\nDeploy with", style="bold")
    # Explicit continuations keep narrow-terminal commands ready to copy.
    console.print(Text(command, style="dim"), soft_wrap=True)


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
            provider_label, _ = _PROVIDER_LABELS[provider]
            console.print(Text(f"{provider_label} estimate", style="bold"))
            console.print(model_id, style="cyan", markup=False)
        with console.status("Estimating model memory...") if not json_output else nullcontext():
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
            _print_workload(console, diagnostics, options, margin)
            if show_all:
                _print_table(console, provider, instances)
            else:
                _print_recommendation(console, provider, instances[0])
            _print_diagnostics(console, diagnostics)
            if not show_all:
                _print_deploy_command(console, provider, model_id, instances[0], options)
            console.print("\nMatch the deployment engine, sharding and workload to this estimate.", style="dim")
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
