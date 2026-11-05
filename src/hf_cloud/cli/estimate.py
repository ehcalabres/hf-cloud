"""Shared estimation workflow and output for cloud provider commands."""

import json
from contextlib import nullcontext
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from hf_cloud.utils.hf_mem import (
    HfMemError,
    InstanceInfo,
    _get_total_vram,
    find_compatible_instances,
    find_minimum_instance,
    load_instance_data,
    run_hf_mem,
    validate_safety_margin,
)

_PROVIDER_LABELS = {
    "sagemaker": ("SageMaker", "Instance Type"),
    "vertex": ("Vertex AI", "Machine Type"),
}


def _instance_output(instance: InstanceInfo, required_vram: float, provider: str) -> dict[str, Any]:
    total_vram = _get_total_vram(instance)
    output = {
        "instanceType": instance.get("instanceType"),
        "numGpu": instance.get("numGpu"),
        "gpuMemGbPerGpu": instance.get("gpuMemGbPerGpu"),
        "totalVramGb": total_vram,
        "usagePercent": round(required_vram / total_vram * 100, 1) if total_vram else 0,
        "vramLeftGb": round(total_vram - required_vram, 1),
        "pricePerHourUsd": instance.get("pricePerHourUsd"),
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
    for label in ("GPUs", "GPU Mem (GB)", "Total VRAM (GB)", "Usage %", "VRAM Left (GB)", "Price/Hour"):
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
                f"{instance['usagePercent']:.1f}%",
                f"{instance['vramLeftGb']:.1f}",
                f"${instance.get('pricePerHourUsd') or 0:.2f}",
            ]
        )
        table.add_row(*row)
    console.print(table)


def _print_recommendation(console: Console, provider: str, model_id: str, instance: dict[str, Any]) -> None:
    _, instance_label = _PROVIDER_LABELS[provider]
    console.print("[bold green]Recommended Instance:[/bold green]")
    console.print(f"  [cyan]{instance_label}:[/cyan] {instance['instanceType']}")
    if provider == "vertex":
        console.print(f"  [cyan]Accelerator:[/cyan] {instance.get('acceleratorType') or 'N/A'}")
    console.print(f"  [cyan]GPUs:[/cyan] {instance['numGpu']}x {instance['gpuMemGbPerGpu']} GB")
    console.print(f"  [cyan]Total VRAM:[/cyan] {instance['totalVramGb']:.0f} GB")
    console.print(
        f"  [cyan]Usage:[/cyan] {instance['usagePercent']:.1f}% ({instance['vramLeftGb']:.1f} GB left)"
    )
    console.print(f"  [cyan]Price/Hour:[/cyan] ${instance.get('pricePerHourUsd') or 0:.2f}")
    console.print()
    instance_flag = "--machine-type" if provider == "vertex" else "--instance-type"
    command = f"hf-cloud {provider} deploy {model_id} --name <name> {instance_flag} {instance['instanceType']}"
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
) -> None:
    """Estimate once, select compatible instances, and render provider-specific output."""
    try:
        instance_data = load_instance_data(provider)
        margin = safety_margin if safety_margin is not None else instance_data.safety_margin_default
        # Fail before starting an expensive subprocess when the margin is invalid.
        validate_safety_margin(margin)
        if not json_output:
            console.print(f"[bold blue]Estimating instance for {model_id}...[/bold blue]")
        with console.status("Running hf-mem estimation...") if not json_output else nullcontext():
            estimate = run_hf_mem(model_id)
        required_vram = estimate.required_vram_gb

        if not json_output:
            console.print(f"  [cyan]Model:[/cyan] {model_id}")
            console.print(f"  [cyan]Required VRAM:[/cyan] {required_vram:.2f} GB")
            console.print(f"  [cyan]Safety margin:[/cyan] {int(margin * 100)}%")
            console.print()

        if show_all:
            selected = find_compatible_instances(required_vram, instance_data.instances, margin)
        else:
            minimum = find_minimum_instance(required_vram, instance_data.instances, margin)
            selected = [minimum] if minimum is not None else []

        if not selected:
            message = "No compatible instances found" if show_all else "No suitable instance found"
            if json_output:
                typer.echo(
                    json.dumps(
                        {"error": message, "instances" if show_all else "instance": [] if show_all else None}
                    )
                )
            else:
                console.print(f"[bold red]{message}.[/bold red]")
                if not show_all:
                    console.print("[dim]Model requires more VRAM than any available instance.[/dim]")
            raise typer.Exit(code=1)

        instances = [_instance_output(instance, required_vram, provider) for instance in selected]
        if json_output:
            output = {
                "modelId": model_id,
                "requiredVramGb": round(required_vram, 2),
                "safetyMargin": margin,
                "instances" if show_all else "instance": instances if show_all else instances[0],
            }
            typer.echo(json.dumps(output, indent=2))
        elif show_all:
            _print_table(console, provider, instances)
        else:
            _print_recommendation(console, provider, model_id, instances[0])
    except HfMemError as e:
        if json_output:
            typer.echo(json.dumps({"error": e.message, "details": e.details}))
        else:
            console.print(f"[bold red]Error:[/bold red] {e.message}")
            if e.details:
                console.print(f"[dim]{e.details}[/dim]")
        raise typer.Exit(code=1) from e
