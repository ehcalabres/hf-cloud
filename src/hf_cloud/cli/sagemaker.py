
"""AWS SageMaker CLI commands."""

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from hf_cloud.cli.estimate import estimate_instances
from hf_cloud.core.config import Config
from hf_cloud.core.exceptions import DeploymentNotFoundError, ProviderError
from hf_cloud.core.state import StateManager
from hf_cloud.providers.sagemaker import SageMakerProvider
from hf_cloud.providers.sagemaker.utils import _decode_ansi_logs
from hf_cloud.utils.hf_mem import EstimateOptions

console = Console()
sagemaker_app = typer.Typer(help="AWS SageMaker deployment management.")


def _get_provider_defaults() -> dict[str, Optional[str]]:
    """Get default values from provider configuration."""
    config = Config()
    provider_config = config.get_provider_config("sagemaker")
    return {
        "role": provider_config.get("default_role"),
        "instance_type": provider_config.get("default_instance_type"),
    }


@sagemaker_app.command(name="deploy")
def deploy(
    model_id: Annotated[
        str, typer.Argument(help="HuggingFace model ID (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')")
    ],
    deployment_name: Annotated[str, typer.Option("--name", "-n", help="Name for the deployment/endpoint")],
    instance_type: Annotated[
        Optional[str], typer.Option("--instance-type", "-i", help="EC2 instance type (e.g., 'ml.g5.xlarge')")
    ] = None,
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="AWS region")] = None,
    role: Annotated[Optional[str], typer.Option("--role", help="IAM role ARN for SageMaker")] = None,
    token: Annotated[
        Optional[str], typer.Option("--token", "-t", help="HuggingFace Hub token for private models")
    ] = None,
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", help="Number of instances")] = 1,
) -> None:
    """Deploy a HuggingFace model to AWS SageMaker."""
    # Load provider configuration defaults
    defaults = _get_provider_defaults()

    # Use provided values or fall back to config defaults, then hardcoded defaults
    effective_role = role or defaults.get("role")
    effective_instance_type = instance_type or defaults.get("instance_type") or "ml.g5.xlarge"

    console.print(f"[bold blue]Deploying {model_id} to AWS SageMaker...[/bold blue]")

    try:
        provider = SageMakerProvider(region=region)

        config = {
            "instance_type": effective_instance_type,
            "region": provider.client.region,
            "role": effective_role,
            "instance_count": instance_count,
        }

        deployment = provider.deploy(
            model_id=model_id,
            deployment_name=deployment_name,
            config=config,
            token=token,
        )

        # Save to local state
        state = StateManager()
        state.add_deployment(deployment)

        console.print()
        console.print("[bold green]Deployment initiated successfully![/bold green]")
        console.print()
        console.print(f"  [cyan]Deployment ID:[/cyan] {deployment.deployment_id}")
        console.print(f"  [cyan]Status:[/cyan] {deployment.status.value}")
        console.print(f"  [cyan]Region:[/cyan] {deployment.region}")
        console.print(f"  [cyan]Instance Type:[/cyan] {deployment.instance_type}")
        console.print(f"  [cyan]Instance Count:[/cyan] {deployment.instance_count}")
        console.print()
        console.print("[dim]Note: Endpoint creation may take 5-10 minutes. Check status with:[/dim]")
        console.print(f"[dim]  hf-cloud sagemaker status {deployment.deployment_id} --region {deployment.region}[/dim]")

    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Deployment failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@sagemaker_app.command(name="ls")
def list_deployments(
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="AWS region")] = None,
    status: Annotated[
        Optional[str], typer.Option("--status", "-s", help="Filter by status (running, creating, failed)")
    ] = None,
) -> None:
    """List all SageMaker deployments."""
    try:
        provider = SageMakerProvider(region=region)

        filters = {}
        if status:
            filters["status"] = status

        deployments = provider.list_deployments(filters)

        if not deployments:
            console.print("[dim]No deployments found.[/dim]")
            return

        table = Table(title="SageMaker Deployments")
        table.add_column("Endpoint Name", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Status", style="magenta")
        table.add_column("Instance Type")
        table.add_column("Region")

        for deployment in deployments:
            status_style = "green" if deployment.status.value == "running" else "yellow"
            table.add_row(
                deployment.deployment_id,
                deployment.model_id[:30] if deployment.model_id else "unknown",
                f"[{status_style}]{deployment.status.value}[/{status_style}]",
                deployment.instance_type or "N/A",
                deployment.region or "N/A",
            )

        console.print(table)

    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@sagemaker_app.command(name="describe")
def describe(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="AWS region")] = None,
) -> None:
    """Show detailed information about a deployment."""
    try:
        provider = SageMakerProvider(region=region)
        deployment = provider.get_deployment(deployment_id)

        console.print()
        console.print(f"[bold cyan]Deployment: {deployment.deployment_name}[/bold cyan]")
        console.print()
        console.print(f"  [bold]ID:[/bold] {deployment.deployment_id}")
        console.print(f"  [bold]Model:[/bold] {deployment.model_id}")
        console.print(f"  [bold]Status:[/bold] {deployment.status.value}")
        console.print(f"  [bold]Provider:[/bold] {deployment.provider}")
        console.print(f"  [bold]Region:[/bold] {deployment.region}")
        console.print()
        console.print("[bold]Resources:[/bold]")
        console.print(f"  Instance Type: {deployment.instance_type}")
        console.print(f"  Instance Count: {deployment.instance_count}")
        console.print()
        console.print("[bold]Endpoint:[/bold]")
        console.print(f"  URL: {deployment.endpoint_url}")
        console.print()
        if deployment.created_at:
            console.print(f"  Created: {deployment.created_at}")
        if deployment.updated_at:
            console.print(f"  Updated: {deployment.updated_at}")

    except DeploymentNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Deployment '{deployment_id}' not found.")
        raise typer.Exit(code=1)
    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@sagemaker_app.command(name="delete")
def delete(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompt")] = False,
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="AWS region")] = None,
) -> None:
    """Delete a SageMaker deployment."""
    try:
        provider = SageMakerProvider(region=region)
        if not force:
            confirm = typer.confirm(
                f"Are you sure you want to delete deployment '{deployment_id}' in '{provider.client.region}'?"
            )
            if not confirm:
                console.print("[dim]Cancelled.[/dim]")
                raise typer.Abort()

        console.print(f"[yellow]Deleting deployment {deployment_id} in {provider.client.region}...[/yellow]")
        provider.delete_deployment(deployment_id)

        # Remove from local state
        state = StateManager()
        state.remove_deployment(deployment_id, provider="sagemaker", region=provider.client.region)

        console.print(f"[bold green]Deployment '{deployment_id}' deleted successfully.[/bold green]")

    except DeploymentNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Deployment '{deployment_id}' not found.")
        raise typer.Exit(code=1)
    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@sagemaker_app.command(name="status")
def status(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="AWS region")] = None,
) -> None:
    """Check the status of a deployment."""
    try:
        provider = SageMakerProvider(region=region)
        deployment_status = provider.get_status(deployment_id)

        status_colors = {
            "creating": "yellow",
            "running": "green",
            "updating": "yellow",
            "deleting": "red",
            "failed": "red",
            "stopped": "dim",
        }

        color = status_colors.get(deployment_status.value, "white")
        console.print(f"Status: [{color}]{deployment_status.value}[/{color}]")

    except DeploymentNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Deployment '{deployment_id}' not found.")
        raise typer.Exit(code=1)
    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@sagemaker_app.command(name="logs")
def logs(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
    tail: Annotated[int, typer.Option("--tail", "-n", help="Number of log lines to show")] = 100,
    raw: Annotated[bool, typer.Option("--raw", help="Show raw logs without ANSI code processing")] = False,
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="AWS region")] = None,
) -> None:
    """View deployment logs from CloudWatch."""
    try:
        provider = SageMakerProvider(region=region)
        log_output = provider.get_logs(deployment_id, tail)

        if log_output:
            if not raw:
                log_output = _decode_ansi_logs(log_output)
            # Use print() instead of console.print() to preserve ANSI codes
            print(log_output)
        else:
            console.print("[dim]No logs available.[/dim]")

    except DeploymentNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Deployment '{deployment_id}' not found.")
        raise typer.Exit(code=1)
    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@sagemaker_app.command(name="invoke")
def invoke(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
    input_text: Annotated[str, typer.Option("--input", "-i", help="Input text for inference")],
    max_new_tokens: Annotated[int, typer.Option("--max-tokens", help="Maximum tokens to generate")] = 100,
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="AWS region")] = None,
) -> None:
    """Test inference on the deployment."""
    try:
        provider = SageMakerProvider(region=region)

        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": max_new_tokens,
            },
        }

        console.print(f"[dim]Invoking endpoint {deployment_id}...[/dim]")
        result = provider.invoke(deployment_id, payload)

        console.print()
        console.print("[bold]Response:[/bold]")
        console.print(result)

    except DeploymentNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Deployment '{deployment_id}' not found.")
        raise typer.Exit(code=1)
    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@sagemaker_app.command(name="estimate")
def estimate(
    model_id: Annotated[
        str, typer.Argument(help="Namespaced Hub model ID, URL, or local path (e.g., openai-community/gpt2)")
    ],
    safety_margin: Annotated[
        Optional[float],
        typer.Option(
            "--safety-margin",
            "-s",
            help="Fraction of VRAM to keep free (e.g., 0.1 for 10%)",
            rich_help_panel="Selection",
        ),
    ] = None,
    show_all: Annotated[
        bool, typer.Option("--all", "-a", help="Show all compatible instances", rich_help_panel="Selection")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output results as JSON", rich_help_panel="Output")
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
    """Estimate the minimum viable SageMaker instance for a model.

    Uses hf-mem to calculate VRAM requirements and matches against
    available SageMaker instance types.
    """
    estimate_instances(
        model_id,
        provider="sagemaker",
        safety_margin=safety_margin,
        show_all=show_all,
        json_output=json_output,
        console=console,
        options=EstimateOptions(
            revision=revision,
            file=file,
            format=format,
            max_model_len=max_model_len,
            batch_size=batch_size,
            kv_cache_dtype=kv_cache_dtype,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
    )
