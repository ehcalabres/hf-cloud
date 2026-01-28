"""AWS SageMaker CLI commands."""

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from ..core.config import Config
from ..core.exceptions import DeploymentNotFoundError, ProviderError
from ..core.state import StateManager
from ..providers.registry import ProviderRegistry
from ..providers.sagemaker.utils import _decode_ansi_logs

console = Console()
sagemaker_app = typer.Typer(help="AWS SageMaker deployment management.")


def _get_provider_defaults() -> dict[str, str | None]:
    """Get default values from provider configuration."""
    config = Config()
    provider_config = config.get_provider_config("sagemaker")
    return {
        "region": provider_config.get("default_region"),
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
    effective_region = region or defaults.get("region") or "us-east-1"
    effective_role = role or defaults.get("role")
    effective_instance_type = instance_type or defaults.get("instance_type") or "ml.g5.xlarge"

    console.print(f"[bold blue]Deploying {model_id} to AWS SageMaker...[/bold blue]")

    try:
        provider = ProviderRegistry.get_provider("sagemaker")

        config = {
            "instance_type": effective_instance_type,
            "region": effective_region,
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
        console.print(f"[dim]  hf-cloud sagemaker status {deployment.deployment_id}[/dim]")

    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Deployment failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@sagemaker_app.command(name="ls")
def list_deployments(
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="Filter by region")] = None,
    status: Annotated[
        Optional[str], typer.Option("--status", "-s", help="Filter by status (running, creating, failed)")
    ] = None,
) -> None:
    """List all SageMaker deployments."""
    try:
        provider = ProviderRegistry.get_provider("sagemaker")

        filters = {}
        if region:
            filters["region"] = region
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
) -> None:
    """Show detailed information about a deployment."""
    try:
        provider = ProviderRegistry.get_provider("sagemaker")
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
) -> None:
    """Delete a SageMaker deployment."""
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete deployment '{deployment_id}'?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Abort()

    try:
        console.print(f"[yellow]Deleting deployment {deployment_id}...[/yellow]")

        provider = ProviderRegistry.get_provider("sagemaker")
        provider.delete_deployment(deployment_id)

        # Remove from local state
        state = StateManager()
        state.remove_deployment(deployment_id)

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
) -> None:
    """Check the status of a deployment."""
    try:
        provider = ProviderRegistry.get_provider("sagemaker")
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
) -> None:
    """View deployment logs from CloudWatch."""
    try:
        provider = ProviderRegistry.get_provider("sagemaker")
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
) -> None:
    """Test inference on the deployment."""
    try:
        provider = ProviderRegistry.get_provider("sagemaker")

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
