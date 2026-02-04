"""Google Cloud Vertex AI CLI commands."""

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from hf_cloud.core.config import Config
from hf_cloud.core.exceptions import DeploymentNotFoundError, ProviderError
from hf_cloud.core.state import StateManager
from hf_cloud.providers.registry import ProviderRegistry

console = Console()
vertex_app = typer.Typer(help="Google Cloud Vertex AI deployment management.")


def _get_provider_defaults() -> dict[str, str | None]:
    """Get default values from provider configuration."""
    config = Config()
    provider_config = config.get_provider_config("vertex")
    return {
        "project": provider_config.get("default_project"),
        "location": provider_config.get("default_location"),
        "machine_type": provider_config.get("default_machine_type"),
    }


@vertex_app.command(name="deploy")
def deploy(
    model_id: Annotated[
        str, typer.Argument(help="HuggingFace model ID (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')")
    ],
    deployment_name: Annotated[str, typer.Option("--name", "-n", help="Name for the deployment/endpoint")],
    machine_type: Annotated[
        Optional[str], typer.Option("--machine-type", "-m", help="vertex machine type (e.g., 'n1-standard-4')")
    ] = None,
    accelerator_type: Annotated[
        Optional[str], typer.Option("--accelerator-type", "-a", help="GPU type (e.g., 'NVIDIA_TESLA_T4')")
    ] = None,
    accelerator_count: Annotated[int, typer.Option("--accelerator-count", help="Number of GPUs")] = 1,
    location: Annotated[Optional[str], typer.Option("--location", "-l", help="vertex location/region")] = None,
    project: Annotated[Optional[str], typer.Option("--project", "-p", help="vertex project ID")] = None,
    token: Annotated[
        Optional[str], typer.Option("--token", "-t", help="HuggingFace Hub token for private models")
    ] = None,
    min_replicas: Annotated[int, typer.Option("--min-replicas", help="Minimum number of replicas")] = 1,
    max_replicas: Annotated[int, typer.Option("--max-replicas", help="Maximum number of replicas")] = 1,
) -> None:
    """Deploy a HuggingFace model to Google Cloud Vertex AI."""
    # Load provider configuration defaults
    defaults = _get_provider_defaults()

    # Use provided values or fall back to config defaults, then hardcoded defaults
    effective_location = location or defaults.get("location") or "us-central1"
    effective_project = project or defaults.get("project")
    effective_machine_type = machine_type or defaults.get("machine_type") or "n1-standard-4"

    console.print(f"[bold blue]Deploying {model_id} to Google Cloud Vertex AI...[/bold blue]")

    try:
        provider = ProviderRegistry.get_provider("vertex")

        config = {
            "machine_type": effective_machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
            "location": effective_location,
            "project": effective_project,
            "min_replica_count": min_replicas,
            "max_replica_count": max_replicas,
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
        console.print(f"  [cyan]Location:[/cyan] {deployment.region}")
        console.print(f"  [cyan]Machine Type:[/cyan] {deployment.instance_type}")
        if accelerator_type:
            console.print(f"  [cyan]Accelerator:[/cyan] {accelerator_type} x {accelerator_count}")
        console.print()
        console.print("[dim]Note: Endpoint creation may take several minutes. Check status with:[/dim]")
        console.print(f"[dim]  hf-cloud vertex status {deployment.deployment_name}[/dim]")

    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Deployment failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@vertex_app.command(name="ls")
def list_deployments(
    location: Annotated[Optional[str], typer.Option("--location", "-l", help="Filter by location")] = None,
) -> None:
    """List all Vertex AI deployments."""
    try:
        provider = ProviderRegistry.get_provider("vertex")

        filters = {}
        if location:
            filters["location"] = location

        deployments = provider.list_deployments(filters)

        if not deployments:
            console.print("[dim]No deployments found.[/dim]")
            return

        table = Table(title="Vertex AI Deployments")
        table.add_column("Endpoint Name", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Status", style="magenta")
        table.add_column("Machine Type")
        table.add_column("Location")

        for deployment in deployments:
            status_style = "green" if deployment.status.value == "running" else "yellow"
            table.add_row(
                deployment.deployment_name,
                deployment.model_id[:30] if deployment.model_id else "unknown",
                f"[{status_style}]{deployment.status.value}[/{status_style}]",
                deployment.instance_type or "N/A",
                deployment.region or "N/A",
            )

        console.print(table)

    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@vertex_app.command(name="describe")
def describe(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
) -> None:
    """Show detailed information about a deployment."""
    try:
        provider = ProviderRegistry.get_provider("vertex")
        deployment = provider.get_deployment(deployment_id)

        console.print()
        console.print(f"[bold cyan]Deployment: {deployment.deployment_name}[/bold cyan]")
        console.print()
        console.print(f"  [bold]ID:[/bold] {deployment.deployment_id}")
        console.print(f"  [bold]Model:[/bold] {deployment.model_id}")
        console.print(f"  [bold]Status:[/bold] {deployment.status.value}")
        console.print(f"  [bold]Provider:[/bold] {deployment.provider}")
        console.print(f"  [bold]Location:[/bold] {deployment.region}")
        console.print()
        console.print("[bold]Resources:[/bold]")
        console.print(f"  Machine Type: {deployment.instance_type}")
        console.print(f"  Replica Count: {deployment.instance_count}")
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


@vertex_app.command(name="delete")
def delete(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompt")] = False,
) -> None:
    """Delete a Vertex AI deployment."""
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete deployment '{deployment_id}'?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Abort()

    try:
        console.print(f"[yellow]Deleting deployment {deployment_id}...[/yellow]")

        provider = ProviderRegistry.get_provider("vertex")
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


@vertex_app.command(name="status")
def status(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
) -> None:
    """Check the status of a deployment."""
    try:
        provider = ProviderRegistry.get_provider("vertex")
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


@vertex_app.command(name="logs")
def logs(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
    tail: Annotated[int, typer.Option("--tail", "-n", help="Number of log lines to show")] = 100,
) -> None:
    """View deployment logs from Cloud Logging."""
    try:
        provider = ProviderRegistry.get_provider("vertex")
        log_output = provider.get_logs(deployment_id, tail)

        if log_output:
            print(log_output)
        else:
            console.print("[dim]No logs available.[/dim]")

    except DeploymentNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Deployment '{deployment_id}' not found.")
        raise typer.Exit(code=1)
    except ProviderError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@vertex_app.command(name="invoke")
def invoke(
    deployment_id: Annotated[str, typer.Argument(help="Deployment/endpoint name")],
    input_text: Annotated[str, typer.Option("--input", "-i", help="Input text for inference")],
    max_new_tokens: Annotated[int, typer.Option("--max-tokens", help="Maximum tokens to generate")] = 100,
) -> None:
    """Test inference on the deployment."""
    try:
        provider = ProviderRegistry.get_provider("vertex")

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
