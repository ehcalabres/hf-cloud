"""Global CLI commands for HF-Cloud."""

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from hf_cloud.core.config import Config
from hf_cloud.core.state import StateManager
from hf_cloud.providers.registry import ProviderRegistry

console = Console()


def list_all_deployments(
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="Filter by provider (sagemaker, azure, vertex)"),
    ] = None,
    status: Annotated[
        Optional[str],
        typer.Option("--status", "-s", help="Filter by status (running, stopped, etc.)"),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", "-r", help="Refresh status from cloud providers"),
    ] = False,
) -> None:
    """List all deployments across all providers."""
    state = StateManager()

    if refresh:
        console.print("[dim]Refreshing deployment status from cloud providers...[/dim]")
        for provider_name in ProviderRegistry.list_providers():
            try:
                provider_obj = ProviderRegistry.get_provider(provider_name)
                remote_deployments = provider_obj.list_deployments()
                for deployment in remote_deployments:
                    state.add_deployment(deployment)
                console.print(f"  [green]✓[/green] {provider_name}")
            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow] {provider_name}: {e}")
        console.print()

    # Get deployments from state
    deployments = state.list_deployments(provider=provider)

    if status:
        deployments = [d for d in deployments if d.get("status") == status]

    if not deployments:
        console.print("[dim]No deployments found.[/dim]")
        return

    # Display as table
    table = Table(title="All Deployments")

    table.add_column("ID", style="cyan", max_width=20)
    table.add_column("Name", style="green")
    table.add_column("Provider", style="blue")
    table.add_column("Model", style="yellow", max_width=25)
    table.add_column("Status", style="magenta")
    table.add_column("Region")

    for deployment in deployments:
        deployment_id = deployment.get("deployment_id", "")
        display_id = deployment_id[:17] + "..." if len(deployment_id) > 20 else deployment_id

        model_id = deployment.get("model_id", "unknown")
        display_model = model_id[:22] + "..." if len(model_id) > 25 else model_id

        status_value = deployment.get("status", "unknown")
        status_style = (
            "green" if status_value == "running" else "yellow" if status_value == "creating" else "red"
        )

        table.add_row(
            display_id,
            deployment.get("deployment_name", "N/A"),
            deployment.get("provider", "N/A"),
            display_model,
            f"[{status_style}]{status_value}[/{status_style}]",
            deployment.get("region", "N/A"),
        )

    console.print(table)


# Provider management commands
providers_app = typer.Typer(help="Manage cloud provider configurations.")


@providers_app.command(name="ls")
def list_providers() -> None:
    """List all available cloud providers."""
    available = ProviderRegistry.list_providers()

    console.print()
    console.print("[bold]Available Providers:[/bold]")
    console.print()

    for provider in available:
        # Check if provider has dependencies installed
        try:
            ProviderRegistry.get_provider(provider)
            status = "[green]✓ ready[/green]"
        except Exception:
            status = "[yellow]○ not configured[/yellow]"

        console.print(f"  {provider:12} {status}")

    console.print()
    console.print("[dim]Configure a provider with: hf-cloud providers configure <provider>[/dim]")


@providers_app.command(name="configure")
def configure_provider(
    provider: Annotated[str, typer.Argument(help="Provider name (sagemaker, azure, vertex)")],
) -> None:
    """Configure credentials and defaults for a provider."""
    if provider not in ProviderRegistry.list_providers():
        console.print(f"[bold red]Error:[/bold red] Unknown provider '{provider}'")
        console.print(f"[dim]Available providers: {', '.join(ProviderRegistry.list_providers())}[/dim]")
        raise typer.Exit(code=1)

    config = Config()
    current_config = config.get_provider_config(provider)

    console.print(f"[bold]Configuring {provider}[/bold]")
    console.print()

    if provider == "sagemaker":
        # SageMaker configuration
        default_region = typer.prompt(
            "Default AWS region",
            default=current_config.get("default_region", "us-east-1"),
        )
        default_role = typer.prompt(
            "Default IAM role ARN (optional, press Enter to skip)",
            default=current_config.get("default_role", ""),
        )
        default_instance_type = typer.prompt(
            "Default instance type (optional, press Enter to skip)",
            default=current_config.get("default_instance_type", ""),
        )

        new_config = {
            "default_region": default_region,
        }
        if default_role:
            new_config["default_role"] = default_role
        if default_instance_type:
            new_config["default_instance_type"] = default_instance_type

        config.set_provider_config(provider, new_config)

    # TODO: Check configs exposed once Microsoft and Google SDKs are integrated
    elif provider == "azure":
        # Azure configuration
        subscription_id = typer.prompt(
            "Azure Subscription ID",
            default=current_config.get("subscription_id", ""),
        )
        default_resource_group = typer.prompt(
            "Default Resource Group",
            default=current_config.get("default_resource_group", ""),
        )
        default_workspace = typer.prompt(
            "Default ML Workspace",
            default=current_config.get("default_workspace", ""),
        )

        config.set_provider_config(
            provider,
            {
                "subscription_id": subscription_id,
                "default_resource_group": default_resource_group,
                "default_workspace": default_workspace,
            },
        )

    elif provider == "vertex" or provider == "vertex":
        # vertex/Vertex AI configuration
        project_id = typer.prompt(
            "vertex Project ID",
            default=current_config.get("default_project", ""),
        )
        default_location = typer.prompt(
            "Default location/region",
            default=current_config.get("default_location", "us-central1"),
        )
        default_machine_type = typer.prompt(
            "Default machine type (optional, press Enter to skip)",
            default=current_config.get("default_machine_type", ""),
        )

        new_config = {
            "default_project": project_id,
            "default_location": default_location,
        }
        if default_machine_type:
            new_config["default_machine_type"] = default_machine_type

        config.set_provider_config("vertex", new_config)

    console.print()
    console.print(f"[bold green]✓[/bold green] Configuration saved for {provider}")
    console.print(f"[dim]Config file: {config.config_file}[/dim]")


@providers_app.command(name="show")
def show_provider_config(
    provider: Annotated[str, typer.Argument(help="Provider name")],
) -> None:
    """Show configuration for a provider."""
    if provider not in ProviderRegistry.list_providers():
        console.print(f"[bold red]Error:[/bold red] Unknown provider '{provider}'")
        raise typer.Exit(code=1)

    config = Config()
    provider_config = config.get_provider_config(provider)

    if not provider_config:
        console.print(f"[dim]No configuration found for {provider}.[/dim]")
        console.print(f"[dim]Run: hf-cloud providers configure {provider}[/dim]")
        return

    console.print(f"[bold]Configuration for {provider}:[/bold]")
    console.print()

    for key, value in provider_config.items():
        # Mask sensitive values
        if "secret" in key.lower() or "token" in key.lower() or "key" in key.lower():
            display_value = "****" + str(value)[-4:] if value else "not set"
        else:
            display_value = value or "not set"
        console.print(f"  {key}: {display_value}")
