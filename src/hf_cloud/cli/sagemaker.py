"""AWS SageMaker CLI commands."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from hf_cloud.cli.deploy_code import sagemaker_code
from hf_cloud.cli.deployment_selection import announce_selection, choose_instance
from hf_cloud.cli.environment import deployment_environment
from hf_cloud.cli.estimate import provider_estimate
from hf_cloud.core.config import Config
from hf_cloud.core.deployment import DeploymentStatus
from hf_cloud.core.exceptions import DeploymentError, DeploymentNotFoundError, HFCloudError, ProviderError
from hf_cloud.core.state import StateManager
from hf_cloud.providers.sagemaker import SageMakerProvider
from hf_cloud.providers.sagemaker.client import SageMakerClient
from hf_cloud.providers.sagemaker.utils import _decode_ansi_logs, get_serving_settings
from hf_cloud.utils.hf_hub import get_model_task
from hf_cloud.utils.sdk_versions import SdkUpdateCheck

console = Console()
sagemaker_app = typer.Typer(help="AWS SageMaker deployment management.")


def _get_provider_defaults() -> dict[str, Optional[str]]:
    """Get default values from provider configuration."""
    config = Config()
    provider_config = config.get_provider_config("sagemaker")
    return {
        "role": provider_config.get("default_role"),
    }


@sagemaker_app.command(name="deploy")
def deploy(
    model_id: Annotated[
        str, typer.Argument(help="HuggingFace model ID (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')")
    ],
    deployment_name: Annotated[str, typer.Option("--name", "-n", help="Name for the deployment/endpoint")],
    instance_type: Annotated[
        Optional[str],
        typer.Option("--instance-type", "-i", help="EC2 type; omitted selects cheapest compatible estimate"),
    ] = None,
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="AWS region")] = None,
    role: Annotated[
        Optional[str],
        typer.Option("--role", help="SageMaker execution role ARN or name; overrides configured default"),
    ] = None,
    role_name: Annotated[
        Optional[str],
        typer.Option(
            "--role-name", help="Execution role name to resolve through IAM; overrides configured default"
        ),
    ] = None,
    token: Annotated[
        Optional[str], typer.Option("--token", "-t", help="HuggingFace Hub token for private models")
    ] = None,
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", help="Number of instances")] = 1,
    tensor_parallel_size: Annotated[
        Optional[int],
        typer.Option(
            "--tensor-parallel-size",
            min=1,
            help="vLLM GPUs per replica (defaults to the bundled instance GPU count, otherwise 1)",
        ),
    ] = None,
    print_code: Annotated[
        bool, typer.Option("--print-code", help="Print standalone Python SDK code without deploying")
    ] = False,
    env: Annotated[
        Optional[list[str]], typer.Option("--env", help="Runtime KEY=VALUE; repeat to override defaults/files")
    ] = None,
    env_file: Annotated[
        Optional[list[Path]], typer.Option("--env-file", help="UTF-8 KEY=VALUE file; repeat in merge order")
    ] = None,
) -> None:
    """Deploy a HuggingFace model to AWS SageMaker."""
    custom_env = deployment_environment(env, env_file)
    if role is not None and role_name is not None:
        raise typer.BadParameter("Use either --role or --role-name, not both", param_hint="--role-name")
    if role_name is not None and (role_name.startswith("arn:") or "/" in role_name):
        raise typer.BadParameter(
            "Provide a role name without a path; use --role for an ARN", param_hint="--role-name"
        )
    # Load provider configuration defaults
    defaults = _get_provider_defaults()

    # Role defaults remain configured; hardware is selected per model below.
    effective_role = role_name if role_name is not None else role if role is not None else defaults.get("role")
    if instance_count < 1:
        raise typer.BadParameter("Instance count must be positive", param_hint="--instance-count")
    effective_instance_type = instance_type
    if effective_instance_type is None:
        typer.echo("Estimating compatible SageMaker instances...", err=True)
        try:
            match = choose_instance(
                model_id,
                "sagemaker",
                token=token,
                tensor_parallel_size=tensor_parallel_size,
            )
        except HFCloudError as error:
            typer.echo(
                f"Error selecting an instance: {error}. Use --instance-type to select explicitly.", err=True
            )
            raise typer.Exit(code=1) from error
        announce_selection(match)
        effective_instance_type = match.instance["instanceType"]
    try:
        get_serving_settings(effective_instance_type, tensor_parallel_size)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--tensor-parallel-size") from error

    if print_code:
        typer.echo(
            sagemaker_code(
                model_id,
                deployment_name,
                effective_instance_type,
                instance_count,
                SageMakerClient.resolve_region(region),
                effective_role,
                custom_env.get("HF_TASK") or get_model_task(model_id, token),
                tensor_parallel_size,
                custom_env,
            )
        )
        return

    console.print(f"[bold blue]Deploying {model_id} to AWS SageMaker...[/bold blue]")
    update_check = SdkUpdateCheck("sagemaker")

    try:
        provider = SageMakerProvider(region=region)

        config = {
            "instance_type": effective_instance_type,
            "region": provider.client.region,
            "role": effective_role,
            "instance_count": instance_count,
            "tensor_parallel_size": tensor_parallel_size,
            "environment_variables": custom_env,
        }

        deployment = provider.deploy(
            model_id=model_id,
            deployment_name=deployment_name,
            config=config,
            token=token,
        )
        if deployment.status != DeploymentStatus.RUNNING:
            raise DeploymentError(
                f"Endpoint deployment returned status '{deployment.status.value}'",
                deployment_id=deployment.deployment_id,
            )

        # Save to local state
        state = StateManager()
        state.add_deployment(deployment)

        console.print()
        console.print("[bold green]Deployment completed successfully![/bold green]")
        console.print()
        console.print(f"  [cyan]Deployment ID:[/cyan] {deployment.deployment_id}")
        console.print(f"  [cyan]Status:[/cyan] {deployment.status.value}")
        console.print(f"  [cyan]Region:[/cyan] {deployment.region}")
        console.print(f"  [cyan]Instance Type:[/cyan] {deployment.instance_type}")
        console.print(f"  [cyan]Instance Count:[/cyan] {deployment.instance_count}")
        console.print()
        console.print("[dim]Check status with:[/dim]")
        console.print(
            f"[dim]  hf-cloud sagemaker status {deployment.deployment_id} --region {deployment.region}[/dim]"
        )

    except (ProviderError, DeploymentError) as e:
        console.print(Text(f"Error: {e}", style="bold red"))
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(Text(f"Deployment failed: {e}", style="bold red"))
        raise typer.Exit(code=1)
    finally:
        update_check.notify(console)


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
    api: Annotated[
        str, typer.Option("--api", help="Request format: auto (use saved deployment), vllm, or huggingface")
    ] = "auto",
) -> None:
    """Test inference on the deployment."""
    if api not in ("auto", "vllm", "huggingface"):
        raise typer.BadParameter("Choose auto, vllm, or huggingface", param_hint="--api")
    try:
        provider = SageMakerProvider(region=region)
        saved = (
            StateManager().get_deployment(
                deployment_id,
                provider="sagemaker",
                region=provider.client.region,
            )
            or {}
        )
        if api == "auto" and not saved.get("config", {}).get("model_server"):
            try:
                saved = provider.get_deployment(deployment_id).to_dict()
            except ProviderError:
                pass  # An explicit --api still works without describe permissions.
        use_vllm = api == "vllm" or (api == "auto" and saved.get("config", {}).get("model_server") == "vllm")

        payload = (
            {"prompt": input_text, "max_tokens": max_new_tokens}
            if use_vllm
            else {
                "inputs": input_text,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                },
            }
        )
        if use_vllm and saved.get("model_id") not in (None, "unknown"):
            payload["model"] = saved["model_id"]

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


# Keep the provider spelling as a shortcut to the canonical estimate command.
sagemaker_app.command(name="estimate")(provider_estimate("sagemaker"))
