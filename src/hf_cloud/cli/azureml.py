"""Published Hugging Face catalog models on Azure ML managed online endpoints."""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from hf_cloud.cli.deploy_code import azureml_code
from hf_cloud.cli.deployment_selection import announce_selection, choose_instance
from hf_cloud.cli.environment import deployment_environment
from hf_cloud.cli.estimate import provider_estimate
from hf_cloud.core.exceptions import HFCloudError
from hf_cloud.core.state import StateManager
from hf_cloud.providers.azureml import AzureMLProvider
from hf_cloud.utils.sdk_versions import SdkUpdateCheck

console = Console()
azureml_app = typer.Typer(help="Azure ML managed online endpoints. Workspace options go before the subcommand.")


@azureml_app.callback()
def workspace_scope(
    ctx: typer.Context,
    subscription_id: Annotated[Optional[str], typer.Option("--subscription-id")] = None,
    resource_group: Annotated[Optional[str], typer.Option("--resource-group", "-g")] = None,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    deployment_name: Annotated[
        Optional[str], typer.Option("--deployment-name", help="Select a component within an existing endpoint")
    ] = None,
) -> None:
    ctx.obj = AzureMLProvider(subscription_id, resource_group, workspace, deployment_name)


@contextmanager
def _errors(*, stderr: bool = False):
    try:
        yield
    except (HFCloudError, OSError, ValueError) as error:
        (Console(stderr=True) if stderr else console).print(Text(f"Error: {error}", style="bold red"))
        raise typer.Exit(code=1) from error


@azureml_app.command()
def catalog(
    ctx: typer.Context,
    model_id: Annotated[str, typer.Argument(help="Hub model ID to look up in the Azure ML catalog")],
    catalog_version: Annotated[
        Optional[str], typer.Option("--catalog-version", help="Published catalog version; defaults to latest")
    ] = None,
) -> None:
    """Check catalog availability and resolve a pinned published model; requires az login."""
    with _errors():
        typer.echo(
            json.dumps(ctx.obj.catalog.resolve(model_id, "azureml", catalog_version).to_dict(), indent=2)
        )


def _export_plan(directory: Path, plan: dict) -> None:
    from hf_cloud.core.storage import write_json_atomic

    paths = [directory / "endpoint.json", directory / "deployment.json"]
    if any(path.exists() for path in paths):
        raise ValueError(
            "Export would overwrite endpoint.json or deployment.json; select an empty export directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths[0],
        {
            "$schema": "https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json",
            **plan["endpoint"],
            "identity": {"type": "system_assigned"},
        },
    )
    write_json_atomic(
        paths[1],
        {
            "$schema": "https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json",
            **plan["deployment"],
        },
    )


@azureml_app.command()
def deploy(
    ctx: typer.Context,
    model_id: Annotated[str, typer.Argument(help="Published Hugging Face Hub model ID (org/model)")],
    name: Annotated[str, typer.Option("--name", "-n", help="New endpoint name")],
    instance_type: Annotated[
        Optional[str],
        typer.Option("--instance-type", "-i", help="VM size; omitted selects cheapest compatible estimate"),
    ] = None,
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", min=1)] = 1,
    deployment_name: Annotated[
        str, typer.Option("--deployment-name", help="Component name inside the new endpoint")
    ] = "main",
    catalog_version: Annotated[
        Optional[str],
        typer.Option("--catalog-version", help="Pin an Azure catalog version (not a Hub revision)"),
    ] = None,
    hf_token_connection: Annotated[
        Optional[str],
        typer.Option("--hf-token-connection", help="Workspace secret connection for a published gated model"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Resolve and print the published deployment plan; reads the catalog, creates no resources",
        ),
    ] = False,
    export_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--export-dir",
            help="Export pinned endpoint/deployment JSON for Azure CLI; use --dry-run to export without deploying",
        ),
    ] = None,
    print_code: Annotated[
        bool,
        typer.Option("--print-code", help="Print pinned Python SDK code; reads the catalog, never deploys"),
    ] = False,
    env: Annotated[
        Optional[list[str]], typer.Option("--env", help="Runtime KEY=VALUE; repeat to override defaults/files")
    ] = None,
    env_file: Annotated[
        Optional[list[Path]], typer.Option("--env-file", help="UTF-8 KEY=VALUE file; repeat in merge order")
    ] = None,
) -> None:
    """Deploy a published catalog model using its provided runtime and scoring configuration."""
    custom_env = deployment_environment(env, env_file)
    if hf_token_connection and "HF_TOKEN" in custom_env:
        raise typer.BadParameter("Use either --hf-token-connection or an HF_TOKEN environment override")
    if print_code and (dry_run or export_dir):
        raise typer.BadParameter("Use --print-code separately from --dry-run and --export-dir")
    with _errors(stderr=print_code or dry_run):
        provider = ctx.obj
        if provider.client.deployment_name:
            raise typer.BadParameter("For a new endpoint, put --deployment-name after deploy")
        resource_id = provider.client.resource_id(name)
        if instance_type is None:
            typer.echo("Estimating compatible Azure ML instances...", err=True)
            match = choose_instance(
                model_id,
                "azureml",
                registry=provider.catalog,
                catalog_version=catalog_version,
            )
            announce_selection(match)
            instance_type = match.instance["instanceType"]
            catalog_version = match.instance["modelAssetId"].rsplit("/", 1)[-1]
        config = {
            "instance_type": instance_type,
            "instance_count": instance_count,
            "deployment_name": deployment_name,
            "catalog_version": catalog_version,
            "hf_token_connection": hf_token_connection,
            "environment_variables": custom_env,
        }
        plan = provider.prepare_deployment(model_id, name, config)
        config["catalog_version"] = plan["deployment"]["model"].rsplit("/", 1)[-1]
        if print_code:
            typer.echo(azureml_code(plan, provider.client))
            return
        if export_dir:
            _export_plan(export_dir, plan)
        if dry_run:
            typer.echo(json.dumps({"resourceId": resource_id, **plan}, indent=2))
            return
        check = SdkUpdateCheck("azureml")
        try:
            console.print(Text(f"Published model: {plan['deployment']['model']}"))
            console.print(
                "Using the catalog's runtime and scoring configuration. Provisioning can take several minutes."
            )
            with console.status("Deploying Azure ML catalog model..."):
                deployment = provider.deploy(model_id, name, config)
            StateManager().add_deployment(deployment)
            console.print("Deployment completed successfully!", style="bold green")
            console.print(
                Text(
                    f"Endpoint: {deployment.deployment_name}\nStatus: {deployment.status.value}\nScoring URI: {deployment.endpoint_url}"
                )
            )
            console.print(
                Text(f"Test: hf-cloud azureml invoke {deployment.deployment_name} --input 'Hello'"), style="dim"
            )
        finally:
            check.notify(console)


@azureml_app.command(name="ls")
def list_deployments(ctx: typer.Context) -> None:
    """List HF Cloud endpoints in the selected workspace."""
    with _errors():
        table = Table(title="Azure ML managed online endpoints")
        for column in ("Name", "Hub model", "Instance", "Replicas", "Status"):
            table.add_column(column)
        for deployment in ctx.obj.list_deployments():
            table.add_row(
                deployment.deployment_name,
                deployment.model_id,
                deployment.instance_type or "—",
                str(deployment.instance_count),
                deployment.status.value,
            )
        console.print(table)


@azureml_app.command()
def describe(ctx: typer.Context, deployment_id: str) -> None:
    """Show an endpoint by name or workspace-scoped ARM ID."""
    with _errors():
        typer.echo(json.dumps(ctx.obj.get_deployment(deployment_id).to_dict(), indent=2))


@azureml_app.command()
def status(ctx: typer.Context, deployment_id: str) -> None:
    """Read actual endpoint and component provisioning status."""
    with _errors():
        deployment = ctx.obj.get_deployment(deployment_id)
        console.print(Text(f"Status: {deployment.status.value}"))
        if deployment.config.get("failure_reason"):
            console.print(Text(deployment.config["failure_reason"]))
        if deployment.status.value == "failed":
            raise typer.Exit(code=1)


@azureml_app.command()
def update(
    ctx: typer.Context,
    deployment_id: str,
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", min=1)],
) -> None:
    """Scale the selected component while preserving its model, runtime and traffic."""
    with _errors():
        deployment = ctx.obj.update_deployment(deployment_id, {"instance_count": instance_count})
        StateManager().add_deployment(deployment)
        console.print(Text(f"Endpoint running with {deployment.instance_count} replicas."), style="green")


@azureml_app.command()
def delete(
    ctx: typer.Context, deployment_id: str, yes: Annotated[bool, typer.Option("--yes", "-y")] = False
) -> None:
    """Delete an endpoint and all its components, releasing their compute."""
    if not yes:
        typer.confirm(f"Delete Azure ML endpoint {deployment_id} and ALL its deployments?", abort=True)
    with _errors():
        resource_id = ctx.obj.client.resource_id(deployment_id)
        if ctx.obj.delete_deployment(deployment_id):
            StateManager().remove_deployment(
                resource_id, provider="azureml", region=ctx.obj.client.workspace.location
            )
            console.print("Endpoint deleted.", style="green")


@azureml_app.command()
def logs(
    ctx: typer.Context,
    deployment_id: str,
    tail: Annotated[int, typer.Option("--tail", "-n", min=1, max=1000)] = 100,
    container: Annotated[
        str, typer.Option("--container", help="inference-server or storage-initializer")
    ] = "inference-server",
) -> None:
    """Read serving or initialization logs for the selected component."""
    with _errors():
        typer.echo(ctx.obj.get_logs(deployment_id, tail, container_type=container))


@azureml_app.command()
def invoke(
    ctx: typer.Context,
    deployment_id: str,
    input_text: Annotated[Optional[str], typer.Option("--input", "-i")] = None,
    json_file: Annotated[Optional[Path], typer.Option("--json-file")] = None,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", min=1, help="Generation limit for published text-generation tasks")
    ] = 100,
) -> None:
    """Invoke the published scoring route using Microsoft Entra ID."""
    if (input_text is None) == (json_file is None):
        raise typer.BadParameter("Provide exactly one of --input or --json-file")
    with _errors():
        if json_file:
            payload = json.loads(json_file.read_text())
        else:
            deployment = ctx.obj.get_deployment(deployment_id)
            task = deployment.config.get("task")
            if deployment.config.get("image"):  # Existing endpoints from the removed custom runtime.
                if task == "chat-completion":
                    payload = {"messages": [{"role": "user", "content": input_text}], "max_tokens": max_tokens}
                elif task == "feature-extraction":
                    payload = {"input": input_text}
                else:
                    payload = {"prompt": input_text, "max_tokens": max_tokens}
            else:
                payload = {"inputs": input_text}
                if task in {
                    "text-generation",
                    "chat-completion",
                    "text2text-generation",
                    "summarization",
                    "translation",
                }:
                    payload["parameters"] = {"max_new_tokens": max_tokens}
        if not isinstance(payload, dict):
            raise typer.BadParameter("Inference JSON must be an object")
        typer.echo(json.dumps(ctx.obj.invoke(deployment_id, payload), indent=2))


azureml_app.command(name="estimate")(provider_estimate("azureml"))
