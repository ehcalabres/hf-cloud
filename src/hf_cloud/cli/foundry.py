"""Microsoft Foundry managed compute commands."""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from hf_cloud.cli.deploy_code import foundry_code
from hf_cloud.cli.deployment_selection import announce_selection, choose_instance
from hf_cloud.cli.estimate import provider_estimate
from hf_cloud.core.exceptions import ConfigurationError, HFCloudError
from hf_cloud.core.state import StateManager
from hf_cloud.providers.foundry import FoundryProvider
from hf_cloud.providers.foundry.profiles import load_profile
from hf_cloud.utils.sdk_versions import SdkUpdateCheck

console = Console()
foundry_app = typer.Typer(help="Microsoft Foundry managed compute. Scope options go before the subcommand.")


@foundry_app.callback()
def foundry_scope(
    ctx: typer.Context,
    subscription_id: Annotated[
        Optional[str], typer.Option("--subscription-id", help="Azure subscription ID")
    ] = None,
    resource_group: Annotated[
        Optional[str], typer.Option("--resource-group", "-g", help="Azure resource group")
    ] = None,
    account: Annotated[Optional[str], typer.Option("--account", help="Foundry AIServices account name")] = None,
    endpoint: Annotated[
        Optional[str], typer.Option("--endpoint", help="Foundry HTTPS account base URL")
    ] = None,
) -> None:
    ctx.obj = FoundryProvider(subscription_id, resource_group, account, endpoint)


@contextmanager
def _errors(*, stderr: bool = False):
    try:
        yield
    except (HFCloudError, OSError, ValueError) as error:
        (Console(stderr=True) if stderr else console).print(Text(f"Error: {error}", style="bold red"))
        raise typer.Exit(code=1) from error


@foundry_app.command()
def catalog(
    ctx: typer.Context,
    model_id: str,
    catalog_version: Annotated[Optional[str], typer.Option("--catalog-version")] = None,
) -> None:
    """Resolve a Hub ID, list published templates and show the publisher's default selection."""
    with _errors():
        model = ctx.obj.catalog.resolve(model_id, "foundry", catalog_version)
        output = model.to_dict()
        try:
            output["default_selection"] = ctx.obj.catalog.foundry_template(model)
        except ConfigurationError as error:
            output["selection_required"] = str(error)
        typer.echo(json.dumps(output, indent=2))


@foundry_app.command()
def deploy(
    ctx: typer.Context,
    model_id: Annotated[
        str, typer.Argument(help="Hub model ID or published azureml:// registry model asset URI")
    ],
    name: Annotated[str, typer.Option("--name", "-n", help="Foundry deployment name")],
    model_asset_id: Annotated[
        Optional[str],
        typer.Option(
            "--model-asset-id",
            help="Advanced: explicit published model asset; Hub IDs are resolved automatically",
        ),
    ] = None,
    template: Annotated[
        Optional[str],
        typer.Option(
            "--template", help="Published template name or URI; omitted selects cheapest compatible estimate"
        ),
    ] = None,
    accelerator_type: Annotated[
        Optional[str],
        typer.Option("--accelerator-type", help="A100_80GB, H100_80GB, H200_141GB, or MI300_192GB"),
    ] = None,
    profile: Annotated[
        Optional[Path],
        typer.Option("--profile", help="Reusable JSON profile with published model/template references"),
    ] = None,
    catalog_version: Annotated[
        Optional[str], typer.Option("--catalog-version", help="Pin a published model version")
    ] = None,
    instance_count: Annotated[
        int,
        typer.Option(
            "--instance-count", "-c", min=1, help="Model instances, each using the template's GPU count"
        ),
    ] = 1,
    version_upgrade_option: Annotated[
        str,
        typer.Option(
            "--version-upgrade-option", help="Model upgrade policy (default preserves selected version)"
        ),
    ] = "NoAutoUpgrade",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the SDK request without provisioning; automatic discovery reads the catalog",
        ),
    ] = False,
    print_code: Annotated[
        bool, typer.Option("--print-code", help="Print Python SDK code; catalog discovery never deploys")
    ] = False,
) -> None:
    """Deploy a published model/template to Foundry GlobalManagedCompute."""
    if print_code and dry_run:
        raise typer.BadParameter("Use --print-code separately from --dry-run")
    if profile and any(
        value is not None for value in (model_asset_id, template, accelerator_type, catalog_version)
    ):
        raise typer.BadParameter("Use --profile or explicit model/template/accelerator flags, not both")
    if catalog_version and model_asset_id:
        raise typer.BadParameter("Use --catalog-version with a Hub ID or pin --model-asset-id directly")
    with _errors(stderr=print_code or dry_run):
        selected = load_profile(profile, model_id) if profile else {}
        config = {
            "model_asset_id": model_asset_id or selected.get("modelAssetId"),
            "deployment_template": template or selected.get("deploymentTemplateId"),
            "accelerator_type": accelerator_type or selected.get("acceleratorType"),
            "instance_count": instance_count,
            "catalog_version": catalog_version,
            "version_upgrade_option": version_upgrade_option,
        }
        provider = ctx.obj
        scope = provider.client.scope(name)
        if not (profile or template or model_asset_id or model_id.startswith("azureml://")):
            typer.echo("Estimating compatible Foundry templates...", err=True)
            match = choose_instance(
                model_id,
                "foundry",
                registry=provider.catalog,
                catalog_version=catalog_version,
                accelerator_type=accelerator_type,
            )
            announce_selection(match)
            config.update(
                {
                    "model_asset_id": match.instance["modelAssetId"],
                    "deployment_template": match.instance["deploymentTemplateId"],
                    "accelerator_type": match.instance["acceleratorType"],
                }
            )
        config = provider.prepare_config(model_id, config)
        resource = provider.deployment_resource(model_id, config)
        if print_code:
            typer.echo(foundry_code(resource, scope, provider.client))
            return
        if dry_run:
            typer.echo(
                json.dumps(
                    {"resourceId": provider.client.resource_id(name), "scope": scope, "resource": resource},
                    indent=2,
                )
            )
            return
        console.print(Text(f"Published model: {config['model_asset_id']}"))
        console.print(Text(f"Published template: {config['deployment_template']}"))
        console.print(
            "Foundry managed compute is in preview. Microsoft supplies the serving runtime.", style="dim"
        )
        check = SdkUpdateCheck("foundry")
        try:
            with console.status("Provisioning Foundry managed compute..."):
                deployment = provider.deploy(model_id, name, config)
            StateManager().add_deployment(deployment)
            console.print("Deployment completed successfully!", style="bold green")
            console.print(
                Text(
                    f"Deployment: {deployment.deployment_name}\nStatus: {deployment.status.value}\nEndpoint: {deployment.endpoint_url}"
                )
            )
        finally:
            check.notify(console)


@foundry_app.command(name="ls")
def list_deployments(ctx: typer.Context) -> None:
    """List managed compute deployments in the selected Foundry account."""
    with _errors():
        deployments = ctx.obj.list_deployments()
        table = Table(title="Foundry managed compute deployments")
        for column in ("Name", "Model asset", "Accelerator", "Instances", "Status"):
            table.add_column(column)
        for deployment in deployments:
            table.add_row(
                deployment.deployment_name,
                deployment.model_id,
                deployment.instance_type,
                str(deployment.instance_count),
                deployment.status.value,
            )
        console.print(table)


@foundry_app.command()
def describe(ctx: typer.Context, deployment_id: str) -> None:
    """Show a deployment by name or account-scoped ARM resource ID."""
    with _errors():
        typer.echo(json.dumps(ctx.obj.get_deployment(deployment_id).to_dict(), indent=2))


@foundry_app.command()
def status(ctx: typer.Context, deployment_id: str) -> None:
    """Read actual Foundry provisioning status."""
    with _errors():
        deployment = ctx.obj.get_deployment(deployment_id)
        console.print(Text(f"Status: {deployment.status.value}"))
        if deployment.config.get("provisioning_details"):
            console.print(deployment.config["provisioning_details"])
        if deployment.status.value == "failed":
            raise typer.Exit(code=1)


@foundry_app.command()
def update(
    ctx: typer.Context,
    deployment_id: str,
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", min=1)],
) -> None:
    """Scale model instances while preserving the selected template and model."""
    with _errors():
        deployment = ctx.obj.update_deployment(deployment_id, {"instance_count": instance_count})
        state = StateManager()
        previous = state.get_deployment(deployment.deployment_id, provider="foundry", region=deployment.region)
        if previous and previous.get("config", {}).get("model_asset_id") == deployment.config.get(
            "model_asset_id"
        ):
            deployment.model_id = previous["model_id"]
        state.add_deployment(deployment)
        console.print(
            Text(f"Deployment running with {deployment.instance_count} model instances."), style="green"
        )


@foundry_app.command()
def delete(
    ctx: typer.Context,
    deployment_id: str,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip deletion confirmation")] = False,
) -> None:
    """Delete only this managed compute deployment and release its GPU allocation."""
    if not yes:
        typer.confirm(f"Delete Foundry deployment {deployment_id}?", abort=True)
    with _errors():
        deployment = ctx.obj.get_deployment(deployment_id)
        if ctx.obj.delete_deployment(deployment_id):
            StateManager().remove_deployment(
                deployment.deployment_id, provider="foundry", region=deployment.region
            )
            console.print("Deployment deleted.", style="green")


@foundry_app.command()
def invoke(
    ctx: typer.Context,
    deployment_id: str,
    input_text: Annotated[
        Optional[str], typer.Option("--input", "-i", help="User message for chat completions")
    ] = None,
    json_file: Annotated[
        Optional[Path],
        typer.Option("--json-file", help="JSON request for chat, embeddings, or another model route"),
    ] = None,
    route: Annotated[
        str, typer.Option("--route", help="Relative managed-deployment route")
    ] = "chat/completions",
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 100,
) -> None:
    """Invoke with Microsoft Entra ID; OpenAI routes use the deployment name as model."""
    if (input_text is None) == (json_file is None):
        raise typer.BadParameter("Provide exactly one of --input or --json-file")
    if input_text is not None and route.removeprefix("v1/") != "chat/completions":
        raise typer.BadParameter("Use --json-file for non-chat routes")
    with _errors():
        payload = (
            json.loads(json_file.read_text())
            if json_file
            else {
                "messages": [{"role": "user", "content": input_text}],
                "max_tokens": max_tokens,
            }
        )
        if not isinstance(payload, dict):
            raise typer.BadParameter("Inference JSON must be an object")
        typer.echo(json.dumps(ctx.obj.invoke_route(deployment_id, payload, route), indent=2))


@foundry_app.command()
def logs(
    ctx: typer.Context,
    deployment_id: str,
    tail: Annotated[int, typer.Option("--tail", "-n", min=1, max=1000)] = 100,
) -> None:
    """Show Azure Monitor management activity for this deployment over the last 24 hours."""
    with _errors():
        output = ctx.obj.get_logs(deployment_id, tail)
        typer.echo(output or "No deployment management activity in the last 24 hours.")
        console.print(
            "Runtime metrics and diagnostic logs are available through Foundry/Azure Monitor.", style="dim"
        )


foundry_app.command(name="estimate")(provider_estimate("foundry"))
