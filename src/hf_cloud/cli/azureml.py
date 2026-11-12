"""Azure ML managed online endpoints, loading models directly from the Hub."""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from hf_cloud.cli.estimate import provider_estimate
from hf_cloud.core.config import Config
from hf_cloud.core.exceptions import HFCloudError
from hf_cloud.core.state import StateManager
from hf_cloud.providers.azureml import AzureMLProvider
from hf_cloud.providers.azureml.serving import DEFAULT_IMAGE, DEFAULT_INSTANCE, serving_plan
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
def _errors():
    try:
        yield
    except (HFCloudError, OSError, ValueError) as error:
        console.print(Text(f"Error: {error}", style="bold red"))
        raise typer.Exit(code=1) from error


@azureml_app.command()
def deploy(
    ctx: typer.Context,
    model_id: Annotated[str, typer.Argument(help="Hugging Face Hub model ID (org/model)")],
    name: Annotated[str, typer.Option("--name", "-n", help="New endpoint name")],
    instance_type: Annotated[Optional[str], typer.Option("--instance-type", "-i")] = None,
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", min=1)] = 1,
    deployment_name: Annotated[
        str, typer.Option("--deployment-name", help="Component name inside the new endpoint")
    ] = "main",
    task: Annotated[
        str, typer.Option("--task", help="text-generation, chat-completion, or feature-extraction")
    ] = "text-generation",
    image: Annotated[str, typer.Option("--image", help="Tagged vLLM-compatible serving image")] = DEFAULT_IMAGE,
    gpu_count: Annotated[
        Optional[int],
        typer.Option("--gpu-count", min=1, help="Required for VM sizes outside the bundled catalog"),
    ] = None,
    hf_token_connection: Annotated[
        Optional[str],
        typer.Option("--hf-token-connection", help="Workspace secret connection containing HF_TOKEN"),
    ] = None,
    revision: Annotated[Optional[str], typer.Option("--revision")] = None,
    dtype: Annotated[Optional[str], typer.Option("--dtype")] = None,
    safety_margin: Annotated[
        float,
        typer.Option("--safety-margin", min=0, max=1, help="Reserved GPU memory fraction; must be less than 1"),
    ] = 0.1,
    max_model_len: Annotated[Optional[int], typer.Option("--max-model-len", min=1)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 1,
    kv_cache_dtype: Annotated[str, typer.Option("--kv-cache-dtype")] = "auto",
    max_num_batched_tokens: Annotated[Optional[int], typer.Option("--max-num-batched-tokens", min=1)] = None,
    trust_remote_code: Annotated[
        bool,
        typer.Option(
            "--trust-remote-code", help="Allow model repository Python code to execute in the serving container"
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the serving plan without authentication or provisioning")
    ] = False,
) -> None:
    """Deploy a Hub model using vLLM on a dedicated GPU VM."""
    with _errors():
        config = {
            "instance_type": instance_type
            or Config().get_provider_config("azureml").get("default_instance_type")
            or DEFAULT_INSTANCE,
            "instance_count": instance_count,
            "deployment_name": deployment_name,
            "task": task,
            "image": image,
            "gpu_count": gpu_count,
            "hf_token_connection": hf_token_connection,
            "revision": revision,
            "dtype": dtype,
            "safety_margin": safety_margin,
            "max_model_len": max_model_len,
            "batch_size": batch_size,
            "kv_cache_dtype": kv_cache_dtype,
            "max_num_batched_tokens": max_num_batched_tokens,
            "trust_remote_code": trust_remote_code,
        }
        plan = serving_plan(model_id, name, config)
        provider = ctx.obj
        resource_id = provider.client.resource_id(name)
        if dry_run:
            typer.echo(json.dumps({"resourceId": resource_id, **plan}, indent=2))
            return
        check = SdkUpdateCheck("azureml")
        try:
            console.print(
                "Azure ML will build/reuse the serving environment, then provision the endpoint. This can take several minutes."
            )
            with console.status("Deploying Azure ML managed online endpoint..."):
                deployment = provider.deploy(model_id, name, config)
            StateManager().add_deployment(deployment)
            console.print("Deployment completed successfully!", style="bold green")
            console.print(
                Text(
                    f"Endpoint: {deployment.deployment_name}\nStatus: {deployment.status.value}\nScoring URI: {deployment.endpoint_url}"
                )
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
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 100,
) -> None:
    """Invoke the endpoint's configured vLLM route using Microsoft Entra ID."""
    if (input_text is None) == (json_file is None):
        raise typer.BadParameter("Provide exactly one of --input or --json-file")
    with _errors():
        if json_file:
            payload = json.loads(json_file.read_text())
        else:
            task = ctx.obj.get_deployment(deployment_id).config["task"]
            if task == "feature-extraction":
                payload = {"input": input_text}
            elif task == "chat-completion":
                payload = {"messages": [{"role": "user", "content": input_text}], "max_tokens": max_tokens}
            else:
                payload = {"prompt": input_text, "max_tokens": max_tokens}
        if not isinstance(payload, dict):
            raise typer.BadParameter("Inference JSON must be an object")
        typer.echo(json.dumps(ctx.obj.invoke(deployment_id, payload), indent=2))


azureml_app.command(name="estimate")(provider_estimate("azureml"))
