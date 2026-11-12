"""HF-Cloud CLI application."""

import typer

from hf_cloud.cli.azureml import azureml_app
from hf_cloud.cli.estimate import estimate as estimate_command
from hf_cloud.cli.foundry import foundry_app
from hf_cloud.cli.global_commands import list_all_deployments, providers_app
from hf_cloud.cli.sagemaker import sagemaker_app
from hf_cloud.cli.sdk_versions import sdk_versions as sdk_versions_command
from hf_cloud.cli.skills import skills_app
from hf_cloud.cli.vertex import vertex_app

app = typer.Typer(
    name="hf-cloud",
    help="Manage HuggingFace model deployments across cloud providers.",
    no_args_is_help=True,
)


# Register provider command groups
app.add_typer(sagemaker_app, name="sagemaker", help="AWS SageMaker commands")
app.add_typer(vertex_app, name="vertex", help="Google Cloud Vertex AI commands")
app.add_typer(azureml_app, name="azureml", help="Azure ML / Foundry classic: published Hub models on managed endpoints")
app.add_typer(foundry_app, name="foundry", help="Microsoft Foundry: published managed-compute deployments (preview)")
app.add_typer(skills_app, name="skills", help="Manage AI assistant skills")

# Register global commands
app.command(name="estimate")(estimate_command)
app.command(name="ls", help="List all deployments across all providers.")(list_all_deployments)
app.add_typer(providers_app, name="providers", help="Manage provider configurations")
providers_app.command(name="versions")(sdk_versions_command)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
