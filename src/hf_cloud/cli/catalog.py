"""Compare publication across the Azure ML and current Foundry catalogs."""

import json
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from hf_cloud.core.catalog_availability import (
    MICROSOFT_PROVIDERS,
    CatalogAvailability,
    check_catalog_availability,
)
from hf_cloud.core.exceptions import HFCloudError

console = Console()
STATUS_LABELS = {
    "available": ("Available", "green"),
    "unavailable": ("Not available", "yellow"),
    "unknown": ("Could not verify", "yellow"),
    "unsupported-selection": ("Selection unsupported", "yellow"),
    "incompatible-profile": ("Profile incompatible", "yellow"),
    "no-template": ("No deployment template", "yellow"),
}


def print_catalog_availability(results: dict[str, CatalogAvailability], output: Console = console) -> None:
    if not results:
        return
    table = Table(title="Microsoft catalog availability", show_lines=False)
    table.add_column("Provider", no_wrap=True)
    table.add_column("Catalog status")
    table.add_column("Details")
    for provider, result in results.items():
        label, style = STATUS_LABELS[result.status]
        details = result.reason or (f"Published version {result.model.version}" if result.model else "")
        table.add_row(provider, Text(label, style=style), Text(details))
    output.print(table)
    output.print("Publication checks do not reserve regional capacity or verify account quota.", style="dim")


def catalog(
    model_id: Annotated[str, typer.Argument(help="Hub model ID (org/model)")],
    providers: Annotated[
        Optional[list[str]], typer.Option("--provider", "-p", help="azureml or foundry; repeat to select both")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j")] = False,
) -> None:
    """Show where a model is published across Azure ML and Foundry; requires Azure login."""
    try:
        selected = (
            list(dict.fromkeys(name.lower() for name in providers)) if providers else list(MICROSOFT_PROVIDERS)
        )
        results = check_catalog_availability(model_id, selected)
        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "modelId": model_id,
                        "catalogAvailability": {key: value.to_dict() for key, value in results.items()},
                    },
                    indent=2,
                )
            )
        else:
            console.print(Text(model_id, style="cyan"))
            print_catalog_availability(results)
            for provider, result in results.items():
                if result.status == "available":
                    console.print(Text(f"\n{provider}: {result.to_dict()['catalogUrl']}", style="cyan"))
                    console.print(Text(f"Details: hf-cloud {provider} catalog {model_id}"), style="dim")
        if any(result.status == "unknown" for result in results.values()):
            raise typer.Exit(code=1)
    except HFCloudError as error:
        if json_output:
            typer.echo(json.dumps({"error": error.message, "details": error.details}))
        else:
            console.print(Text(str(error), style="bold red"))
        raise typer.Exit(code=1) from error
