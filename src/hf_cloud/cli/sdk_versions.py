"""Inspect installed SDKs without initializing providers or requiring credentials."""

import json
from dataclasses import asdict
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from hf_cloud.utils.sdk_versions import PROVIDER_SDKS, check_sdk_versions

console = Console()


def sdk_versions(
    provider: Annotated[
        Optional[str], typer.Option("--provider", "-p", help="Filter by provider (sagemaker, vertex, foundry)")
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Bypass the 24-hour release cache")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
) -> None:
    """Show installed provider SDKs and check stable releases on PyPI."""
    if provider is not None:
        provider = provider.lower()
        if provider not in PROVIDER_SDKS:
            raise typer.BadParameter(f"Choose from: {', '.join(PROVIDER_SDKS)}", param_hint="--provider")
    results = check_sdk_versions([provider] if provider else None, refresh=refresh)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "source": "https://pypi.org",
                    "sdks": [{**asdict(result), "status": result.status} for result in results],
                },
                indent=2,
            )
        )
        return
    table = Table(title="Provider SDK versions")
    for column in ("Provider", "Package", "Installed", "Latest stable", "Status"):
        table.add_column(column)
    for result in results:
        status = result.status.replace("-", " ")
        if result.status == "requires-newer-python":
            status += f" ({result.requires_python})"
        table.add_row(result.provider, result.package, result.installed or "—", result.latest or "—", status)
    console.print(table)
    console.print(
        "[dim]Source: PyPI · cached for 24 hours · --refresh checks again. No packages are changed.[/dim]"
    )
