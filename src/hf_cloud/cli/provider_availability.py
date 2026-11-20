"""Compact provider-wide deployment and estimate summary."""

from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from hf_cloud.cli.catalog import STATUS_LABELS
from hf_cloud.core.catalog_availability import CatalogAvailability

DEPLOYMENT_LABELS = {
    **STATUS_LABELS,
    "catalog-available": ("Catalog available", "green"),
    "runtime-unverified": ("Runtime unverified", "yellow"),
    "unavailable": ("Not in catalog", "yellow"),
}
ESTIMATE_LABELS = {
    "included": "Included",
    "excluded": "Excluded",
    "no-memory-fit": "No memory fit",
    "unpriced": "Price unavailable",
}


def print_provider_availability(
    results: dict[str, dict[str, Any]],
    checks: dict[str, CatalogAvailability],
    output: Console,
    *,
    verbose: bool = False,
) -> None:
    table = Table(title="Provider availability", box=box.SIMPLE_HEAD, header_style="dim", title_style="dim")
    table.add_column("Provider", no_wrap=True)
    table.add_column("Deployment check", no_wrap=True)
    table.add_column("Cost estimate")
    for provider, result in results.items():
        label, style = DEPLOYMENT_LABELS[result["deploymentStatus"]]
        table.add_row(provider, Text(label, style=style), ESTIMATE_LABELS[result["estimateStatus"]])
    output.print(table)
    output.print("Runtime support, access, regional capacity and quota are not verified.", style="dim")
    if verbose:
        for provider, result in results.items():
            output.print(Text(f"{provider}: {result['reason']}", style="dim"))
            if result.get("estimateReason") and result["estimateReason"] != result["reason"]:
                output.print(Text(f"  Estimate: {result['estimateReason']}", style="dim"))
            check = checks.get(provider)
            if check:
                if check.details:
                    output.print(Text(check.details, style="dim"))
                for warning in check.estimate_warnings:
                    output.print(Text(warning, style="dim"))
