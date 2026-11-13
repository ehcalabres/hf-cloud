"""Describe deployment evidence separately from memory fit and estimated prices."""

from pathlib import Path
from typing import Any, Optional

from huggingface_hub.utils import HFValidationError, validate_repo_id

from hf_cloud.core.catalog_availability import CatalogAvailability
from hf_cloud.utils.hf_mem import EstimateOptions


def provider_availability(
    model_id: str,
    providers: list[str],
    options: EstimateOptions,
    checks: dict[str, CatalogAvailability],
    instances: list[dict[str, Any]],
    excluded: dict[str, str],
    weights_format: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Reuse existing evidence; never infer runtime support from a successful memory estimate."""
    hub_selection = not Path(model_id).exists()
    try:
        validate_repo_id(model_id)
    except HFValidationError:
        hub_selection = False
    hub_selection = hub_selection and not (
        options.revision
        or options.file
        or options.format not in (None, "safetensors")
        or weights_format not in (None, "safetensors")
    )
    output = {}
    for provider in providers:
        matches = [row for row in instances if row["provider"] == provider]
        estimate = (
            "excluded"
            if provider in excluded
            else "no-memory-fit"
            if not matches
            else "included"
            if any(row["pricePerHourUsd"] is not None for row in matches)
            else "unpriced"
        )
        check = checks.get(provider)
        if check:
            status = "catalog-available" if check.status == "available" else check.status
            path = "catalog"
            reason = check.reason or (
                f"Published model asset verified in the {check.model.registry} registry."
                if check.model
                else "Microsoft catalog publication checked."
            )
        elif provider in {"sagemaker", "vertex"}:
            path = "hub"
            if hub_selection:
                status = "runtime-unverified"
                runtime = (
                    "SageMaker ModelBuilder's selected runtime"
                    if provider == "sagemaker"
                    else "the configured Vertex Hugging Face container"
                )
                reason = (
                    f"Deploys from the Hub using {runtime}; no provider catalog entry is required. "
                    "Model architecture/task support, container version and Hub access have not been validated."
                )
            else:
                status = "unsupported-selection"
                reason = (
                    "The current deploy command accepts a Hub model ID using default weights; it cannot "
                    "reproduce local paths, URLs, selected revisions/files, or non-Safetensors estimates. "
                    "Memory and cost estimates can still be useful for these selections."
                )
        else:
            path, status, reason = "unknown", "unknown", "No deployment check is implemented for this provider."
        output[provider] = {
            "deploymentStatus": status,
            "deploymentPath": path,
            "estimateStatus": estimate,
            "runtimeVerified": False,
            "reason": reason,
        }
        if provider in excluded:
            output[provider]["estimateReason"] = excluded[provider]
    return output
