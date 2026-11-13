"""Reusable references to published Foundry model and deployment-template assets."""

import json
import math
import re
from pathlib import Path
from typing import Any

from hf_cloud.core.exceptions import ConfigurationError
from hf_cloud.utils.hf_mem import InstanceData

ACCELERATOR_MEMORY_GIB = {"A100_80GB": 80, "H100_80GB": 80, "H200_141GB": 141, "MI300_192GB": 192}
UPGRADE_OPTIONS = {"NoAutoUpgrade", "OnceNewDefaultVersionAvailable", "OnceCurrentVersionExpired"}


def validate_asset_id(value: str, asset_type: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        rf"azureml://registries/[A-Za-z0-9_-]+/{asset_type}/[A-Za-z0-9_.-]+/(versions|labels)/[A-Za-z0-9_.-]+",
        value,
    ):
        raise ConfigurationError(
            f"Invalid Foundry {asset_type} asset ID",
            details=f"Copy the complete azureml:// registry {asset_type} URI from the Foundry catalog",
        )
    return value


def load_profile(path: Path, model_id: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"Cannot read Foundry profile {path}", details=str(error)) from error
    if not isinstance(data, dict) or data.get("modelId") != model_id:
        raise ConfigurationError("Foundry profile modelId must match the requested Hub model ID")
    validate_asset_id(data.get("modelAssetId"), "models")
    validate_asset_id(data.get("deploymentTemplateId"), "deploymenttemplates")
    if data.get("acceleratorType") not in ACCELERATOR_MEMORY_GIB:
        raise ConfigurationError(f"Profile acceleratorType must be one of: {', '.join(ACCELERATOR_MEMORY_GIB)}")
    for key in ("acceleratorsPerInstance", "maxModelLen"):
        if key in data and (type(data[key]) is not int or data[key] < 1):
            raise ConfigurationError(f"Profile {key} must be a positive integer")
    instance_rate = data.get("pricePerInstanceHourUsd")
    accelerator_rate = data.get("pricePerAcceleratorHourUsd")
    if instance_rate is not None and accelerator_rate is not None:
        raise ConfigurationError(
            "Provide either pricePerInstanceHourUsd or pricePerAcceleratorHourUsd, not both"
        )
    for key in ("pricingSource", "pricingObservedAt", "pricingRegion"):
        if data.get(key) is not None and not isinstance(data[key], str):
            raise ConfigurationError(f"Profile {key} must be a string")
    for key, price in (
        ("pricePerInstanceHourUsd", instance_rate),
        ("pricePerAcceleratorHourUsd", accelerator_rate),
    ):
        if price is not None and (type(price) not in (int, float) or not math.isfinite(price) or price < 0):
            raise ConfigurationError(f"Profile {key} must be a finite nonnegative number")
    if accelerator_rate is not None:
        if "acceleratorsPerInstance" not in data:
            raise ConfigurationError("pricePerAcceleratorHourUsd requires acceleratorsPerInstance")
        try:
            instance_rate = accelerator_rate * data["acceleratorsPerInstance"]
        except OverflowError as error:
            raise ConfigurationError("Calculated Foundry instance rate is too large") from error
        if not math.isfinite(instance_rate):
            raise ConfigurationError("Calculated Foundry instance rate is too large")
        data["pricePerInstanceHourUsd"] = instance_rate
    if instance_rate is not None and (
        not isinstance(data.get("pricingSource"), str) or not data["pricingSource"].strip()
    ):
        raise ConfigurationError("Include pricingSource when supplying a Foundry profile price")
    return data


def profile_catalog(path: Path, model_id: str) -> InstanceData:
    data = load_profile(path, model_id)
    if "acceleratorsPerInstance" not in data or "maxModelLen" not in data:
        raise ConfigurationError(
            "Foundry estimates require acceleratorsPerInstance and maxModelLen in the profile"
        )
    accelerator = data["acceleratorType"]
    return InstanceData(
        provider="foundry",
        instances=[
            {
                "instanceType": data["deploymentTemplateId"].split("/deploymenttemplates/")[1].split("/")[0],
                "acceleratorType": accelerator,
                "numGpu": data["acceleratorsPerInstance"],
                "gpuMemGbPerGpu": ACCELERATOR_MEMORY_GIB[accelerator],
                "pricePerHourUsd": data.get("pricePerInstanceHourUsd"),
                "maxContextLength": data["maxModelLen"],
                "modelAssetId": data["modelAssetId"],
                "deploymentTemplateId": data["deploymentTemplateId"],
                "profilePath": str(path),
                "pricingSource": data.get("pricingSource"),
            }
        ],
        safety_margin_default=0.1,
        last_updated=data.get("pricingObservedAt"),
        pricing_region=data.get("pricingRegion", "GlobalManagedCompute"),
        pricing_source=data.get("pricingSource"),
    )
