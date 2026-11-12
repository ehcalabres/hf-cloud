"""Plans for published Azure ML catalog models, without image or scoring code overrides."""

import json
import re
from typing import Any, Optional

from hf_cloud.core.exceptions import ConfigurationError
from hf_cloud.providers.azure_catalog import PublishedModel, asset_parts

DEFAULT_INSTANCE = "Standard_NC24ads_A100_v4"
CONFIG_KEYS = {"deployment_name", "instance_type", "instance_count", "catalog_version", "hf_token_connection"}


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{1,30}[a-z0-9]", name):
        raise ConfigurationError(
            "Azure ML endpoint/deployment names must be 3–32 lowercase letters, digits or hyphens, starting with a letter and ending with a letter or digit"
        )
    return name


def validate_config(config: dict[str, Any], token: Optional[str] = None) -> None:
    if set(config) - CONFIG_KEYS:
        raise ConfigurationError(
            "Azure ML uses the published catalog runtime. Image, engine, Hub revision and arbitrary serving overrides are not supported; use --catalog-version to pin a published version."
        )
    if token:
        raise ConfigurationError(
            "Use --hf-token-connection for published gated models; raw Hub tokens are not accepted"
        )
    count = config.get("instance_count", 1)
    if type(count) is not int or count < 1:
        raise ConfigurationError("Instance count must be a positive integer")
    instance = config.get("instance_type", DEFAULT_INSTANCE)
    if not isinstance(instance, str) or not re.fullmatch(r"Standard_[A-Za-z0-9_]+", instance):
        raise ConfigurationError("Use an Azure ML VM size such as Standard_NC24ads_A100_v4")
    validate_name(config.get("deployment_name", "main"))
    connection = config.get("hf_token_connection")
    if connection and (not isinstance(connection, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", connection)):
        raise ConfigurationError("Invalid workspace token connection name")


def serving_plan(
    model: PublishedModel, endpoint_name: str, config: dict[str, Any], token: Optional[str] = None
) -> dict:
    validate_config(config, token)
    validate_name(endpoint_name)
    asset_parts(model.asset_id, "models", "HuggingFace")
    deployment_name = config.get("deployment_name", "main")
    connection = config.get("hf_token_connection")
    metadata = {
        "model_asset_id": model.asset_id,
        "catalog_version": model.version,
        "task": model.task or "unknown",
        "serving_mode": "catalog",
    }
    if connection:
        metadata["hf_token_connection"] = connection
    return {
        "endpoint": {
            "name": endpoint_name,
            "auth_mode": "aad_token",
            "tags": {"hf-cloud": "true"},
            "properties": {
                "hf-cloud-deployment": deployment_name,
                "hf-cloud-model-id": model.hub_id,
                **({"enforce_access_to_default_secret_stores": "enabled"} if connection else {}),
            },
        },
        "deployment": {
            "name": deployment_name,
            "endpoint_name": endpoint_name,
            "model": model.asset_id,
            "instance_type": config.get("instance_type", DEFAULT_INSTANCE),
            "instance_count": config.get("instance_count", 1),
            **(
                {
                    "environment_variables": {
                        "HF_TOKEN": "${{azureml://connections/" + connection + "/credentials/HF_TOKEN}}"
                    }
                }
                if connection
                else {}
            ),
            "properties": {"hf-cloud-model-id": model.hub_id, "hf-cloud-config": json.dumps(metadata)},
        },
        "traffic": {deployment_name: 100},
    }
