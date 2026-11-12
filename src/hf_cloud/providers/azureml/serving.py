"""A Hub-ID-first serving plan, independent of Azure authentication."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from huggingface_hub.utils import HFValidationError, validate_repo_id

from hf_cloud.core.exceptions import ConfigurationError
from hf_cloud.utils.hf_mem import load_instance_data, validate_safety_margin

DEFAULT_IMAGE = "vllm/vllm-openai:v0.32.0"
DEFAULT_INSTANCE = "Standard_NC24ads_A100_v4"
TASK_ROUTES = {
    "text-generation": "/v1/completions",
    "chat-completion": "/v1/chat/completions",
    "feature-extraction": "/v1/embeddings",
}
RUNTIME_DIR = Path(__file__).parent / "runtime"
CONFIG_KEYS = {
    "deployment_name",
    "image",
    "instance_type",
    "instance_count",
    "gpu_count",
    "batch_size",
    "task",
    "safety_margin",
    "dtype",
    "kv_cache_dtype",
    "max_model_len",
    "max_num_batched_tokens",
    "revision",
    "trust_remote_code",
    "hf_token_connection",
}


def validate_name(name: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,30}[a-z0-9]", name):
        raise ConfigurationError(
            "Azure ML endpoint/deployment names must be 3–32 lowercase letters, digits or hyphens, starting with a letter and ending with a letter or digit"
        )
    return name


def serving_plan(
    model_id: str, endpoint_name: str, config: dict[str, Any], token: Optional[str] = None
) -> dict:
    if set(config) - CONFIG_KEYS:
        raise ConfigurationError("Unsupported Azure ML deployment settings; use the documented serving options")
    validate_name(endpoint_name)
    deployment_name = validate_name(config.get("deployment_name", "main"))
    try:
        validate_repo_id(model_id)
    except HFValidationError as error:
        raise ConfigurationError("Provide a Hugging Face model ID", details=str(error)) from error
    if model_id.count("/") != 1:
        raise ConfigurationError("Provide a namespaced Hub model ID, such as Qwen/Qwen3-0.6B")
    if token:
        raise ConfigurationError(
            "Use --hf-token-connection to inject a workspace secret; raw Hub tokens are not stored in Azure ML deployment configuration"
        )
    image = config.get("image", DEFAULT_IMAGE)
    if (
        not isinstance(image, str)
        or not re.fullmatch(r"[A-Za-z0-9._/@:+-]+", image)
        or "://" in image
        or ("@" in image and not re.fullmatch(r"[^@]+@sha256:[a-fA-F0-9]{64}", image))
        or not (":" in image.rsplit("/", 1)[-1])
    ):
        raise ConfigurationError(
            "Use a tagged or digest-pinned vLLM-compatible image, without credentials or whitespace"
        )
    instance = config.get("instance_type", DEFAULT_INSTANCE)
    catalog = next(
        (row for row in load_instance_data("azureml").instances if row["instanceType"] == instance), None
    )
    gpu_count = config.get("gpu_count")
    if gpu_count is None:
        gpu_count = catalog["numGpu"] if catalog else None
    if type(gpu_count) is not int or gpu_count < 1:
        raise ConfigurationError(
            "Unknown GPU layout: provide --gpu-count or select an instance from azureml estimate"
        )
    if catalog and gpu_count != catalog["numGpu"]:
        raise ConfigurationError("--gpu-count must match the selected VM's GPU count")
    count = config.get("instance_count", 1)
    batch = config.get("batch_size", 1)
    if type(count) is not int or count < 1 or type(batch) is not int or batch < 1:
        raise ConfigurationError("Instance count and batch size must be positive integers")
    task = config.get("task", "text-generation")
    if task not in TASK_ROUTES:
        raise ConfigurationError(f"Choose a vLLM serving task: {', '.join(TASK_ROUTES)}")
    margin = config.get("safety_margin", 0.1)
    validate_safety_margin(margin)
    dtype = config.get("dtype") or (
        "half" if catalog and "T4" in catalog.get("acceleratorType", "") else "auto"
    )
    if dtype not in {"auto", "half", "float16", "bfloat16", "float", "float32"}:
        raise ConfigurationError("Invalid model dtype")
    if config.get("kv_cache_dtype", "auto") not in {"auto", "fp8", "fp8_e4m3", "fp8_e5m2", "bfloat16"}:
        raise ConfigurationError("Invalid KV cache dtype")
    args = [
        "--model",
        model_id,
        "--served-model-name",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        str(gpu_count),
        "--gpu-memory-utilization",
        str(1 - margin),
        "--max-num-seqs",
        str(batch),
        "--dtype",
        dtype,
        "--kv-cache-dtype",
        config.get("kv_cache_dtype", "auto"),
    ]
    for option in ("max_model_len", "max_num_batched_tokens"):
        if config.get(option) is not None:
            if type(config[option]) is not int or config[option] < 1:
                raise ConfigurationError(f"{option} must be a positive integer")
            args.extend(["--" + option.replace("_", "-"), str(config[option])])
    if config.get("revision"):
        args.extend(["--revision", config["revision"]])
    if config.get("trust_remote_code"):
        args.append("--trust-remote-code")
    if task == "feature-extraction":
        args.extend(["--runner", "pooling"])
    env = {"HF_MODEL_ID": model_id, "HF_CLOUD_VLLM_ARGS": json.dumps(args)}
    connection = config.get("hf_token_connection")
    if connection:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", connection):
            raise ConfigurationError("Invalid workspace token connection name")
        env["HF_TOKEN"] = "${{azureml://connections/" + connection + "/credentials/HF_TOKEN}}"
    dockerfile = (RUNTIME_DIR / "Dockerfile").read_text().replace("__BASE_IMAGE__", image)
    inference_config = {
        "liveness_route": {"path": "/health", "port": 8000},
        "readiness_route": {"path": "/health", "port": 8000},
        "scoring_route": {"path": TASK_ROUTES[task], "port": 8000},
    }
    fingerprint = hashlib.sha256(
        (
            dockerfile + (RUNTIME_DIR / "start.py").read_text() + json.dumps(inference_config, sort_keys=True)
        ).encode()
    ).hexdigest()[:16]
    return {
        "endpoint": {
            "name": endpoint_name,
            "auth_mode": "aad_token",
            "tags": {"hf-cloud": "true"},
            "properties": {
                "hf-cloud-deployment": deployment_name,
                "hf-cloud-model-id": model_id,
                **({"enforce_access_to_default_secret_stores": "enabled"} if connection else {}),
            },
        },
        "deployment": {
            "name": deployment_name,
            "endpoint_name": endpoint_name,
            "instance_type": instance,
            "instance_count": count,
            "environment_variables": env,
            "properties": {
                "hf-cloud-model-id": model_id,
                "hf-cloud-task": task,
                "hf-cloud-config": json.dumps({**config, "gpu_count": gpu_count, "image": image, "task": task}),
            },
        },
        "environment": {
            "name": "hf-cloud-vllm",
            "version": fingerprint,
            "dockerfile": dockerfile,
            "inference_config": inference_config,
        },
        "traffic": {deployment_name: 100},
    }
