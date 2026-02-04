"""Vertex AI utility functions."""

import re
import time


def sanitize_endpoint_name(name: str) -> str:
    """Sanitize endpoint name to comply with Vertex AI naming rules.

    Vertex AI endpoint names must:
    - Be 1-128 characters
    - Start with a letter
    - Contain only letters, numbers, underscores, and hyphens

    Args:
        name: Desired endpoint name

    Returns:
        Sanitized endpoint name
    """
    # Replace invalid characters with hyphens
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", name)

    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)

    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")

    # Ensure starts with letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = "hf-" + sanitized

    # Truncate to 128 characters
    sanitized = sanitized[:128]

    # If empty, generate a default name
    if not sanitized:
        sanitized = f"hf-cloud-{int(time.time())}"

    return sanitized


def get_serving_container_uri(location: str = "us-central1", use_gpu: bool = False) -> str:
    """Get the HuggingFace serving container URI for Vertex AI.

    Args:
        location: GCP location/region
        use_gpu: Whether to use the GPU version of the container
    Returns:
        Container image URI
    """
    region_prefix = location.split("-")[0]  # us, europe, asia

    # Map to appropriate region for container registry
    registry_region = "us"
    if region_prefix == "europe":
        registry_region = "europe"
    elif region_prefix == "asia":
        registry_region = "asia"

    # Use the HuggingFace DLC for Vertex AI
    cuda_version = f"{registry_region}-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cu121.2-3.transformers.4-48.ubuntu2204.py311"
    cpu_version = f"{registry_region}-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-3.transformers.4-48.ubuntu2204.py311"

    return cuda_version if use_gpu else cpu_version


def get_vertex_env_vars(model_id: str, token: str | None = None, task: str | None = None) -> dict[str, str]:
    """Get environment variables for Vertex AI deployment.

    Args:
        model_id: HuggingFace model ID
        token: Optional HuggingFace token
        task: Optional task type

    Returns:
        Environment variables dictionary
    """
    env_vars = {
        "HF_MODEL_ID": model_id,
    }

    if token:
        env_vars["HF_TOKEN"] = token

    if task:
        env_vars["HF_TASK"] = task

    return env_vars
