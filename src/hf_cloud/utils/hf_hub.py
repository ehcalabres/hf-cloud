
"""HuggingFace Hub integration utilities."""

from typing import Any, Optional

from hf_cloud.core.exceptions import HFCloudError


def get_model_info(model_id: str, token: Optional[str] = None) -> dict[str, Any]:
    """Get model information from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID
        token: Optional HuggingFace Hub token

    Returns:
        Model information dictionary

    Raises:
        HFCloudError: If model info cannot be retrieved
    """
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, token=token)

        return {
            "model_id": model_id,
            "pipeline_tag": info.pipeline_tag,
            "library_name": getattr(info, "library_name", None),
            "tags": getattr(info, "tags", []),
            "downloads": getattr(info, "downloads", 0),
            "likes": getattr(info, "likes", 0),
            "private": getattr(info, "private", False),
            "gated": getattr(info, "gated", False),
        }

    except ImportError:
        raise HFCloudError(
            "huggingface-hub is not installed",
            details="Install with: pip install huggingface-hub",
        )
    except Exception as e:
        if "401" in str(e) or "unauthorized" in str(e).lower():
            raise HFCloudError(
                f"Access denied for model {model_id}",
                details="This model may be private or gated. Provide a valid --token.",
            )
        if "404" in str(e) or "not found" in str(e).lower():
            raise HFCloudError(
                f"Model {model_id} not found",
                details="Check the model ID and ensure it exists on HuggingFace Hub.",
            )
        raise HFCloudError(f"Failed to get model info: {e}")


def validate_model_exists(model_id: str, token: Optional[str] = None) -> bool:
    """Check if a model exists on HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID
        token: Optional HuggingFace Hub token

    Returns:
        True if model exists
    """
    try:
        get_model_info(model_id, token)
        return True
    except HFCloudError:
        return False


def get_model_task(model_id: str, token: Optional[str] = None) -> str:
    """Get the pipeline task for a model.

    Args:
        model_id: HuggingFace model ID
        token: Optional HuggingFace Hub token

    Returns:
        Pipeline task (e.g., 'text-generation', 'text-classification')
    """
    try:
        info = get_model_info(model_id, token)
        return info.get("pipeline_tag") or "text-generation"
    except HFCloudError:
        return "text-generation"


def download_model_config(
    model_id: str,
    token: Optional[str] = None,
) -> dict[str, Any]:
    """Download and parse model config.json from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID
        token: Optional HuggingFace Hub token

    Returns:
        Model configuration dictionary
    """
    try:
        import json

        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            token=token,
        )

        with open(config_path, "r") as f:
            return json.load(f)

    except Exception:
        return {}


# TODO: Update these estimates with the usage of `hf-mem` library
def estimate_model_size(model_id: str, token: Optional[str] = None) -> int:
    """Estimate model size in bytes.

    Args:
        model_id: HuggingFace model ID
        token: Optional HuggingFace Hub token

    Returns:
        Estimated model size in bytes, or 0 if unknown
    """
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, token=token)

        # Try to get size from safetensors
        if hasattr(info, "safetensors") and info.safetensors:
            return info.safetensors.get("total", 0)

        # Try to get size from siblings (model files)
        if hasattr(info, "siblings"):
            total_size = 0
            for sibling in info.siblings:
                if hasattr(sibling, "size") and sibling.size:
                    total_size += sibling.size
            return total_size

        return 0

    except Exception:
        return 0


# TODO: Refine recommendations based on more model data and provider specific instance types
def get_model_requirements(model_id: str, token: Optional[str] = None) -> dict[str, Any]:
    """Get model resource requirements based on size and type.

    Args:
        model_id: HuggingFace model ID
        token: Optional HuggingFace Hub token

    Returns:
        Dictionary with recommended resources
    """
    size_bytes = estimate_model_size(model_id, token)
    size_gb = size_bytes / (1024**3)

    # Estimate GPU memory needed (model + overhead)
    gpu_memory_needed = size_gb * 1.5  # 50% overhead for inference

    if gpu_memory_needed < 8:
        return {
            "min_gpu_memory_gb": 8,
            "recommended_instance": "ml.g5.xlarge",
            "estimated_model_size_gb": size_gb,
        }
    elif gpu_memory_needed < 16:
        return {
            "min_gpu_memory_gb": 16,
            "recommended_instance": "ml.g5.2xlarge",
            "estimated_model_size_gb": size_gb,
        }
    elif gpu_memory_needed < 24:
        return {
            "min_gpu_memory_gb": 24,
            "recommended_instance": "ml.g5.4xlarge",
            "estimated_model_size_gb": size_gb,
        }
    elif gpu_memory_needed < 48:
        return {
            "min_gpu_memory_gb": 48,
            "recommended_instance": "ml.g5.8xlarge",
            "estimated_model_size_gb": size_gb,
        }
    else:
        return {
            "min_gpu_memory_gb": 96,
            "recommended_instance": "ml.g5.12xlarge",
            "estimated_model_size_gb": size_gb,
        }
