
"""SageMaker utility functions."""

import re
import time
from typing import TYPE_CHECKING, Any, Optional

from huggingface_hub import model_info

from hf_cloud.core.exceptions import ProviderError

if TYPE_CHECKING:
    from sagemaker.serve.model_builder import SchemaBuilder


def get_schema_builder_from_model(model_id: str, task: str) -> "SchemaBuilder":
    """Get a SchemaBuilder for a model based on its task.

    Args:
        model_id: HuggingFace model ID
        task: Pipeline task

    Returns:
        SchemaBuilder instance
    """
    try:
        from sagemaker.serve.model_builder import SchemaBuilder
    except ImportError:
        raise ProviderError(
            "sagemaker",
            "SageMaker SDK is not installed. Install with: pip install --upgrade hf-cloud",
        )

    # Define sample inputs/outputs based on task
    task_schemas = {
        "text-generation": {
            "sample_input": {"inputs": "Hello, I'm a language model,", "parameters": {"max_new_tokens": 10}},
            "sample_output": [
                {"generated_text": "Hello, I'm a language model, and I can help you with a variety of tasks."}
            ],
        },
        "text2text-generation": {
            "sample_input": {"inputs": "Translate to French: Hello, how are you?"},
            "sample_output": [{"generated_text": "Bonjour, comment allez-vous?"}],
        },
        "text-classification": {
            "sample_input": {"inputs": "I love this product!"},
            "sample_output": [{"label": "POSITIVE", "score": 0.99}],
        },
        "token-classification": {
            "sample_input": {"inputs": "My name is John and I live in New York."},
            "sample_output": [[{"entity": "PER", "word": "John", "score": 0.99}]],
        },
        "question-answering": {
            "sample_input": {
                "question": "What is the capital of France?",
                "context": "Paris is the capital of France.",
            },
            "sample_output": {"answer": "Paris", "score": 0.99},
        },
        "fill-mask": {
            "sample_input": {"inputs": "The capital of France is [MASK]."},
            "sample_output": [{"token_str": "Paris", "score": 0.99}],
        },
        "summarization": {
            "sample_input": {"inputs": "This is a long text that needs to be summarized."},
            "sample_output": [{"summary_text": "A short summary."}],
        },
        "translation": {
            "sample_input": {"inputs": "Hello, how are you?"},
            "sample_output": [{"translation_text": "Bonjour, comment allez-vous?"}],
        },
        "feature-extraction": {
            "sample_input": {"inputs": "Hello world"},
            "sample_output": [[0.1, 0.2, 0.3]],
        },
        "sentence-similarity": {
            "sample_input": {"inputs": {"source_sentence": "Hello", "sentences": ["Hi", "Goodbye"]}},
            "sample_output": [0.9, 0.1],
        },
    }

    # Get schema for task, default to text-generation
    schema = task_schemas.get(task, task_schemas["text-generation"])

    return SchemaBuilder(
        sample_input=schema["sample_input"],
        sample_output=schema["sample_output"],
    )


def get_sagemaker_endpoint_name(model_id: str) -> str:
    """Generate a SageMaker endpoint name from a model ID.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Sanitized endpoint name
    """
    # Replace slashes and other invalid characters
    name = model_id.replace("/", "-").replace("_", "-")

    # Add timestamp for uniqueness
    timestamp = int(time.time())
    name = f"{name}-{timestamp}"

    return sanitize_endpoint_name(name)


def get_sagemaker_env_vars(model_id: str, token: Optional[str] = None, task: Optional[str] = None) -> dict[str, str]:
    """Get environment variables for SageMaker deployment.

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


def sanitize_endpoint_name(name: str) -> str:
    """Sanitize endpoint name to comply with SageMaker naming rules.

    SageMaker endpoint names must:
    - Be 1-63 characters
    - Start with a letter or number
    - Contain only letters, numbers, and hyphens

    Args:
        name: Desired endpoint name

    Returns:
        Sanitized endpoint name
    """
    # Replace invalid characters with hyphens
    sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", name)

    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)

    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")

    # Ensure starts with alphanumeric
    if sanitized and not sanitized[0].isalnum():
        sanitized = "hf-" + sanitized

    # Truncate to 63 characters
    sanitized = sanitized[:63]

    # If empty, generate a default name
    if not sanitized:
        sanitized = f"hf-cloud-{int(time.time())}"

    return sanitized


def get_model_config(model_id: str, token: Optional[str] = None) -> dict[str, Any]:
    """Get model configuration from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID
        token: Optional HuggingFace token

    Returns:
        Model configuration dictionary
    """
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, token=token)

        # Determine task
        task = "text-generation"  # Default
        if info.pipeline_tag:
            task = info.pipeline_tag

        return {
            "model_id": model_id,
            "task": task,
            "library_name": getattr(info, "library_name", None),
            "tags": getattr(info, "tags", []),
        }

    except Exception:
        # If we can't get model info, return defaults
        return {
            "model_id": model_id,
            "task": "text-generation",
            "library_name": None,
            "tags": [],
        }


def estimate_instance_type(model_id: str, token: Optional[str] = None) -> str:
    """Estimate appropriate instance type based on model size.

    Args:
        model_id: HuggingFace model ID
        token: Optional HuggingFace token

    Returns:
        Recommended instance type
    """
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, token=token)

        # Get model size in bytes from safetensors
        model_size = 0
        if hasattr(info, "safetensors") and info.safetensors:
            model_size = info.safetensors.get("total", 0)

        # Convert to GB
        model_size_gb = model_size / (1024**3)

        # Recommend instance based on size
        if model_size_gb < 1:
            return "ml.g5.xlarge"  # 24GB GPU
        elif model_size_gb < 7:
            return "ml.g5.2xlarge"  # 24GB GPU
        elif model_size_gb < 15:
            return "ml.g5.4xlarge"  # 24GB GPU
        elif model_size_gb < 40:
            return "ml.g5.12xlarge"  # 96GB GPU (4x24GB)
        else:
            return "ml.g5.48xlarge"  # 192GB GPU (8x24GB)

    except Exception:
        # Default to a mid-size instance
        return "ml.g5.xlarge"


def validate_instance_type(instance_type: str) -> bool:
    """Validate that an instance type is valid for SageMaker.

    Args:
        instance_type: EC2 instance type

    Returns:
        True if valid
    """
    # Common SageMaker instance type patterns
    valid_prefixes = [
        "ml.t2",
        "ml.t3",
        "ml.m4",
        "ml.m5",
        "ml.m6i",
        "ml.c4",
        "ml.c5",
        "ml.c6i",
        "ml.p2",
        "ml.p3",
        "ml.p4d",
        "ml.g4dn",
        "ml.g5",
        "ml.inf1",
        "ml.inf2",
        "ml.trn1",
    ]

    for prefix in valid_prefixes:
        if instance_type.startswith(prefix):
            return True

    return False


def _decode_ansi_logs(log_output: str) -> str:
    """Convert escaped ANSI codes back to actual escape characters.

    CloudWatch logs may contain ANSI codes as #033 (octal) or similar escapes.
    This converts them back so they render properly in the terminal.
    """
    import re

    # Convert #033 (octal escape) to actual escape character
    log_output = log_output.replace("#033", "\x1b")

    # Also handle \\033 and \033 patterns
    log_output = log_output.replace("\\033", "\x1b")

    # Handle unicode escape sequences like \u001b
    log_output = re.sub(r"\\u001[bB]", "\x1b", log_output)

    return log_output
