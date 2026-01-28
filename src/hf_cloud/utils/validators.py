"""Input validation utilities."""

import re
from typing import Tuple

from ..core.exceptions import ConfigurationError


def validate_model_id(model_id: str) -> Tuple[bool, str]:
    """Validate a HuggingFace model ID.

    Model IDs can be:
    - Simple: 'gpt2', 'bert-base-uncased'
    - Namespaced: 'username/model-name', 'organization/model-name'

    Args:
        model_id: Model ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_id:
        return False, "Model ID cannot be empty"

    if len(model_id) > 200:
        return False, "Model ID is too long (max 200 characters)"

    # Check for valid characters
    # HuggingFace allows: letters, numbers, hyphens, underscores, periods, and slashes
    pattern = r"^[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)?$"
    if not re.match(pattern, model_id):
        return False, "Model ID contains invalid characters"

    # Check for double slashes or leading/trailing special chars
    if "//" in model_id:
        return False, "Model ID cannot contain double slashes"

    if model_id.startswith(("/", "-", "_", ".")):
        return False, "Model ID cannot start with special characters"

    if model_id.endswith(("/", "-", "_", ".")):
        return False, "Model ID cannot end with special characters"

    return True, ""


def validate_deployment_name(name: str, provider: str = "sagemaker") -> Tuple[bool, str]:
    """Validate a deployment name for a specific provider.

    Args:
        name: Deployment name to validate
        provider: Cloud provider name

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Deployment name cannot be empty"

    if provider == "sagemaker":
        # SageMaker endpoint names: 1-63 chars, alphanumeric and hyphens
        if len(name) > 63:
            return False, "Deployment name too long (max 63 characters for SageMaker)"

        if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$", name):
            return (
                False,
                "Deployment name must start and end with alphanumeric, contain only alphanumeric and hyphens",
            )

    elif provider == "azure":
        # Azure ML endpoint names: 3-32 chars, lowercase alphanumeric and hyphens
        if len(name) < 3:
            return False, "Deployment name too short (min 3 characters for Azure)"

        if len(name) > 32:
            return False, "Deployment name too long (max 32 characters for Azure)"

        if not re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", name):
            return (
                False,
                "Deployment name must be lowercase, start and end with alphanumeric",
            )

    elif provider == "gcp":
        # GCP Vertex AI: 1-63 chars, lowercase alphanumeric and hyphens
        if len(name) > 63:
            return False, "Deployment name too long (max 63 characters for GCP)"

        if not re.match(r"^[a-z]([a-z0-9-]*[a-z0-9])?$", name):
            return (
                False,
                "Deployment name must start with lowercase letter, contain only lowercase alphanumeric and hyphens",
            )

    return True, ""


def validate_instance_type(instance_type: str, provider: str = "sagemaker") -> Tuple[bool, str]:
    """Validate an instance type for a specific provider.

    Args:
        instance_type: Instance type to validate
        provider: Cloud provider name

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not instance_type:
        return False, "Instance type cannot be empty"

    if provider == "sagemaker":
        # SageMaker instance types start with 'ml.'
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

        if not any(instance_type.startswith(prefix) for prefix in valid_prefixes):
            return (
                False,
                f"Invalid SageMaker instance type. Must start with one of: {', '.join(valid_prefixes[:5])}...",
            )

    elif provider == "azure":
        # Azure VM sizes typically start with 'Standard_'
        if not instance_type.startswith("Standard_"):
            return False, "Azure VM size must start with 'Standard_'"

    elif provider == "gcp":
        # GCP machine types
        valid_patterns = [
            r"^n1-",
            r"^n2-",
            r"^e2-",
            r"^c2-",
            r"^a2-",
            r"^g2-",
        ]

        if not any(re.match(pattern, instance_type) for pattern in valid_patterns):
            return False, "Invalid GCP machine type"

    return True, ""


def validate_region(region: str, provider: str = "sagemaker") -> Tuple[bool, str]:
    """Validate a region for a specific provider.

    Args:
        region: Region to validate
        provider: Cloud provider name

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not region:
        return False, "Region cannot be empty"

    if provider == "sagemaker":
        # AWS regions
        aws_regions = [
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "eu-central-1",
            "eu-north-1",
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-southeast-1",
            "ap-southeast-2",
            "ap-south-1",
            "sa-east-1",
            "ca-central-1",
        ]

        if region not in aws_regions:
            return False, f"Invalid AWS region. Valid regions include: {', '.join(aws_regions[:5])}..."

    elif provider == "azure":
        # Azure regions
        if not re.match(r"^[a-z]+[0-9]?$", region):
            return False, "Invalid Azure region format"

    elif provider == "gcp":
        # GCP regions
        if not re.match(r"^[a-z]+-[a-z]+[0-9]$", region):
            return False, "Invalid GCP region format (e.g., 'us-central1')"

    return True, ""


def validate_config(config: dict, provider: str) -> None:
    """Validate a deployment configuration.

    Args:
        config: Configuration dictionary
        provider: Cloud provider name

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Validate instance type if provided
    if "instance_type" in config:
        is_valid, error = validate_instance_type(config["instance_type"], provider)
        if not is_valid:
            raise ConfigurationError(error)

    # Validate region if provided
    if "region" in config:
        is_valid, error = validate_region(config["region"], provider)
        if not is_valid:
            raise ConfigurationError(error)

    # Validate instance count
    if "instance_count" in config:
        count = config["instance_count"]
        if not isinstance(count, int) or count < 1:
            raise ConfigurationError("Instance count must be a positive integer")
        if count > 10:
            raise ConfigurationError("Instance count cannot exceed 10")
