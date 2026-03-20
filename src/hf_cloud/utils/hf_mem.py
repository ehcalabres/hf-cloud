
"""HF-Mem integration for VRAM estimation."""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

from ..core.exceptions import HFCloudError


class HfMemError(HFCloudError):
    """Error running hf-mem estimation."""

    pass


@dataclass(frozen=True)
class HfMemEstimate:
    """Result from hf-mem VRAM estimation."""

    model_id: str
    required_vram_gb: float
    bytes_count: int
    raw: dict[str, Any]


class InstanceInfo(TypedDict, total=False):
    """Instance type information."""

    instanceType: str
    numGpu: int
    gpuMemGbPerGpu: float
    startupTimeoutSeconds: int
    pricePerHourUsd: float
    acceleratorType: str  # Vertex AI only


@dataclass
class InstanceData:
    """Container for instance data from JSON file."""

    provider: str
    instances: list[InstanceInfo]
    safety_margin_default: float


def _bytes_to_gb(n_bytes: float) -> float:
    """Convert bytes to gigabytes."""
    return n_bytes / (1024.0**3)


def _get_total_vram(instance: InstanceInfo) -> float:
    """Calculate total VRAM for an instance."""
    return instance.get("numGpu", 0) * instance.get("gpuMemGbPerGpu", 0)


def _run_command(
    cmd: list[str], *, env: Optional[dict[str, str]] = None, timeout_s: int = 120
) -> subprocess.CompletedProcess[str]:
    """Run a command and return the result."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )


def run_hf_mem(model_id: str, timeout: int = 120) -> HfMemEstimate:
    """Run hf-mem to estimate VRAM requirements for a model.

    Args:
        model_id: HuggingFace model ID
        timeout: Command timeout in seconds

    Returns:
        HfMemEstimate with VRAM requirements

    Raises:
        HfMemError: If hf-mem fails or returns invalid output
    """
    try:
        result = _run_command(
            ["uvx", "hf-mem", "--json-output", "--model-id", model_id],
            timeout_s=timeout,
        )

        if result.returncode != 0:
            raise HfMemError(
                f"hf-mem failed for model {model_id}",
                details=f"exit code: {result.returncode}\nstderr: {result.stderr or 'none'}",
            )

        raw_output = result.stdout.strip()
        if not raw_output:
            raise HfMemError(
                f"hf-mem returned empty output for {model_id}",
                details=result.stderr or "No stderr available",
            )

        payload = json.loads(raw_output)
        bytes_count = payload.get("bytes_count")

        if bytes_count is None:
            raise HfMemError(
                f"hf-mem output missing 'bytes_count' for {model_id}",
                details=f"Output: {raw_output}",
            )

        required_vram_gb = _bytes_to_gb(float(bytes_count))

        return HfMemEstimate(
            model_id=model_id,
            required_vram_gb=required_vram_gb,
            bytes_count=bytes_count,
            raw=payload,
        )

    except subprocess.TimeoutExpired:
        raise HfMemError(
            f"hf-mem timed out for model {model_id}",
            details=f"Command exceeded {timeout}s timeout",
        )
    except json.JSONDecodeError as e:
        raise HfMemError(
            f"Invalid JSON output from hf-mem for {model_id}",
            details=str(e),
        )
    except FileNotFoundError:
        raise HfMemError(
            "hf-mem not found",
            details="Ensure uvx is installed and hf-mem is available. Run: pip install uv",
        )


def load_instance_data(provider: str) -> InstanceData:
    """Load instance data from JSON file.

    Args:
        provider: Provider name ('sagemaker' or 'vertex')

    Returns:
        InstanceData containing instances and safety margin

    Raises:
        HfMemError: If instance data file not found or invalid
    """
    data_dir = Path(__file__).parent.parent / "data" / "instances"
    json_path = data_dir / f"{provider}.json"

    if not json_path.exists():
        raise HfMemError(
            f"Instance data not found for {provider}",
            details=f"Expected file at {json_path}",
        )

    try:
        with open(json_path) as f:
            data = json.load(f)
            return InstanceData(
                provider=data.get("provider", provider),
                instances=data.get("instances", []),
                safety_margin_default=data.get("safetyMarginDefault", 0.1),
            )
    except json.JSONDecodeError as e:
        raise HfMemError(
            f"Invalid JSON in instance data for {provider}",
            details=str(e),
        )


def find_minimum_instance(
    required_vram_gb: float,
    instances: list[InstanceInfo],
    safety_margin: float = 0.1,
) -> Optional[InstanceInfo]:
    """Find the minimum viable instance for the required VRAM.

    Args:
        required_vram_gb: Required VRAM in GB
        instances: List of available instances
        safety_margin: Fraction of VRAM to keep free (e.g., 0.1 for 10%)

    Returns:
        Minimum viable instance or None if no suitable instance found
    """
    # Calculate required VRAM accounting for safety margin
    # If safety_margin is 0.1, we need VRAM where required <= VRAM * 0.9
    # So VRAM >= required / 0.9
    min_required_vram = required_vram_gb / (1 - safety_margin) if safety_margin < 1 else float("inf")

    # Sort by total VRAM (ascending) to find minimum
    sorted_instances = sorted(instances, key=_get_total_vram)

    for instance in sorted_instances:
        total_vram = _get_total_vram(instance)
        if total_vram >= min_required_vram:
            return instance

    return None


def find_compatible_instances(
    required_vram_gb: float,
    instances: list[InstanceInfo],
    safety_margin: float = 0.1,
) -> list[InstanceInfo]:
    """Find all instances that meet the VRAM requirement.

    Args:
        required_vram_gb: Required VRAM in GB
        instances: List of available instances
        safety_margin: Fraction of VRAM to keep free (e.g., 0.1 for 10%)

    Returns:
        List of compatible instances sorted by total VRAM (ascending)
    """
    # Calculate required VRAM accounting for safety margin
    min_required_vram = required_vram_gb / (1 - safety_margin) if safety_margin < 1 else float("inf")

    compatible = [inst for inst in instances if _get_total_vram(inst) >= min_required_vram]

    return sorted(compatible, key=_get_total_vram)
