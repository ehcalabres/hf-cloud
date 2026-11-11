"""Pinned hf-mem v1 adapter and instance catalog helpers."""

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

from hf_cloud.core.exceptions import HFCloudError

# The v1 release is still pending. Use the exact reviewed source until its release
# is validated; a moving branch or unversioned uvx package can break the contract.
HF_MEM_REVISION = "2448e7c6457949951495b6dba2f01ac3bebdf36a"
HF_MEM_SOURCE = f"git+https://github.com/alvarobartt/hf-mem.git@{HF_MEM_REVISION}"
HF_MEM_VERSION = "hf-mem@1.0.0"


class HfMemError(HFCloudError):
    """Error running hf-mem estimation."""


@dataclass(frozen=True)
class EstimateOptions:
    """Workload and weight selection passed unchanged to hf-mem."""

    revision: Optional[str] = None
    file: Optional[str] = None
    format: Optional[str] = None
    max_model_len: Optional[int] = None
    batch_size: int = 1
    kv_cache_dtype: str = "auto"
    max_num_batched_tokens: Optional[int] = None

    def validate(self) -> None:
        for name in ("max_model_len", "batch_size", "max_num_batched_tokens"):
            value = getattr(self, name)
            if (name == "batch_size" or value is not None) and (type(value) is not int or value < 1):
                raise HfMemError(f"--{name.replace('_', '-')} must be a positive integer")
        if self.format not in (None, "safetensors", "gguf", "onnx"):
            raise HfMemError("--format must be safetensors, gguf, or onnx")
        if not self.kv_cache_dtype:
            raise HfMemError("--kv-cache-dtype cannot be empty")

    def command_args(self) -> list[str]:
        self.validate()
        args = []
        for name in (
            "revision",
            "file",
            "format",
            "max_model_len",
            "batch_size",
            "kv_cache_dtype",
            "max_num_batched_tokens",
        ):
            value = getattr(self, name)
            if value is not None:
                args.extend([f"--{name.replace('_', '-')}", str(value)])
        return args


@dataclass(frozen=True)
class GpuFit:
    """Upstream's fit assessment for a specific GPU layout."""

    fits: bool
    required_bytes: int
    gpu_count: int
    memory_per_gpu_bytes: int
    needed_gpu_count: Optional[int]
    max_model_len: int


@dataclass(frozen=True)
class HfMemEstimate:
    """Single-model estimate; totals exclude CPU-offloaded weights."""

    model_id: str
    required_vram_gb: float  # GiB; retain the existing Python attribute name.
    bytes_count: int
    raw: dict[str, Any]
    fit: Optional[GpuFit] = None
    warnings: tuple[str, ...] = ()
    cpu_memory_bytes: int = 0


class InstanceInfo(TypedDict, total=False):
    """Instance type information."""

    instanceType: str
    numGpu: int
    gpuMemGbPerGpu: float
    startupTimeoutSeconds: int
    pricePerHourUsd: Optional[float]
    acceleratorType: str
    provider: str
    maxContextLength: int
    modelAssetId: str
    deploymentTemplateId: str
    profilePath: str
    pricingSource: Optional[str]


@dataclass
class InstanceData:
    """Container for a bundled instance catalog."""

    provider: str
    instances: list[InstanceInfo]
    safety_margin_default: float
    last_updated: Optional[str] = None
    pricing_region: Optional[str] = None
    pricing_source: Optional[str] = None


def _bytes_to_gb(n_bytes: float) -> float:
    """Convert bytes to GiB (the historical function name is retained)."""
    return n_bytes / (1024.0**3)


def _get_total_vram(instance: InstanceInfo) -> float:
    """Calculate aggregate GPU memory in GiB."""
    return instance.get("numGpu", 0) * instance.get("gpuMemGbPerGpu", 0)


def _run_command(
    cmd: list[str],
    *,
    env: Optional[dict[str, str]] = None,
    timeout_s: int = 120,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd, env=merged_env, input=input_text, text=True, capture_output=True, timeout=timeout_s
    )


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise HfMemError(f"Invalid hf-mem {name}: expected a non-negative integer")
    return value


def _parse_estimate(model_id: str, payload: Any, require_fit: bool) -> HfMemEstimate:
    if not isinstance(payload, dict):
        raise HfMemError("Invalid memory output from hf-mem: expected a JSON object")
    if payload.get("version") != HF_MEM_VERSION:
        raise HfMemError("Unexpected hf-mem version", details=f"Expected {HF_MEM_VERSION}")
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list) or not all(isinstance(warning, str) for warning in warnings):
        raise HfMemError("Invalid hf-mem warnings: expected a list of strings")
    if "files" in payload:
        files = payload["files"]
        names = (
            [item.get("file") for item in files if isinstance(item, dict)] if isinstance(files, list) else []
        )
        raise HfMemError(
            "Multiple weight files found; select one with --file or a GGUF quantization tag",
            details="Available files: " + ", ".join(str(name) for name in names if name),
        )
    total = _nonnegative_int(payload.get("total"), "total memory bytes")
    offloaded = payload.get("offloaded")
    if offloaded is not None and not isinstance(offloaded, dict):
        raise HfMemError("Invalid hf-mem offloaded memory")
    cpu_memory = _nonnegative_int(offloaded.get("bytes"), "offloaded bytes") if offloaded is not None else 0
    fit = None
    raw_fit = payload.get("fit")
    if raw_fit is not None:
        if not isinstance(raw_fit, dict) or type(raw_fit.get("fits")) is not bool:
            raise HfMemError("Invalid hf-mem GPU fit")
        count = raw_fit.get("gpus")
        count = 1 if count is None else _nonnegative_int(count, "GPU count")
        memory = _nonnegative_int(raw_fit.get("gpu_memory"), "GPU memory bytes")
        needed = raw_fit.get("needed")
        if needed is not None:
            needed = _nonnegative_int(needed, "needed GPU count")
        if count < 1 or memory < 1 or needed == 0:
            raise HfMemError("Invalid hf-mem GPU fit counts or memory")
        fit = GpuFit(
            fits=raw_fit["fits"],
            required_bytes=_nonnegative_int(raw_fit.get("required"), "required GPU bytes"),
            gpu_count=count,
            memory_per_gpu_bytes=memory,
            needed_gpu_count=needed,
            max_model_len=_nonnegative_int(raw_fit.get("max_model_len"), "maximum context length"),
        )
    if require_fit and fit is None:
        raise HfMemError("hf-mem did not return the requested GPU fit")
    return HfMemEstimate(model_id, _bytes_to_gb(total), total, payload, fit, tuple(warnings), cpu_memory)


def run_hf_mem(
    model_id: str,
    timeout: int = 120,
    *,
    options: Optional[EstimateOptions] = None,
    gpu_memory: Optional[str] = None,
    gpu_memory_utilization: float = 0.92,
) -> HfMemEstimate:
    """Run the pinned v1 CLI, inheriting HF_TOKEN and the user's Hub token cache.

    The separate Python 3.10+ environment keeps hf-mem's Python requirement from
    changing HF Cloud's own Python support. A listing is never used as an estimate.
    """
    options = options or EstimateOptions()
    args = options.command_args()
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
        raise HfMemError("GPU memory utilization must be finite and in (0, 1]")
    if gpu_memory is not None:
        args.extend(["--gpu-memory", gpu_memory, "--gpu-memory-utilization", str(gpu_memory_utilization)])
    try:
        result = _run_command(
            [
                "uvx",
                "--python",
                ">=3.10",
                "--from",
                HF_MEM_SOURCE,
                "hf-mem",
                model_id,
                "--output",
                "json",
                *args,
            ],
            timeout_s=timeout,
        )
        if result.returncode != 0:
            raise HfMemError(
                f"hf-mem failed for model {model_id}",
                details=f"exit code: {result.returncode}\nstderr: {result.stderr or 'none'}",
            )
        if not result.stdout.strip():
            raise HfMemError(f"hf-mem returned empty output for {model_id}", details=result.stderr or None)
        return _parse_estimate(model_id, json.loads(result.stdout), require_fit=gpu_memory is not None)
    except subprocess.TimeoutExpired:
        raise HfMemError(
            f"hf-mem timed out for model {model_id}", details=f"Command exceeded {timeout}s timeout"
        ) from None
    except json.JSONDecodeError as error:
        raise HfMemError(f"Invalid JSON output from hf-mem for {model_id}", details=str(error)) from error
    except FileNotFoundError:
        raise HfMemError("hf-mem not found", details="Install uv to provide uvx: pip install uv") from None


def run_hf_mem_batch(
    model_id: str,
    layouts: list[tuple[int, float]],
    *,
    options: EstimateOptions,
    gpu_memory_utilization: float,
    show_all: bool = False,
    timeout: int = 120,
) -> list[HfMemEstimate]:
    """Evaluate ordered GPU layouts in one process with shared model metadata."""
    options.validate()
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
        raise HfMemError("GPU memory utilization must be finite and in (0, 1]")
    if not layouts:
        return []
    request = {
        "version": HF_MEM_VERSION,
        "model": model_id,
        "layouts": layouts,
        "options": asdict(options),
        "utilization": gpu_memory_utilization,
        "show_all": show_all,
    }
    try:
        result = _run_command(
            [
                "uvx",
                "--python",
                ">=3.10",
                "--from",
                HF_MEM_SOURCE,
                "python",
                "-I",  # Avoid sibling utility modules shadowing Python standard libraries.
                str(Path(__file__).with_name("_hf_mem_worker.py")),
            ],
            input_text=json.dumps(request),
            timeout_s=timeout,
        )
        if result.returncode != 0:
            raise HfMemError(f"hf-mem failed for model {model_id}", details=result.stderr or None)
        payload = json.loads(result.stdout)
        if not isinstance(payload, list) or not 0 < len(payload) <= len(layouts):
            raise HfMemError("Invalid hf-mem batch result")
        estimates = [_parse_estimate(model_id, item, require_fit=True) for item in payload]
        for (count, memory), estimate in zip(layouts, estimates):
            if estimate.fit.gpu_count != count or estimate.fit.memory_per_gpu_bytes != int(memory * 1024**3):
                raise HfMemError("hf-mem GPU fit does not match the requested GPU layout")
        if show_all:
            complete = len(estimates) == len(layouts)
        else:
            complete = not any(item.fit.fits for item in estimates[:-1]) and (
                estimates[-1].fit.fits or len(estimates) == len(layouts)
            )
        if not complete:
            raise HfMemError("Incomplete hf-mem batch result")
        return estimates
    except subprocess.TimeoutExpired:
        raise HfMemError(
            f"hf-mem timed out for model {model_id}", details=f"Command exceeded {timeout}s timeout"
        ) from None
    except json.JSONDecodeError as error:
        raise HfMemError("Invalid JSON output from hf-mem batch", details=str(error)) from error
    except FileNotFoundError:
        raise HfMemError("hf-mem not found", details="Install uv to provide uvx: pip install uv") from None


def list_instance_providers() -> list[str]:
    """Discover providers with bundled estimate catalogs without loading cloud SDKs."""
    data_dir = Path(__file__).parent.parent / "data" / "instances"
    return sorted(path.stem for path in data_dir.glob("*.json"))


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
                last_updated=data.get("last_updated"),
                pricing_region=data.get("pricingRegion"),
            )
    except json.JSONDecodeError as e:
        raise HfMemError(
            f"Invalid JSON in instance data for {provider}",
            details=str(e),
        )


def validate_safety_margin(safety_margin: float) -> None:
    """Reject margins that cannot represent a fraction of reserved VRAM."""
    if not math.isfinite(safety_margin) or not 0 <= safety_margin < 1:
        raise HfMemError("Safety margin must be finite and satisfy 0 <= margin < 1")


def _minimum_vram(required_vram_gb: float, safety_margin: float) -> float:
    validate_safety_margin(safety_margin)
    if not math.isfinite(required_vram_gb) or required_vram_gb < 0:
        raise HfMemError("Required VRAM must be finite and non-negative")
    return required_vram_gb / (1 - safety_margin)


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
    min_required_vram = _minimum_vram(required_vram_gb, safety_margin)
    return min(
        (inst for inst in instances if _get_total_vram(inst) >= min_required_vram),
        key=_get_total_vram,
        default=None,
    )


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
    min_required_vram = _minimum_vram(required_vram_gb, safety_margin)

    compatible = [inst for inst in instances if _get_total_vram(inst) >= min_required_vram]

    return sorted(compatible, key=_get_total_vram)
