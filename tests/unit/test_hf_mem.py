"""Tests for the pinned hf-mem v1 subprocess contract."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from hf_cloud.utils.hf_mem import (
    HF_MEM_SOURCE,
    HF_MEM_VERSION,
    EstimateOptions,
    HfMemError,
    find_compatible_instances,
    find_minimum_instance,
    run_hf_mem,
    run_hf_mem_batch,
)

GIB = 1024**3


@pytest.fixture
def payload():
    return {
        "version": HF_MEM_VERSION,
        "model": "org/model",
        "revision": "main",
        "file": None,
        "format": "safetensors",
        "total": 11 * GIB,
        "weights": {"bytes": 8 * GIB},
        "kv_cache": {"bytes": GIB},
        "overhead": {"bytes": 2 * GIB},
        "components": None,
        "moe": None,
        "offloaded": {"bytes": 3 * GIB},
        "warnings": ["Check host RAM"],
        "fit": {
            "gpu_memory": 24 * GIB,
            "usable": 22 * GIB,
            "required": 17 * GIB,
            "gpus": 4,
            "needed": 1,
            "fits": True,
            "max_model_len": 8192,
        },
    }


@pytest.fixture
def command(monkeypatch, payload):
    run = Mock(side_effect=lambda *args, **kwargs: subprocess.CompletedProcess([], 0, json.dumps(payload), ""))
    monkeypatch.setattr("hf_cloud.utils.hf_mem._run_command", run)
    return run


def test_pinned_v1_command_and_total(command):
    estimate = run_hf_mem("org/model", timeout=42)
    command.assert_called_once_with(
        [
            "uvx",
            "--python",
            ">=3.10",
            "--from",
            "hf-mem==1.0.0",
            "hf-mem",
            "org/model",
            "--output",
            "json",
            "--batch-size",
            "1",
            "--kv-cache-dtype",
            "auto",
        ],
        timeout_s=42,
    )
    assert estimate.bytes_count == 11 * GIB
    assert estimate.required_vram_gb == 11
    assert estimate.cpu_memory_bytes == 3 * GIB
    assert estimate.warnings == ("Check host RAM",)
    # GPU fit includes overhead per device, unlike the general model total.
    assert estimate.fit.required_bytes == 17 * GIB


def test_all_options_and_gpu_layout_are_forwarded(command):
    options = EstimateOptions(
        revision="branch/one",
        file="q4.gguf",
        format="gguf",
        max_model_len=4096,
        batch_size=4,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=2048,
    )
    run_hf_mem("org/model", options=options, gpu_memory="4x24", gpu_memory_utilization=0.8)
    args = command.call_args.args[0]
    for option, value in (
        ("--revision", "branch/one"),
        ("--file", "q4.gguf"),
        ("--format", "gguf"),
        ("--max-model-len", "4096"),
        ("--batch-size", "4"),
        ("--kv-cache-dtype", "fp8"),
        ("--max-num-batched-tokens", "2048"),
        ("--gpu-memory", "4x24"),
        ("--gpu-memory-utilization", "0.8"),
    ):
        assert args[args.index(option) + 1] == value
    assert "--experimental" not in args and "--model-id" not in args


def test_batch_token_is_forwarded_only_through_subprocess_environment(command, payload):
    command.side_effect = lambda *args, **kwargs: subprocess.CompletedProcess([], 0, json.dumps([payload]), "")
    run_hf_mem_batch(
        "org/model", [(4, 24)], options=EstimateOptions(), gpu_memory_utilization=0.9, token="private-token"
    )
    assert command.call_args.kwargs["env"] == {"HF_TOKEN": "private-token"}
    assert "private-token" not in str(command.call_args.args)
    assert "private-token" not in command.call_args.kwargs["input_text"]


@pytest.mark.parametrize("total", [None, -1, True, "100", float("nan"), float("inf"), 1.5, []])
def test_invalid_total_is_rejected(command, payload, total):
    payload["total"] = total
    with pytest.raises(HfMemError, match="total memory"):
        run_hf_mem("org/model")


def test_large_integer_total_stays_exact(command, payload):
    payload["total"] = 9007199254740993
    assert run_hf_mem("org/model").bytes_count == 9007199254740993


@pytest.mark.parametrize("version", [None, "hf-mem@0.6.0", "hf-mem@2.0.0"])
def test_unexpected_version_is_rejected(command, payload, version):
    payload["version"] = version
    with pytest.raises(HfMemError, match="Unexpected hf-mem version"):
        run_hf_mem("org/model")


@pytest.mark.parametrize("format", ["gguf", "onnx"])
def test_listing_requires_explicit_file(command, payload, format):
    payload.clear()
    payload.update(
        version=HF_MEM_VERSION, format=format, files=[{"file": "first"}, {"file": "second"}], warnings=[]
    )
    with pytest.raises(HfMemError, match="select one with --file") as error:
        run_hf_mem("org/model", gpu_memory="1x24")
    assert "first, second" in error.value.details


@pytest.mark.parametrize("format", ["safetensors", "gguf", "onnx"])
def test_single_file_supported_formats(command, payload, format):
    payload.update(format=format, file="selected", offloaded=None, kv_cache=None, overhead=None)
    assert run_hf_mem("org/model").raw["format"] == format
    assert run_hf_mem("org/model").cpu_memory_bytes == 0


@pytest.mark.parametrize("fit", [None, [], {"fits": "yes"}])
def test_missing_or_invalid_requested_fit(command, payload, fit):
    payload["fit"] = fit
    with pytest.raises(HfMemError, match="fit"):
        run_hf_mem("org/model", gpu_memory="1x24")


@pytest.mark.parametrize("warnings", [None, "warning", [123]])
def test_invalid_warnings(command, payload, warnings):
    payload["warnings"] = warnings
    with pytest.raises(HfMemError, match="warnings"):
        run_hf_mem("org/model")


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("max_model_len", 0),
        ("max_num_batched_tokens", -1),
        ("format", "pickle"),
        ("kv_cache_dtype", ""),
    ],
)
def test_invalid_options_fail_before_subprocess(command, field, value):
    options = replace(EstimateOptions(), **{field: value})
    with pytest.raises(HfMemError):
        run_hf_mem("org/model", options=options)
    command.assert_not_called()


@pytest.mark.parametrize("utilization", [0, -1, 1.01, float("nan"), float("inf")])
def test_invalid_utilization(command, utilization):
    with pytest.raises(HfMemError, match="utilization"):
        run_hf_mem("org/model", gpu_memory_utilization=utilization)
    command.assert_not_called()


@pytest.mark.parametrize("output", ["[]", "null", '"text"'])
def test_non_object_json(command, output):
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 0, output, "")
    with pytest.raises(HfMemError, match="JSON object"):
        run_hf_mem("org/model")


@pytest.mark.parametrize("output,message", [("", "empty output"), ("not json", "Invalid JSON")])
def test_invalid_stdout(command, output, message):
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 0, output, "")
    with pytest.raises(HfMemError, match=message):
        run_hf_mem("org/model")


def test_nonzero_exit_preserves_actionable_error(command):
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 1, "", "Use a namespaced Hub ID (org/name)")
    with pytest.raises(HfMemError) as error:
        run_hf_mem("gpt2")
    assert "org/name" in error.value.details


@pytest.mark.parametrize(
    "error,message",
    [(FileNotFoundError(), "not found"), (subprocess.TimeoutExpired("hf-mem", 120), "timed out")],
)
def test_process_failures(command, error, message):
    command.side_effect = error
    with pytest.raises(HfMemError, match=message):
        run_hf_mem("org/model")


@pytest.mark.parametrize("format", ["safetensors", "gguf", "onnx"])
def test_real_upstream_output_fixtures(command, format):
    """Fixtures were produced by the exact pinned commit using synthetic local files."""
    payload = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "hf_mem" / f"v1_{format}_fit.json").read_text()
    )
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
    estimate = run_hf_mem("org/synthetic-model", gpu_memory="4x24" if format == "safetensors" else "1x24")
    assert estimate.bytes_count == payload["total"]
    assert estimate.fit.required_bytes == payload["fit"]["required"]
    assert estimate.raw["format"] == format
    if format == "safetensors":
        assert estimate.fit.required_bytes > estimate.bytes_count
    if format == "onnx":
        assert any("ONNX Runtime overhead" in warning for warning in estimate.warnings)


def test_real_upstream_listing_fixture(command):
    payload = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "hf_mem" / "v1_gguf_listing.json").read_text()
    )
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
    with pytest.raises(HfMemError, match="select one with --file") as error:
        run_hf_mem("org/synthetic-model")
    assert "model-Q4_K_M.gguf" in error.value.details
    assert "model-Q8_0.gguf" in error.value.details


def test_batch_runs_one_isolated_process_and_validates_request(command, payload):
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 0, json.dumps([payload]), "")
    options = EstimateOptions(max_model_len=1024, batch_size=4)
    results = run_hf_mem_batch("org/model", [(4, 24), (8, 80)], options=options, gpu_memory_utilization=0.8)
    command.assert_called_once()
    args = command.call_args.args[0]
    assert args[:5] == ["uvx", "--python", ">=3.10", "--from", HF_MEM_SOURCE]
    assert args[5:7] == ["python", "-I"]
    request = json.loads(command.call_args.kwargs["input_text"])
    assert request["layouts"] == [[4, 24], [8, 80]]
    assert request["utilization"] == 0.8
    assert request["options"]["max_model_len"] == 1024
    assert request["options"]["batch_size"] == 4
    assert len(results) == 1 and results[0].fit.fits


@pytest.mark.parametrize("output", ["{}", "[]", "[{}]", "null"])
def test_batch_rejects_invalid_output(command, output):
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 0, output, "")
    with pytest.raises(HfMemError):
        run_hf_mem_batch("org/model", [(4, 24)], options=EstimateOptions(), gpu_memory_utilization=0.9)


def test_batch_rejects_mismatched_layout(command, payload):
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 0, json.dumps([payload]), "")
    with pytest.raises(HfMemError, match="does not match"):
        run_hf_mem_batch("org/model", [(1, 24)], options=EstimateOptions(), gpu_memory_utilization=0.9)


@pytest.mark.parametrize("show_all,first_fits", [(True, True), (False, False)])
def test_batch_rejects_truncated_results(command, payload, show_all, first_fits):
    payload["fit"]["fits"] = first_fits
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 0, json.dumps([payload]), "")
    with pytest.raises(HfMemError, match="Incomplete"):
        run_hf_mem_batch(
            "org/model",
            [(4, 24), (8, 80)],
            options=EstimateOptions(),
            gpu_memory_utilization=0.9,
            show_all=show_all,
        )


def test_batch_preserves_listing_error(command):
    path = Path(__file__).parents[1] / "fixtures" / "hf_mem" / "v1_gguf_listing.json"
    command.side_effect = None
    command.return_value = subprocess.CompletedProcess([], 0, json.dumps([json.loads(path.read_text())]), "")
    with pytest.raises(HfMemError, match="select one with --file"):
        run_hf_mem_batch("org/model", [(4, 24)], options=EstimateOptions(), gpu_memory_utilization=0.9)


@pytest.mark.parametrize("select", [find_minimum_instance, find_compatible_instances])
@pytest.mark.parametrize("margin", [-0.1, 1, 1.1, float("nan"), float("inf")])
def test_matching_rejects_invalid_margin(select, margin):
    with pytest.raises(HfMemError, match="Safety margin"):
        select(8, [], margin)


@pytest.mark.parametrize("select", [find_minimum_instance, find_compatible_instances])
@pytest.mark.parametrize("required", [-1, float("nan"), float("inf")])
def test_matching_rejects_invalid_required_vram(select, required):
    with pytest.raises(HfMemError, match="Required VRAM"):
        select(required, [])


@pytest.mark.parametrize("required,margin", [(0, 0), (8, 0), (8, 0.5), (16.1, 0.5), (100, 0.1)])
def test_minimum_matches_sorted_compatible_instances(required, margin):
    instances = [
        {"instanceType": "large", "numGpu": 4, "gpuMemGbPerGpu": 8},
        {"instanceType": "first-small", "numGpu": 1, "gpuMemGbPerGpu": 8},
        {"instanceType": "second-small", "numGpu": 2, "gpuMemGbPerGpu": 4},
        {"instanceType": "medium", "numGpu": 2, "gpuMemGbPerGpu": 8},
    ]
    original_order = instances.copy()
    compatible = find_compatible_instances(required, instances, margin)
    minimum = find_minimum_instance(required, instances, margin)
    assert minimum is (compatible[0] if compatible else None)
    assert instances == original_order
    if required == 8 and margin == 0:
        assert minimum["instanceType"] == "first-small"
    if required == 8 and margin == 0.5:
        assert minimum["instanceType"] == "medium"
