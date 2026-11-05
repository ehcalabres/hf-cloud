"""Contract tests for estimation output shared by both providers."""

import json
from dataclasses import asdict
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.utils.hf_mem import (
    HF_MEM_VERSION,
    EstimateOptions,
    GpuFit,
    HfMemError,
    HfMemEstimate,
    InstanceData,
)

runner = CliRunner()


def _fake_estimate(model_id, *, options, gpu_memory, gpu_memory_utilization, required=16):
    count, memory = map(int, gpu_memory.split("x"))
    fit = GpuFit(
        required <= count * memory * gpu_memory_utilization,
        required * 1024**3,
        count,
        memory * 1024**3,
        1,
        8192,
    )
    return HfMemEstimate(
        model_id, required, required * 1024**3, {"version": HF_MEM_VERSION, "format": "safetensors"}, fit
    )


@pytest.fixture
def estimation(monkeypatch):
    instances = [
        {"instanceType": "large", "numGpu": 4, "gpuMemGbPerGpu": 8, "pricePerHourUsd": 2},
        {"instanceType": "small", "numGpu": 1, "gpuMemGbPerGpu": 8, "pricePerHourUsd": 1},
    ]
    for instance in instances:
        instance["acceleratorType"] = "TEST_GPU"
    run = Mock(side_effect=_fake_estimate)
    load = Mock(side_effect=lambda provider: InstanceData(provider, instances, 0.2))
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem", run)
    monkeypatch.setattr("hf_cloud.cli.estimate.load_instance_data", load)
    return run, load


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
@pytest.mark.parametrize("show_all", [False, True])
def test_estimate_json_contract(estimation, provider, show_all):
    run, load = estimation
    flags = ["--all"] if show_all else []
    result = runner.invoke(app, [provider, "estimate", "org/model", "--json", *flags])
    assert result.exit_code == 0, result.output
    instance = {
        "instanceType": "large",
        "numGpu": 4,
        "gpuMemGbPerGpu": 8,
        "totalVramGb": 32,
        "usagePercent": 50.0,
        "vramLeftGb": 16.0,
        "pricePerHourUsd": 2,
        "requiredVramGb": 16,
        "neededGpuCount": 1,
        "maxModelLen": 8192,
        "cpuMemoryGb": 0,
        "cpuMemoryBytes": 0,
        "warnings": [],
        "weightsFormat": "safetensors",
    }
    if provider == "vertex":
        instance["acceleratorType"] = "TEST_GPU"
    assert json.loads(result.stdout) == {
        "modelId": "org/model",
        "requiredVramGb": 16,
        "safetyMargin": 0.2,
        "hfMemVersion": HF_MEM_VERSION,
        "memoryUnit": "GiB",
        "workload": asdict(EstimateOptions()),
        "warnings": [],
        "cpuMemoryGb": 0,
        "estimatedContextLength": None,
        "cpuMemoryBytes": 0,
        "instances" if show_all else "instance": [instance] if show_all else instance,
    }
    assert run.call_count == 2
    assert run.call_args.kwargs["gpu_memory"] == "4x8"
    load.assert_called_once_with(provider)


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
def test_margin_override(estimation, provider):
    result = runner.invoke(app, [provider, "estimate", "org/model", "--json", "--safety-margin", "0"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["safetyMargin"] == 0


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
@pytest.mark.parametrize("margin", ["-0.1", "1", "1.1", "nan", "inf"])
def test_invalid_margin_fails_before_subprocess(estimation, provider, margin):
    run, _ = estimation
    result = runner.invoke(app, [provider, "estimate", "org/model", "--json", "--safety-margin", margin])
    assert result.exit_code == 1, result.output
    assert "Safety margin" in json.loads(result.stdout)["error"]
    run.assert_not_called()


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
@pytest.mark.parametrize("show_all", [False, True])
def test_no_suitable_instances_json(estimation, provider, show_all):
    run, _ = estimation
    run.side_effect = lambda *args, **kwargs: _fake_estimate(*args, **kwargs, required=100)
    flags = ["--all"] if show_all else []
    result = runner.invoke(app, [provider, "estimate", "org/model", "--json", *flags])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert {key: data[key] for key in ("error", "instances" if show_all else "instance")} == (
        {"error": "No compatible instances found", "instances": []}
        if show_all
        else {"error": "No suitable instance found", "instance": None}
    )


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
def test_estimation_failure_json(estimation, provider):
    run, _ = estimation
    run.side_effect = HfMemError("Estimation failed", details="test failure")
    result = runner.invoke(app, [provider, "estimate", "org/model", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"error": "Estimation failed", "details": "test failure"}


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
@pytest.mark.parametrize("show_all", [False, True])
def test_human_output(estimation, provider, show_all, monkeypatch):
    # Keep the complete suggested deployment command on screen for assertions.
    monkeypatch.setattr("hf_cloud.cli.estimate.Console.width", property(lambda self: 240))
    flags = ["--all"] if show_all else []
    result = runner.invoke(app, [provider, "estimate", "org/model", *flags])
    assert result.exit_code == 0, result.output
    assert "large" in result.stdout
    assert "32" in result.stdout
    assert "50.0%" in result.stdout
    if show_all:
        assert "Compatible" in result.stdout
        assert "small" not in result.stdout
    else:
        assert f"hf-cloud {provider} deploy org/model" in result.stdout
        assert ("--machine-type" if provider == "vertex" else "--instance-type") in result.stdout
        if provider == "vertex":
            assert "--accelerator-type TEST_GPU --accelerator-count 4" in result.stdout


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
def test_workload_and_weight_options(estimation, provider):
    run, _ = estimation
    result = runner.invoke(
        app,
        [
            provider,
            "estimate",
            "org/model",
            "--json",
            "--revision",
            "branch/one",
            "--file",
            "Q4_K_M",
            "--format",
            "gguf",
            "--max-model-len",
            "4096",
            "--batch-size",
            "4",
            "--kv-cache-dtype",
            "fp8",
            "--max-num-batched-tokens",
            "2048",
        ],
    )
    assert result.exit_code == 0, result.output
    expected = EstimateOptions("branch/one", "Q4_K_M", "gguf", 4096, 4, "fp8", 2048)
    assert run.call_args.kwargs["options"] == expected
    assert json.loads(result.stdout)["workload"] == asdict(expected)


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
@pytest.mark.parametrize(
    "option,value",
    [
        ("--max-model-len", "0"),
        ("--batch-size", "-1"),
        ("--max-num-batched-tokens", "0"),
        ("--format", "pickle"),
    ],
)
def test_invalid_workload_does_not_run_hf_mem(estimation, provider, option, value):
    run, _ = estimation
    result = runner.invoke(app, [provider, "estimate", "org/model", "--json", option, value])
    assert result.exit_code == 1, result.output
    assert "error" in json.loads(result.stdout)
    run.assert_not_called()


@pytest.mark.parametrize("json_output", [True, False])
def test_diagnostics_surface_upstream_warnings_and_cpu_memory(estimation, json_output):
    from dataclasses import replace

    run, _ = estimation

    def estimate(*args, **kwargs):
        base = _fake_estimate(*args, **kwargs)
        return replace(base, warnings=("Check [host] memory",), cpu_memory_bytes=3 * 1024**3)

    run.side_effect = estimate
    result = runner.invoke(app, ["vertex", "estimate", "org/model", *(["--json"] if json_output else [])])
    assert result.exit_code == 0, result.output
    if json_output:
        data = json.loads(result.stdout)
        assert data["warnings"] == ["Check [host] memory"]
        assert data["cpuMemoryGb"] == 3
        assert data["instance"]["warnings"] == data["warnings"]
    else:
        assert "Check [host] memory" in result.stdout
        assert "CPU-offloaded weights: 3.00 GiB" in result.stdout


@pytest.mark.parametrize("flags", [["--revision", "branch/one"], ["--file", "model.safetensors"]])
def test_weight_selection_does_not_suggest_deploying_other_weights(estimation, flags):
    result = runner.invoke(app, ["vertex", "estimate", "org/model", *flags])
    assert result.exit_code == 0, result.output
    assert "Recommended Instance" in result.stdout
    assert "hf-cloud vertex deploy" not in result.stdout


def test_alternate_format_does_not_suggest_an_unsupported_deployment(estimation):
    from dataclasses import replace

    run, _ = estimation

    def gguf(*args, **kwargs):
        estimate = _fake_estimate(*args, **kwargs)
        return replace(estimate, raw={"format": "gguf"})

    run.side_effect = gguf
    result = runner.invoke(app, ["sagemaker", "estimate", "org/model", "--format", "gguf"])
    assert result.exit_code == 0, result.output
    assert "hf-cloud sagemaker deploy" not in result.stdout
