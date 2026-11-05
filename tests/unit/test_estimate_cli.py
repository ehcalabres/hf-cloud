"""Contract tests for estimation output shared by both providers."""

import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.utils.hf_mem import HfMemError, HfMemEstimate, InstanceData

runner = CliRunner()


@pytest.fixture
def estimation(monkeypatch):
    instances = [
        {"instanceType": "large", "numGpu": 4, "gpuMemGbPerGpu": 8, "pricePerHourUsd": 2},
        {"instanceType": "small", "numGpu": 1, "gpuMemGbPerGpu": 8, "pricePerHourUsd": 1},
    ]
    for instance in instances:
        instance["acceleratorType"] = "TEST_GPU"
    run = Mock(return_value=HfMemEstimate("org/model", 16, 16 * 1024**3, {}))
    load = Mock(side_effect=lambda provider: InstanceData(provider, instances, 0.2))
    monkeypatch.setattr("hf_cloud.cli.estimate.run_hf_mem", run)
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
    }
    if provider == "vertex":
        instance["acceleratorType"] = "TEST_GPU"
    assert json.loads(result.stdout) == {
        "modelId": "org/model",
        "requiredVramGb": 16,
        "safetyMargin": 0.2,
        "instances" if show_all else "instance": [instance] if show_all else instance,
    }
    run.assert_called_once_with("org/model")
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
    run.return_value = HfMemEstimate("org/model", 100, 100 * 1024**3, {})
    flags = ["--all"] if show_all else []
    result = runner.invoke(app, [provider, "estimate", "org/model", "--json", *flags])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == (
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
