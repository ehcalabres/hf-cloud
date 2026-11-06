"""Price ranking and provider comparison contracts, including missing prices."""

import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.estimation import hourly_price
from hf_cloud.utils.hf_mem import EstimateOptions, GpuFit, HfMemError, HfMemEstimate, InstanceData

runner = CliRunner()
GIB = 1024**3


def instance(name, count, memory, price):
    return {
        "instanceType": name,
        "numGpu": count,
        "gpuMemGbPerGpu": memory,
        "pricePerHourUsd": price,
        "acceleratorType": "NVIDIA_A100_80GB",
    }


@pytest.fixture
def comparison(monkeypatch):
    catalogs = {
        "sagemaker": InstanceData(
            "sagemaker",
            [
                instance("sm-fit", 1, 24, 3),
                instance("sm-large", 8, 80, 20),
                instance("cheap-unfit", 1, 8, 0.01),
                instance("unknown", 1, 48, None),
            ],
            0.1,
            "2026-03-16",
        ),
        "vertex": InstanceData(
            "vertex", [instance("vertex-fit", 1, 24, 1), instance("vertex-high", 4, 80, 30)], 0.1, "2025-01-01"
        ),
    }

    def assess(model, layouts, *, options, gpu_memory_utilization, show_all):
        return [
            HfMemEstimate(
                model,
                16,
                16 * GIB,
                {"format": "safetensors", "kv_cache": {"max_model_len": 1024}},
                GpuFit(count * memory * gpu_memory_utilization >= 16, 16 * GIB, count, memory * GIB, 1, 4096),
                ("Check host RAM",),
                3 * GIB,
            )
            for count, memory in layouts
        ]

    run = Mock(side_effect=assess)
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem_batch", run)
    monkeypatch.setattr("hf_cloud.cli.comparison.list_instance_providers", lambda: list(catalogs))
    monkeypatch.setattr("hf_cloud.cli.comparison.load_instance_data", lambda name: catalogs[name])
    return run, catalogs


def invoke(*flags):
    return runner.invoke(app, ["estimate", "org/model", "--json", *flags])


def test_global_ranking_only_includes_fits_and_shares_layout_assessments(comparison):
    run, catalogs = comparison
    result = invoke()
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert [item["instanceType"] for item in data["instances"]] == [
        "vertex-fit",
        "sm-fit",
        "sm-large",
        "vertex-high",
        "unknown",
    ]
    assert data["cheapest"]["provider"] == "vertex"
    assert data["mostExpensive"]["instanceType"] == "vertex-high"
    assert data["totalCompatible"] == 5 and data["pricedCount"] == 4 and data["unpricedCount"] == 1
    assert data["instances"][-1]["pricePerHourUsd"] is None
    assert data["pricing"] == {
        "source": "bundled-catalogs",
        "currency": "USD",
        "unit": "hour",
        "catalogs": {
            "sagemaker": {"lastUpdated": "2026-03-16", "region": None},
            "vertex": {"lastUpdated": "2025-01-01", "region": None},
        },
    }
    run.assert_called_once_with(
        "org/model",
        [(1, 8), (1, 24), (1, 48), (4, 80), (8, 80)],
        options=EstimateOptions(),
        gpu_memory_utilization=0.9,
        show_all=True,
    )
    assert all("provider" not in item for catalog in catalogs.values() for item in catalog.instances)


def test_descending_and_limit_preserve_full_price_range(comparison):
    result = invoke("--sort", "most-expensive", "--limit", "1")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert len(data["instances"]) == 1
    assert data["instances"][0]["instanceType"] == "vertex-high"
    assert data["cheapest"]["instanceType"] == "vertex-fit"
    assert data["totalCompatible"] == 5


def test_provider_filter_and_duplicates(comparison):
    run, _ = comparison
    result = invoke("--provider", "SageMaker", "--provider", "sagemaker")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["providers"] == ["sagemaker"]
    assert all(row["provider"] == "sagemaker" for row in data["instances"])
    assert data["cheapest"]["instanceType"] == "sm-fit"
    assert len(run.call_args.args[1]) == 4


def test_unsupported_provider_is_actionable_without_estimation(comparison):
    run, _ = comparison
    result = invoke("--provider", "azure")
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert "azure" in data["details"] and "vertex" in data["details"]
    run.assert_not_called()


def test_workload_forwarding(comparison):
    run, _ = comparison
    result = invoke(
        "--max-model-len",
        "4096",
        "--batch-size",
        "4",
        "--kv-cache-dtype",
        "fp8",
        "--max-num-batched-tokens",
        "2048",
        "--revision",
        "dev",
        "--file",
        "Q4_K_M",
        "--format",
        "gguf",
        "--safety-margin",
        "0.2",
    )
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["options"] == EstimateOptions("dev", "Q4_K_M", "gguf", 4096, 4, "fp8", 2048)
    assert run.call_args.kwargs["gpu_memory_utilization"] == 0.8


@pytest.mark.parametrize(
    "flags",
    [
        ("--safety-margin", "1"),
        ("--safety-margin", "nan"),
        ("--batch-size", "0"),
        ("--format", "pickle"),
        ("--limit", "0"),
        ("--sort", "invalid"),
    ],
)
def test_invalid_options_do_not_run_model_estimation(comparison, flags):
    run, _ = comparison
    assert invoke(*flags).exit_code != 0
    run.assert_not_called()


@pytest.mark.parametrize("price", [None, -1, float("nan"), float("inf"), True])
def test_unknown_prices_never_win_a_ranking(comparison, price):
    _, catalogs = comparison
    catalogs["vertex"].instances[0]["pricePerHourUsd"] = price
    result = invoke()
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["cheapest"]["instanceType"] == "sm-fit"
    assert data["unpricedCount"] == 2
    assert "NaN" not in result.stdout and "Infinity" not in result.stdout


def test_no_priced_matches_is_explicit(comparison):
    _, catalogs = comparison
    for catalog in catalogs.values():
        for row in catalog.instances:
            row["pricePerHourUsd"] = None
    result = invoke()
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["totalCompatible"] == 5
    assert data["cheapest"] is None and data["mostExpensive"] is None
    assert "No catalog prices" in data["error"]


def test_no_compatible_matches(comparison):
    _, catalogs = comparison
    for catalog in catalogs.values():
        for row in catalog.instances:
            row["gpuMemGbPerGpu"] = 0.1
    result = invoke()
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["instances"] == [] and data["cheapest"] is None
    assert data["error"] == "No compatible instances found"


def test_model_error_is_not_a_partial_price_comparison(comparison):
    run, _ = comparison
    run.side_effect = HfMemError("Metadata unavailable")
    result = invoke()
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == "Metadata unavailable"


def test_human_comparison_displays_extremes_and_price_basis(comparison):
    result = runner.invoke(app, ["estimate", "org/model", "--limit", "2"])
    assert result.exit_code == 0, result.output
    assert "Cheapest" in result.stdout and "Most expensive" in result.stdout
    assert "vertex-fit" in result.stdout and "vertex-high" in result.stdout
    assert "2025-01-01" in result.stdout and "not live quotes" in result.stdout


def test_zero_price_is_distinct_from_missing_price():
    assert hourly_price({"pricePerHourUsd": 0}) == 0
    assert hourly_price({}) is None
