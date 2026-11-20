"""Deployment defaults rank fits by price and honor serving constraints."""

from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from hf_cloud.cli.deployment_selection import choose_instance
from hf_cloud.core.exceptions import ConfigurationError
from hf_cloud.utils.hf_mem import GpuFit, HfMemEstimate, InstanceData

GIB = 1024**3


@pytest.fixture
def candidates(monkeypatch):
    catalog = InstanceData(
        "sagemaker",
        [
            {"instanceType": "too-small", "numGpu": 1, "gpuMemGbPerGpu": 8, "pricePerHourUsd": 0.1},
            {"instanceType": "expensive-single", "numGpu": 1, "gpuMemGbPerGpu": 48, "pricePerHourUsd": 4},
            {"instanceType": "cheap-dual", "numGpu": 2, "gpuMemGbPerGpu": 24, "pricePerHourUsd": 1.5},
            {"instanceType": "unknown-price", "numGpu": 4, "gpuMemGbPerGpu": 80, "pricePerHourUsd": None},
        ],
        0.1,
    )
    monkeypatch.setattr("hf_cloud.cli.deployment_selection.load_instance_data", lambda _: catalog)

    def assess(model, layouts, **kwargs):
        return [
            HfMemEstimate(
                model,
                32,
                32 * GIB,
                {"kv_cache": {"max_model_len": 4096}},
                fit=GpuFit(
                    count * memory * kwargs["gpu_memory_utilization"] >= 32,
                    32 * GIB,
                    count,
                    memory * GIB,
                    1,
                    4096,
                ),
            )
            for count, memory in layouts
        ]

    run = Mock(side_effect=assess)
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem_batch", run)
    return catalog, run


@pytest.mark.parametrize("provider", ["sagemaker", "vertex", "azureml", "foundry"])
def test_cheapest_fit_wins_over_smallest_memory_and_unknown_price(candidates, monkeypatch, provider):
    catalog, run = candidates
    registry = Mock()
    registry.resolve.return_value = NS(asset_id="azureml://registries/HuggingFace/models/model/versions/7")
    monkeypatch.setattr("hf_cloud.cli.deployment_selection.template_catalog", lambda *_: (catalog, []))
    choice = choose_instance(
        "org/model", provider, registry=registry if provider in ("azureml", "foundry") else None
    )
    assert choice.instance["instanceType"] == "cheap-dual"
    run.assert_called_once()
    assert run.call_args.kwargs["gpu_memory_utilization"] == 0.9
    assert run.call_args.kwargs["show_all"] is True
    if provider == "azureml":
        assert choice.instance["modelAssetId"].endswith("/versions/7")


def test_explicit_tensor_parallelism_assesses_only_used_gpus(candidates):
    _, run = candidates
    choice = choose_instance("org/model", "sagemaker", tensor_parallel_size=1)
    assert choice.instance["instanceType"] == "expensive-single"
    assert all(count == 1 for count, _ in run.call_args.args[1])


def test_gpu_count_constraint_filters_vertex_candidates(candidates):
    assert choose_instance("org/model", "vertex", accelerator_count=2).instance["instanceType"] == "cheap-dual"


@pytest.mark.parametrize("unpriced", [False, True])
def test_no_priced_fit_has_no_fallback(candidates, unpriced):
    catalog, _ = candidates
    catalog.instances = [
        dict(item, pricePerHourUsd=None) if unpriced else item
        for item in catalog.instances
        if unpriced or item["instanceType"] == "too-small"
    ]
    with pytest.raises(ConfigurationError, match="No priced compatible"):
        choose_instance("org/model", "sagemaker")


def test_catalog_unavailable_stops_before_estimation(candidates):
    _, run = candidates
    registry = Mock()
    registry.resolve.side_effect = ConfigurationError("Model is not published")
    with pytest.raises(ConfigurationError, match="not published"):
        choose_instance("org/model", "azureml", registry=registry)
    run.assert_not_called()


def test_foundry_template_context_limit_is_respected(candidates, monkeypatch):
    catalog, _ = candidates
    catalog = replace(catalog, instances=[dict(item, maxContextLength=2048) for item in catalog.instances])
    monkeypatch.setattr("hf_cloud.cli.deployment_selection.template_catalog", lambda *_: (catalog, []))
    with pytest.raises(ConfigurationError, match="No priced compatible"):
        choose_instance("org/model", "foundry", registry=Mock())
