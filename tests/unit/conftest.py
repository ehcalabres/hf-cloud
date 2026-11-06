"""Shared fake catalogs and GPU assessments for estimate command tests."""

from unittest.mock import Mock

import pytest

from hf_cloud.utils.hf_mem import GpuFit, HfMemEstimate, InstanceData

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
    monkeypatch.setattr("hf_cloud.cli.estimate.list_instance_providers", lambda: list(catalogs))
    monkeypatch.setattr("hf_cloud.cli.estimate.load_instance_data", lambda name: catalogs[name])
    return run, catalogs
