"""GPU fit matching, caching, ordering, and failure behavior."""

from unittest.mock import Mock

import pytest

from hf_cloud.core.estimation import select_instances
from hf_cloud.utils.hf_mem import EstimateOptions, GpuFit, HfMemError, HfMemEstimate

GIB = 1024**3


def instance(name, count, memory):
    return {"instanceType": name, "numGpu": count, "gpuMemGbPerGpu": memory}


def assessment(layout, fits=True):
    count, memory = map(int, layout.split("x"))
    # Deliberately very different from fit.required: selection must trust fit.
    return HfMemEstimate(
        "org/model",
        1,
        GIB,
        {},
        GpuFit(fits, 20 * GIB, count, memory * GIB, 1 if fits else None, 4096),
        ("A warning",),
    )


def test_gpu_topology_overrides_aggregate_vram(monkeypatch):
    run = Mock(
        side_effect=lambda model, **kwargs: assessment(kwargs["gpu_memory"], kwargs["gpu_memory"] == "1x32")
    )
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem", run)
    catalog = [instance("four-small", 4, 8), instance("one-large", 1, 32)]
    result = select_instances("org/model", catalog, 0.1, EstimateOptions())
    assert [match.instance["instanceType"] for match in result.matches] == ["one-large"]
    assert run.call_count == 2
    assert result.matches[0].estimate.fit.required_bytes == 20 * GIB


def test_duplicate_layouts_reuse_fit_and_keep_catalog_order(monkeypatch):
    run = Mock(side_effect=lambda model, **kwargs: assessment(kwargs["gpu_memory"]))
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem", run)
    catalog = [instance("large", 4, 24), instance("first", 1, 24), instance("second", 1, 24)]
    original = catalog.copy()
    options = EstimateOptions(max_model_len=1024, batch_size=4)
    result = select_instances("org/model", catalog, 0.2, options, show_all=True)
    assert [match.instance["instanceType"] for match in result.matches] == ["first", "second", "large"]
    assert result.matches[0].estimate is result.matches[1].estimate
    assert run.call_count == 2
    assert all(call.kwargs["gpu_memory_utilization"] == 0.8 for call in run.call_args_list)
    assert all(call.kwargs["options"] is options for call in run.call_args_list)
    assert result.warnings == ["A warning"]
    assert catalog == original


def test_minimum_stops_at_first_fit_without_applying_margin_twice(monkeypatch):
    run = Mock(side_effect=lambda model, **kwargs: assessment(kwargs["gpu_memory"]))
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem", run)
    # fit.required deliberately exceeds this catalog's aggregate VRAM. Upstream
    # fit.fits is authoritative; the selector must not reapply a scalar threshold.
    result = select_instances(
        "org/model", [instance("small", 1, 8), instance("large", 1, 32)], 0.5, EstimateOptions()
    )
    assert len(result.matches) == 1
    run.assert_called_once_with(
        "org/model", options=EstimateOptions(), gpu_memory="1x8", gpu_memory_utilization=0.5
    )


def test_failures_never_become_silent_partial_recommendations(monkeypatch):
    run = Mock(side_effect=HfMemError("Hub unreachable"))
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem", run)
    with pytest.raises(HfMemError, match="Hub unreachable"):
        select_instances("org/model", [instance("one", 1, 24)], 0.1, EstimateOptions(), show_all=True)


def test_mismatched_fit_is_rejected(monkeypatch):
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem", lambda *args, **kwargs: assessment("4x80"))
    with pytest.raises(HfMemError, match="does not match"):
        select_instances("org/model", [instance("one", 1, 24)], 0.1, EstimateOptions())


@pytest.mark.parametrize("margin", [-0.1, 1, float("nan")])
def test_invalid_margin_never_calls_hf_mem(monkeypatch, margin):
    run = Mock()
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem", run)
    with pytest.raises(HfMemError, match="Safety margin"):
        select_instances("org/model", [], margin, EstimateOptions())
    run.assert_not_called()


def test_empty_catalog_does_not_run_estimation(monkeypatch):
    run = Mock()
    monkeypatch.setattr("hf_cloud.core.estimation.run_hf_mem", run)
    assert select_instances("org/model", [], 0.1, EstimateOptions()).matches == []
    run.assert_not_called()
