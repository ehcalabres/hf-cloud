"""Match catalog instances using hf-mem's GPU-layout-specific assessments."""

import math
from dataclasses import dataclass
from typing import Optional

from hf_cloud.utils.hf_mem import (
    EstimateOptions,
    HfMemError,
    HfMemEstimate,
    InstanceInfo,
    _get_total_vram,
    run_hf_mem_batch,
    validate_safety_margin,
)


@dataclass(frozen=True)
class InstanceMatch:
    instance: InstanceInfo
    estimate: HfMemEstimate


@dataclass
class InstanceSelection:
    matches: list[InstanceMatch]
    warnings: list[str]
    reference: Optional[HfMemEstimate] = None


def select_instances(
    model_id: str,
    instances: list[InstanceInfo],
    safety_margin: float,
    options: EstimateOptions,
    *,
    show_all: bool = False,
) -> InstanceSelection:
    """Check layouts in ascending total VRAM order; equal capacities keep catalog order.

    All layouts share one subprocess and one model metadata session. Every
    evaluated layout uses upstream's fit calculation, without an aggregate-memory
    prefilter or a second application of the safety margin.
    """
    validate_safety_margin(safety_margin)
    options.validate()
    selection = InstanceSelection([], [])
    ordered = sorted(instances, key=_get_total_vram)
    layouts = []
    for instance in ordered:
        count, memory = instance.get("numGpu", 0), instance.get("gpuMemGbPerGpu", 0)
        if type(count) is not int or count < 1 or not math.isfinite(memory) or memory <= 0:
            raise HfMemError(f"Invalid GPU layout in instance catalog: {instance.get('instanceType')}")
        layout = (count, memory)
        if layout not in layouts:
            layouts.append(layout)
    estimates = (
        run_hf_mem_batch(
            model_id, layouts, options=options, gpu_memory_utilization=1 - safety_margin, show_all=show_all
        )
        if layouts
        else []
    )
    by_layout = dict(zip(layouts, estimates))
    for instance in ordered:
        estimate = by_layout[(instance["numGpu"], instance["gpuMemGbPerGpu"])]
        fit = estimate.fit
        selection.reference = selection.reference or estimate
        for warning in estimate.warnings:
            if warning not in selection.warnings:
                selection.warnings.append(warning)
        if fit.fits:
            selection.matches.append(InstanceMatch(instance, estimate))
            if not show_all:
                break
    return selection
