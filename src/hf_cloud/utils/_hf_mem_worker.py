"""Isolated batch worker for the published hf-mem release pinned by the adapter.

Executed by uvx's Python, not imported with hf-mem into HF Cloud's environment.
The pinned v1 API has no reusable estimation session. These narrowly scoped
hooks reuse source metadata and Safetensors summaries, while every layout still
runs upstream's unmodified estimation and fit calculations.
"""

import asyncio
import copy
import json
import sys
from collections import OrderedDict
from typing import Any


class CachedSource:
    """One command's source cache; no disk cache or cross-command stale metadata."""

    MAX_READ_BYTES = 64 * 1024**2

    def __init__(self, source: Any):
        self.source = source
        self._json: dict[str, Any] = {}
        self._reads: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
        self._read_bytes = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    async def json(self, path: str) -> Any:
        if path not in self._json:
            self._json[path] = asyncio.create_task(self.source.json(path))
        # Upstream normally returns a fresh JSON object for every read.
        return copy.deepcopy(await self._json[path])

    async def read(self, path: str, start: int, end: int) -> bytes:
        key = (path, start, end)
        if key in self._reads:
            self._reads.move_to_end(key)
            return self._reads[key]
        data = await self.source.read(path, start, end)
        if key not in self._reads and len(data) <= self.MAX_READ_BYTES:
            while self._reads and self._read_bytes + len(data) > self.MAX_READ_BYTES:
                _, evicted = self._reads.popitem(last=False)
                self._read_bytes -= len(evicted)
            self._reads[key] = data
            self._read_bytes += len(data)
        return data

    async def close(self) -> None:
        # arun closes after every assessment; the batch owns the real sources.
        pass


async def estimate_layouts(request: dict[str, Any]) -> list[dict[str, Any]]:
    from hf_mem import __version__, _safetensors, estimate

    if f"hf-mem@{__version__}" != request["version"]:
        raise RuntimeError("Unexpected hf-mem version in batch worker")
    open_source = estimate.open_source
    read_components = _safetensors.read_components
    pool_min_files = _safetensors.POOL_MIN_FILES
    sources: dict[tuple, tuple[CachedSource, Any]] = {}
    summaries: dict[tuple, Any] = {}

    async def open_cached(model, revision=None, token=None, progress=None):
        key = (model, revision, token)
        if key not in sources:
            source, file = await open_source(model, revision, token, progress)
            sources[key] = (CachedSource(source), file)
        return sources[key]

    async def read_cached(source, components, warnings=None, quantization=None):
        key = (
            id(source),
            tuple((name, tuple(paths)) for name, paths in components.items()),
            json.dumps(quantization, sort_keys=True),
        )
        if key not in summaries:
            emitted: list[str] = []
            value = await read_components(source, components, emitted, quantization)
            summaries[key] = (value, emitted)
        value, emitted = summaries[key]
        if warnings is not None:
            warnings.extend(emitted)
        return copy.deepcopy(value)

    estimate.open_source = open_cached
    _safetensors.read_components = read_cached
    _safetensors.POOL_MIN_FILES = 32  # Match the standalone CLI's large-repo behavior.
    results = []
    try:
        for count, memory in request["layouts"]:
            result = await estimate.arun(
                request["model"],
                **request["options"],
                gpu_memory=f"{count}x{memory:g}",
                gpu_memory_utilization=request["utilization"],
            )
            payload = result.to_dict()
            results.append(payload)
            # Return listings to the adapter's existing actionable error handler.
            if "files" in payload or (not request["show_all"] and payload["fit"]["fits"]):
                break
        return results
    finally:
        estimate.open_source = open_source
        _safetensors.read_components = read_components
        _safetensors.POOL_MIN_FILES = pool_min_files
        await asyncio.gather(*(source.source.close() for source, _ in sources.values()), return_exceptions=True)


def main() -> None:
    try:
        request = json.load(sys.stdin)
        results = asyncio.run(estimate_layouts(request))
        print(json.dumps(results))
    except Exception as error:
        print(f"hf-mem: error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
