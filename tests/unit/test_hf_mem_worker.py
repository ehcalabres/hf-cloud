"""Metadata cache semantics without installing hf-mem in HF Cloud's environment."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from hf_cloud.utils._hf_mem_worker import CachedSource


def test_json_is_loaded_once_but_each_caller_gets_a_copy():
    async def check():
        source = SimpleNamespace(json=AsyncMock(return_value={"layers": [1, 2]}))
        cached = CachedSource(source)
        first, second = await asyncio.gather(cached.json("config.json"), cached.json("config.json"))
        first["layers"].append(3)
        assert second == {"layers": [1, 2]}
        assert await cached.json("config.json") == second
        source.json.assert_awaited_once_with("config.json")

    asyncio.run(check())


def test_reads_are_reused_and_memory_is_bounded():
    async def check():
        source = SimpleNamespace(read=AsyncMock(return_value=b"abcd"), close=AsyncMock())
        cached = CachedSource(source)
        cached.MAX_READ_BYTES = 4
        assert await cached.read("one", 0, 4) == b"abcd"
        assert await cached.read("one", 0, 4) == b"abcd"
        assert source.read.await_count == 1
        await cached.read("two", 0, 4)
        assert cached._read_bytes == 4
        await cached.read("one", 0, 4)
        assert source.read.await_count == 3
        await cached.close()
        source.close.assert_not_awaited()

    asyncio.run(check())
