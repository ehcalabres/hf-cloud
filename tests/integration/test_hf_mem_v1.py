"""Opt-in checks against the real pinned CLI, using only local model metadata.

Run with HF_CLOUD_TEST_HF_MEM=1. The first run needs Git, uv and dependency
network access; no Hub model download or cloud provider operation is performed.
"""

import json
import os
import struct
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.utils.hf_mem import (
    HF_MEM_VERSION,
    EstimateOptions,
    HfMemError,
    InstanceData,
    run_hf_mem,
    run_hf_mem_batch,
)

pytestmark = pytest.mark.skipif(os.environ.get("HF_CLOUD_TEST_HF_MEM") != "1", reason="opt-in real hf-mem CLI")


@pytest.fixture
def local_model(tmp_path):
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 1024,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensors = {"model.embed_tokens.weight": [1000, 256]}
    for layer in range(4):
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = [256, 256]
        tensors[f"model.layers.{layer}.mlp.up_proj.weight"] = [512, 256]
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, shape in tensors.items():
        size = shape[0] * shape[1] * 2
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    with (tmp_path / "model.safetensors").open("wb") as weights:
        weights.write(struct.pack("<Q", len(raw)) + raw)
        weights.truncate(8 + len(raw) + offset)
    return str(tmp_path)


def test_real_workload_and_per_device_overhead(local_model):
    small = run_hf_mem(local_model, options=EstimateOptions(max_model_len=256), gpu_memory="4x24")
    large = run_hf_mem(local_model, options=EstimateOptions(max_model_len=512, batch_size=2), gpu_memory="4x24")
    assert large.raw["version"] == HF_MEM_VERSION
    assert large.raw["kv_cache"]["bytes"] == 4 * small.raw["kv_cache"]["bytes"]
    assert large.fit.gpu_count == 4 and large.fit.fits
    assert large.fit.required_bytes > large.bytes_count
    assert large.fit.max_model_len == 1024


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
def test_real_cli_estimates_for_each_provider(local_model, provider, monkeypatch):
    catalog = InstanceData(
        provider,
        [{"instanceType": "test-instance", "numGpu": 4, "gpuMemGbPerGpu": 24, "acceleratorType": "TEST_GPU"}],
        0.1,
    )
    monkeypatch.setattr("hf_cloud.cli.estimate.load_instance_data", lambda name: catalog)
    result = CliRunner().invoke(
        app,
        [
            provider,
            "estimate",
            local_model,
            "--json",
            "--max-model-len",
            "512",
            "--batch-size",
            "2",
            "--kv-cache-dtype",
            "fp8",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["memoryUnit"] == "GiB"
    assert data["estimatedContextLength"] == 512
    assert data["instance"]["numGpu"] == 4
    assert data["instance"]["requiredVramGb"] == data["requiredVramGb"]
    assert data["workload"]["kv_cache_dtype"] == "fp8"


@pytest.mark.parametrize("batched_tokens", [None, 2048])
def test_batch_matches_standalone_for_each_layout(local_model, batched_tokens):
    layouts = [(1, 24), (4, 24), (8, 80)]
    options = EstimateOptions(max_model_len=512, batch_size=2, max_num_batched_tokens=batched_tokens)
    batched = run_hf_mem_batch(local_model, layouts, options=options, gpu_memory_utilization=0.9, show_all=True)
    for (count, memory), result in zip(layouts, batched):
        single = run_hf_mem(
            local_model, options=options, gpu_memory=f"{count}x{memory}", gpu_memory_utilization=0.9
        )
        assert result.raw == single.raw


def test_batch_loads_hub_metadata_once_across_layouts(local_model, monkeypatch, tmp_path):
    root = Path(local_model)
    requests = Counter()

    class Hub(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            requests[path] += 1
            if path.startswith("/api/models/test/model/tree/"):
                data = json.dumps(
                    [
                        {"type": "file", "path": file.name, "size": file.stat().st_size}
                        for file in root.iterdir()
                        if file.is_file()
                    ]
                ).encode()
                self.send_response(200)
            elif "/resolve/" in path:
                file = root / path.rsplit("/", 1)[-1]
                if not file.is_file():
                    self.send_error(404)
                    return
                data = file.read_bytes()
                requested = self.headers.get("Range")
                if requested:
                    start, end = map(int, requested.removeprefix("bytes=").split("-"))
                    data = data[start : end + 1]
                    self.send_response(206)
                else:
                    self.send_response(200)
            else:
                self.send_error(404)
                return
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "isolated-hf-home"))
    monkeypatch.setenv("HF_TOKEN", "local-test-token")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    try:
        estimates = run_hf_mem_batch(
            "test/model",
            [(1, 24), (4, 24), (8, 80)],
            options=EstimateOptions(),
            gpu_memory_utilization=0.9,
            show_all=True,
        )
        assert len(estimates) == 3
        assert requests == {
            "/api/models/test/model/tree/main": 1,
            "/test/model/resolve/main/config.json": 1,
            "/test/model/resolve/main/model.safetensors": 1,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def gguf_model(tmp_path):
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = {
        "general.architecture": "llama",
        "llama.block_count": 4,
        "llama.context_length": 2048,
        "llama.embedding_length": 256,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": 2,
    }
    entries = b""
    for key, value in metadata.items():
        entries += string(key)
        entries += (
            struct.pack("<I", 8) + string(value) if isinstance(value, str) else struct.pack("<Iq", 11, value)
        )
    tensor = string("blk.0.attn_q.weight") + struct.pack("<IQQIQ", 2, 16, 16, 1, 0)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + entries + tensor
    header += b"\0" * (-len(header) % 32)
    for name in ("model-Q4_K_M.gguf", "model-Q8_0.gguf"):
        (tmp_path / name).write_bytes(header + b"\0" * (16 * 16 * 2))
    return str(tmp_path)


def test_batch_gguf_matches_single_file_and_preserves_listing_errors(gguf_model):
    options = EstimateOptions(file="Q4_K_M", max_model_len=512)
    batched = run_hf_mem_batch(
        gguf_model, [(1, 24), (4, 24)], options=options, gpu_memory_utilization=0.9, show_all=True
    )
    for count, result in zip((1, 4), batched):
        single = run_hf_mem(gguf_model, options=options, gpu_memory=f"{count}x24", gpu_memory_utilization=0.9)
        assert result.raw == single.raw
    with pytest.raises(HfMemError, match="select one with --file") as error:
        run_hf_mem_batch(gguf_model, [(1, 24)], options=EstimateOptions(), gpu_memory_utilization=0.9)
    assert "model-Q4_K_M.gguf" in error.value.details


def test_batch_onnx_preserves_weights_and_warnings(tmp_path):
    def varint(value):
        result = bytearray()
        while value >= 128:
            result.append((value & 127) | 128)
            value >>= 7
        result.append(value)
        return bytes(result)

    def field(number, value):
        if isinstance(value, int):
            return varint(number << 3) + varint(value)
        return varint(number << 3 | 2) + varint(len(value)) + value

    tensor = field(1, 10) + field(1, 2) + field(2, 10) + field(8, b"weights") + field(9, b"\0" * 40)
    (tmp_path / "model.onnx").write_bytes(field(1, 8) + field(7, field(5, tensor)))
    options = EstimateOptions(format="onnx")
    batched = run_hf_mem_batch(
        str(tmp_path), [(1, 24), (4, 24)], options=options, gpu_memory_utilization=0.9, show_all=True
    )
    for count, result in zip((1, 4), batched):
        single = run_hf_mem(
            str(tmp_path), options=options, gpu_memory=f"{count}x24", gpu_memory_utilization=0.9
        )
        assert result.raw == single.raw
        assert any("ONNX Runtime overhead" in warning for warning in result.warnings)
