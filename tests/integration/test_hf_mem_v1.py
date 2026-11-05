"""Opt-in checks against the real pinned CLI, using only local model metadata.

Run with HF_CLOUD_TEST_HF_MEM=1. The first run needs Git, uv and dependency
network access; no Hub model download or cloud provider operation is performed.
"""

import json
import os
import struct

import pytest
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.utils.hf_mem import HF_MEM_VERSION, EstimateOptions, InstanceData, run_hf_mem

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
