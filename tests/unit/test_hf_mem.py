"""Tests for hf-mem integration."""

import subprocess

import pytest

from hf_cloud.utils.hf_mem import HfMemError, run_hf_mem


def test_run_hf_mem_success(monkeypatch):
    """It should parse bytes_count from hf-mem JSON output."""
    result = subprocess.CompletedProcess(
        args=["uvx", "hf-mem"],
        returncode=0,
        stdout='{"bytes_count": 1073741824}',
        stderr="",
    )
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return result

    monkeypatch.setattr("hf_cloud.utils.hf_mem._run_command", _fake_run)

    estimate = run_hf_mem("gpt2")

    assert captured["cmd"] == [
        "uvx",
        "hf-mem",
        "--model-id",
        "gpt2",
        "--json-output",
        "--experimental",
    ]
    assert estimate.model_id == "gpt2"
    assert estimate.bytes_count == 1073741824
    assert estimate.required_vram_gb == pytest.approx(1.0)


def test_run_hf_mem_success_with_total_memory(monkeypatch):
    """It should parse total_memory when bytes_count is absent."""
    result = subprocess.CompletedProcess(
        args=["uvx", "hf-mem"],
        returncode=0,
        stdout='{"model_id":"foo/bar","memory":1,"total_memory":2147483648}',
        stderr="",
    )
    monkeypatch.setattr("hf_cloud.utils.hf_mem._run_command", lambda *args, **kwargs: result)

    estimate = run_hf_mem("foo/bar")

    assert estimate.model_id == "foo/bar"
    assert estimate.bytes_count == 2147483648
    assert estimate.required_vram_gb == pytest.approx(2.0)


def test_run_hf_mem_success_with_memory_only(monkeypatch):
    """It should parse memory when bytes_count/total_memory are absent."""
    result = subprocess.CompletedProcess(
        args=["uvx", "hf-mem"],
        returncode=0,
        stdout='{"model_id":"foo/bar","memory":3221225472}',
        stderr="",
    )
    monkeypatch.setattr("hf_cloud.utils.hf_mem._run_command", lambda *args, **kwargs: result)

    estimate = run_hf_mem("foo/bar")

    assert estimate.model_id == "foo/bar"
    assert estimate.bytes_count == 3221225472
    assert estimate.required_vram_gb == pytest.approx(3.0)


def test_run_hf_mem_non_zero_exit(monkeypatch):
    """It should raise when hf-mem exits with an error."""
    result = subprocess.CompletedProcess(
        args=["uvx", "hf-mem"],
        returncode=1,
        stdout="",
        stderr="boom",
    )
    monkeypatch.setattr("hf_cloud.utils.hf_mem._run_command", lambda *args, **kwargs: result)

    with pytest.raises(HfMemError, match="hf-mem failed for model gpt2"):
        run_hf_mem("gpt2")


def test_run_hf_mem_timeout(monkeypatch):
    """It should surface timeout errors as HfMemError."""
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="uvx hf-mem", timeout=120)

    monkeypatch.setattr("hf_cloud.utils.hf_mem._run_command", _raise_timeout)

    with pytest.raises(HfMemError, match="hf-mem timed out for model gpt2"):
        run_hf_mem("gpt2")


def test_run_hf_mem_missing_command(monkeypatch):
    """It should surface missing uvx/hf-mem binary as HfMemError."""
    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("uvx not found")

    monkeypatch.setattr("hf_cloud.utils.hf_mem._run_command", _raise_not_found)

    with pytest.raises(HfMemError, match="hf-mem not found"):
        run_hf_mem("gpt2")
