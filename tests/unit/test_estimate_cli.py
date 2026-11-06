"""Provider estimate shortcuts share the canonical command's behavior and options."""

import json

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.utils.hf_mem import HfMemError

runner = CliRunner()


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--sort", "most-expensive"],
        ["--sort", "most-expensive", "--limit", "1"],
        [
            "--max-model-len",
            "4096",
            "--batch-size",
            "4",
            "--kv-cache-dtype",
            "fp8",
            "--max-num-batched-tokens",
            "2048",
            "--safety-margin",
            "0.2",
        ],
        ["--revision", "dev", "--file", "Q4_K_M", "--format", "gguf"],
        ["--all"],
    ],
)
def test_provider_shortcut_matches_filtered_canonical_json(comparison, provider, flags):
    canonical = runner.invoke(app, ["estimate", "org/model", "--provider", provider, "--json", *flags])
    shortcut = runner.invoke(app, [provider, "estimate", "org/model", "--json", *flags])
    assert canonical.exit_code == shortcut.exit_code == 0, shortcut.output
    assert json.loads(shortcut.stdout) == json.loads(canonical.stdout)
    assert json.loads(shortcut.stdout)["providers"] == [provider]


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
def test_provider_shortcut_matches_human_output(comparison, provider):
    canonical = runner.invoke(app, ["estimate", "org/model", "--provider", provider])
    shortcut = runner.invoke(app, [provider, "estimate", "org/model"])
    assert canonical.exit_code == shortcut.exit_code == 0
    assert shortcut.stdout == canonical.stdout
    assert "Catalog price range" in shortcut.stdout


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
def test_default_is_price_order_and_shared_reserve(comparison, provider):
    run, catalogs = comparison
    # The minimum-memory instance costs more than the larger instance. Catalog
    # default margins differ too; shortcuts must use the canonical default.
    catalogs[provider].safety_margin_default = 0.3
    catalogs[provider].instances[0]["pricePerHourUsd"] = 100
    result = runner.invoke(app, [provider, "estimate", "org/model", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    prices = [row["pricePerHourUsd"] for row in data["instances"] if row["pricePerHourUsd"] is not None]
    assert prices == sorted(prices)
    assert data["cheapest"]["instanceType"] != catalogs[provider].instances[0]["instanceType"]
    assert data["safetyMargin"] == 0.1
    assert run.call_args.kwargs["gpu_memory_utilization"] == 0.9


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
@pytest.mark.parametrize(
    "flags",
    [
        ["--safety-margin", "nan"],
        ["--batch-size", "0"],
        ["--format", "pickle"],
        ["--limit", "0"],
        ["--sort", "invalid"],
    ],
)
def test_invalid_options_share_the_same_errors(comparison, provider, flags):
    run, _ = comparison
    canonical = runner.invoke(app, ["estimate", "org/model", "--provider", provider, "--json", *flags])
    shortcut = runner.invoke(app, [provider, "estimate", "org/model", "--json", *flags])
    assert canonical.exit_code == shortcut.exit_code != 0
    if canonical.exit_code == 1:
        assert json.loads(shortcut.stdout) == json.loads(canonical.stdout)
    run.assert_not_called()


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
def test_shortcut_provider_cannot_be_overridden(comparison, provider):
    run, _ = comparison
    result = runner.invoke(app, [provider, "estimate", "org/model", "--provider", "vertex"])
    assert result.exit_code == 2
    run.assert_not_called()


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
def test_model_failure_shares_the_same_json(comparison, provider):
    run, _ = comparison
    run.side_effect = HfMemError("Metadata unavailable", details="test failure")
    canonical = runner.invoke(app, ["estimate", "org/model", "--provider", provider, "--json"])
    shortcut = runner.invoke(app, [provider, "estimate", "org/model", "--json"])
    assert canonical.exit_code == shortcut.exit_code == 1
    assert json.loads(shortcut.stdout) == json.loads(canonical.stdout)


def test_option_definitions_and_defaults_cannot_drift():
    root = get_command(app)
    canonical = {param.name: param for param in root.commands["estimate"].params if param.name != "providers"}
    for provider in ("sagemaker", "vertex"):
        shortcut = {param.name: param for param in root.commands[provider].commands["estimate"].params}
        assert shortcut.keys() == canonical.keys()
        for name, param in canonical.items():
            assert shortcut[name].default == param.default
            assert shortcut[name].opts == param.opts
            assert str(shortcut[name].type) == str(param.type)
