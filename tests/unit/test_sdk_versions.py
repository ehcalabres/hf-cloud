"""SDK checks remain advisory, cached, and independent of cloud SDK imports."""

import importlib
import io
import json
from dataclasses import asdict
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.deployment import DeploymentStatus
from hf_cloud.utils import sdk_versions as sdk


@pytest.fixture
def releases(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_CLOUD_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(sdk, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sdk, "version", lambda package: "3.4.0")
    monkeypatch.setattr(sdk.time, "time", lambda: 100_000)
    fetch = Mock(return_value={"latest": "3.10.0", "requires_python": ">=3.9"})
    monkeypatch.setattr(sdk, "_latest_release", fetch)
    return fetch


@pytest.mark.parametrize(
    "installed,latest,requires_python,status",
    [
        ("3.9.0", "3.10.0", None, "update-available"),
        ("3.10.0", "3.9.0", None, "up-to-date"),
        ("3.10.0", "3.10.0", None, "up-to-date"),
        ("3.10.0rc1", "3.10.0", None, "update-available"),
        (None, None, None, "not-installed"),
        ("3.4.0", None, None, "check-unavailable"),
        ("3.4.0", "invalid", None, "check-unavailable"),
        ("3.4.0", "3.10.0", ">=100", "requires-newer-python"),
    ],
)
def test_version_comparison(installed, latest, requires_python, status):
    assert sdk.SdkVersion("sagemaker", "sagemaker", installed, latest, requires_python).status == status


@pytest.mark.parametrize(
    "release,yanked", [("3.10.0", False), ("4.0.0rc1", False), ("4.0.0.dev1", False), ("3.10.0", True)]
)
def test_pypi_metadata_and_release_filtering(monkeypatch, release, yanked):
    payload = {"info": {"version": release, "requires_python": ">=3.10"}, "urls": [{"yanked": yanked}]}
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    open_url = Mock(return_value=response)
    monkeypatch.setattr(sdk, "urlopen", open_url)
    if yanked or "rc" in release or "dev" in release:
        with pytest.raises(ValueError, match="stable, non-yanked"):
            sdk._latest_release("sagemaker")
    else:
        assert sdk._latest_release("sagemaker") == {"latest": release, "requires_python": ">=3.10"}
    request = open_url.call_args.args[0]
    assert request.full_url == "https://pypi.org/pypi/sagemaker/json"
    assert open_url.call_args.kwargs["timeout"] == sdk.REQUEST_TIMEOUT


def test_checks_cache_results_until_expiry_and_refresh_bypasses_cache(releases, monkeypatch):
    first = sdk.check_sdk_versions(["sagemaker"])
    assert releases.call_count == 2
    assert sdk.check_sdk_versions(["sagemaker"]) == first
    assert releases.call_count == 2
    sdk.check_sdk_versions(["sagemaker"], refresh=True)
    assert releases.call_count == 4
    monkeypatch.setattr(sdk.time, "time", lambda: 100_000 + sdk.CACHE_TTL)
    sdk.check_sdk_versions(["sagemaker"])
    assert releases.call_count == 6


def test_offline_failure_is_cached_and_can_be_retried_explicitly(releases):
    releases.side_effect = OSError("Offline")
    results = sdk.check_sdk_versions(["sagemaker"])
    assert all(result.status == "check-unavailable" for result in results)
    sdk.check_sdk_versions(["sagemaker"])
    assert releases.call_count == 2
    releases.side_effect = None
    assert all(
        result.status == "update-available" for result in sdk.check_sdk_versions(["sagemaker"], refresh=True)
    )


@pytest.mark.parametrize(
    "contents", ["invalid JSON", "[]", '{"checked_at": "bad"}', '{"checked_at":100000,"latest":"invalid"}']
)
def test_broken_cache_is_ignored(releases, contents):
    (sdk.CACHE_DIR / "sagemaker.json").write_text(contents)
    assert sdk.check_sdk_versions(["sagemaker"])[0].status == "update-available"
    assert releases.call_count == 2


def test_missing_distributions_do_not_trigger_network_requests(releases, monkeypatch):
    def missing(package):
        raise sdk.PackageNotFoundError(package)

    monkeypatch.setattr(sdk, "version", missing)
    assert all(result.status == "not-installed" for result in sdk.check_sdk_versions())
    releases.assert_not_called()


def test_unwritable_cache_does_not_hide_available_updates(releases, monkeypatch):
    monkeypatch.setattr(sdk, "write_json_atomic", Mock(side_effect=PermissionError))
    assert sdk.check_sdk_versions(["sagemaker"])[0].status == "update-available"


def test_cached_notices_need_no_network_or_thread(releases, monkeypatch):
    sdk.check_sdk_versions(["sagemaker"])
    releases.reset_mock()
    thread = Mock()
    monkeypatch.setattr(sdk, "Thread", thread)
    output = io.StringIO()
    check = sdk.SdkUpdateCheck("sagemaker")
    check.notify(Console(file=output, width=150))
    assert "SDK update available: sagemaker 3.4.0 → 3.10.0" in output.getvalue()
    thread.assert_not_called()
    releases.assert_not_called()


def test_cold_check_does_not_wait_for_network_or_print_from_worker(releases, monkeypatch):
    started, unblock = Event(), Event()
    workers = []

    def delayed_release(package):
        started.set()
        assert unblock.wait(timeout=2)
        return {"latest": "3.10.0", "requires_python": ">=3.9"}

    def worker(**kwargs):
        thread = Thread(**kwargs)
        workers.append(thread)
        return thread

    releases.side_effect = delayed_release
    monkeypatch.setattr(sdk, "Thread", worker)
    check = sdk.SdkUpdateCheck("sagemaker")
    try:
        assert started.wait(timeout=1)
        assert workers[0].daemon
        output = io.StringIO()
        check.notify(Console(file=output))
        assert output.getvalue() == ""
    finally:
        unblock.set()
        workers[0].join(timeout=2)
    assert not workers[0].is_alive()
    assert check.results[0].status == "update-available"
    assert output.getvalue() == ""


def test_opt_out_skips_cache_and_network(releases, monkeypatch):
    monkeypatch.setenv("HF_CLOUD_NO_UPDATE_CHECK", "1")
    check_versions = Mock()
    monkeypatch.setattr(sdk, "check_sdk_versions", check_versions)
    assert sdk.SdkUpdateCheck("sagemaker").results == []
    check_versions.assert_not_called()


def test_advisory_setup_failure_does_not_escape(releases, monkeypatch):
    monkeypatch.setattr(sdk, "check_sdk_versions", Mock(side_effect=OSError("Broken metadata")))
    assert sdk.SdkUpdateCheck("sagemaker").results == []


def test_manual_report_filters_provider_and_keeps_json_clean(releases, monkeypatch):
    monkeypatch.setenv("HF_CLOUD_NO_UPDATE_CHECK", "1")
    result = CliRunner().invoke(app, ["providers", "versions", "--provider", "SageMaker", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["source"] == "https://pypi.org"
    assert data["sdks"] == [
        {**asdict(row), "status": row.status} for row in sdk.check_sdk_versions(["sagemaker"])
    ]
    assert len(data["sdks"]) == 2
    assert releases.call_count == 2
    result = CliRunner().invoke(app, ["providers", "versions", "--refresh"])
    assert result.exit_code == 0, result.output
    assert "Provider SDK versions" in result.stdout
    assert releases.call_count == 6


def test_manual_report_rejects_unknown_provider_before_network(releases):
    result = CliRunner().invoke(app, ["providers", "versions", "--provider", "unknown"])
    assert result.exit_code == 2
    assert "Choose from" in result.output
    releases.assert_not_called()


@pytest.mark.parametrize("provider", ["sagemaker", "vertex"])
@pytest.mark.parametrize("fails", [False, True])
def test_deploy_commands_show_notices_on_completion_without_changing_outcome(monkeypatch, provider, fails):
    module = importlib.import_module(f"hf_cloud.cli.{provider}")
    monkeypatch.setattr(module, "_get_provider_defaults", lambda: {})
    check = Mock()
    checker = Mock(return_value=check)
    monkeypatch.setattr(module, "SdkUpdateCheck", checker)
    deployment = SimpleNamespace(
        deployment_id="test",
        deployment_name="test",
        status=DeploymentStatus.RUNNING if provider == "sagemaker" else DeploymentStatus.CREATING,
        region="us-east-1",
        instance_type="test",
        instance_count=1,
    )
    cloud = Mock()
    cloud.client.region = "us-east-1"
    cloud.deploy.return_value = deployment
    if fails:
        cloud.deploy.side_effect = RuntimeError("Test failure")
    if provider == "sagemaker":
        monkeypatch.setattr(module, "SageMakerProvider", Mock(return_value=cloud))
    else:
        monkeypatch.setattr(module.ProviderRegistry, "get_provider", Mock(return_value=cloud))
    state = Mock()
    monkeypatch.setattr(module, "StateManager", Mock(return_value=state))
    result = CliRunner().invoke(app, [provider, "deploy", "org/model", "--name", "test"])
    assert result.exit_code == (1 if fails else 0), result.output
    checker.assert_called_once_with(provider)
    check.notify.assert_called_once_with(module.console)
    assert state.add_deployment.call_count == (0 if fails else 1)
