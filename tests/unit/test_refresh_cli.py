"""Refresh integration tests using local state and fake providers."""

from unittest.mock import Mock

from typer.testing import CliRunner

from hf_cloud.cli import app
from hf_cloud.core.deployment import Deployment, DeploymentStatus
from hf_cloud.core.state import StateManager


def _deployment(name, provider, region):
    return Deployment(name, name, provider, "gpt2", DeploymentStatus.RUNNING, {}, region=region)


def test_vertex_refresh_persists_one_batch(tmp_path, monkeypatch):
    state = StateManager(tmp_path / "state.json")
    save = Mock(wraps=state._save_state)
    monkeypatch.setattr(state, "_save_state", save)
    monkeypatch.setattr("hf_cloud.cli.global_commands.StateManager", lambda: state)
    provider = Mock()
    provider.list_deployments.return_value = [
        _deployment(f"endpoint-{i}", "vertex", "us-central1") for i in range(5)
    ]
    monkeypatch.setattr("hf_cloud.cli.global_commands.ProviderRegistry.get_provider", lambda name: provider)

    result = CliRunner().invoke(app, ["ls", "--refresh", "--provider", "vertex"])
    assert result.exit_code == 0, result.output
    save.assert_called_once_with()
    assert len(StateManager(state.state_file).list_deployments()) == 5


def test_sagemaker_refresh_saves_each_successful_region_once(tmp_path, monkeypatch):
    state = StateManager(tmp_path / "state.json")
    for region in ("us-east-1", "eu-west-1"):
        state.add_deployment(_deployment("existing", "sagemaker", region))
    save = Mock(wraps=state._save_state)
    monkeypatch.setattr(state, "_save_state", save)
    monkeypatch.setattr("hf_cloud.cli.global_commands.StateManager", lambda: state)
    monkeypatch.setattr(
        "hf_cloud.providers.sagemaker.client.SageMakerClient.resolve_region", lambda: "us-east-1"
    )

    def regional_provider(region):
        provider = Mock()
        if region == "eu-west-1":
            provider.list_deployments.side_effect = RuntimeError("Region unavailable")
        else:
            provider.list_deployments.return_value = [
                _deployment(f"endpoint-{i}", "sagemaker", region) for i in range(5)
            ]
        return provider

    monkeypatch.setattr("hf_cloud.providers.sagemaker.SageMakerProvider", regional_provider)
    result = CliRunner().invoke(app, ["ls", "--refresh", "--provider", "sagemaker"])
    assert result.exit_code == 0, result.output
    assert "Region unavailable" in result.stdout
    save.assert_called_once_with()
    reloaded = StateManager(state.state_file)
    assert len(reloaded.list_deployments()) == 7
    assert reloaded.get_deployment("existing", provider="sagemaker", region="eu-west-1") is not None
