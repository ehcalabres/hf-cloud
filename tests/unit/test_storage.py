"""Tests for complete-file replacement of local configuration and state."""

import json

import pytest

from hf_cloud.core.config import Config
from hf_cloud.core.storage import write_json_atomic


def test_atomic_json_round_trip(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"old": true}')
    write_json_atomic(path, {"new": ["model", 123]})
    assert json.loads(path.read_text()) == {"new": ["model", 123]}
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_follows_symlink(tmp_path):
    target_dir = tmp_path / "managed-config"
    target_dir.mkdir()
    target = target_dir / "config.json"
    target.write_text('{"old": true}')
    link = tmp_path / "config.json"
    link.symlink_to(target)
    write_json_atomic(link, {"new": True})
    assert link.is_symlink()
    assert json.loads(target.read_text()) == {"new": True}
    assert list(target_dir.iterdir()) == [target]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["serialization", "replacement"])
def test_failed_write_preserves_destination_and_cleans_up(tmp_path, monkeypatch, existing, failure):
    path = tmp_path / "data.json"
    original = '{"existing": "data"}'
    if existing:
        path.write_text(original)

    def fail_replace(*args):
        raise OSError("Replacement failed")

    if failure == "replacement":
        monkeypatch.setattr("hf_cloud.core.storage.os.replace", fail_replace)
    data = {"bad": object()} if failure == "serialization" else {"new": True}
    with pytest.raises(TypeError if failure == "serialization" else OSError):
        write_json_atomic(path, data)
    if existing:
        assert path.read_text() == original
    else:
        assert not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


def test_config_updates_remain_readable(tmp_path):
    path = tmp_path / "config.json"
    config = Config(path)
    config.set_provider_config("vertex", {"default_project": "test-project"})
    config.update_provider_config("vertex", {"default_location": "europe-west1"})
    config.set_default_provider("vertex")

    reloaded = Config(path)
    assert reloaded.get_default_provider() == "vertex"
    assert reloaded.get_provider_config("vertex") == {
        "default_project": "test-project",
        "default_location": "europe-west1",
    }
