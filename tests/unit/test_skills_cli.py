"""Tests for skills CLI commands."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hf_cloud.cli.skills import build_skill_md, skills_app

runner = CliRunner()


def test_build_skill_md_contains_core_commands() -> None:
    """Generated skill should include essential command hints."""
    content = build_skill_md()
    assert "hf cloud sagemaker deploy MODEL_ID --name NAME" in content
    assert "hf cloud vertex estimate MODEL_ID" in content
    assert "replace `hf cloud` with `hf-cloud`" in content


def test_skills_add_to_custom_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`skills add --dest` should create SKILL.md in target directory."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(skills_app, ["add", "--dest", "my-skills"])

    assert result.exit_code == 0
    skill_file = Path("my-skills") / "hf-cloud" / "SKILL.md"
    assert skill_file.exists()


def test_skills_add_without_force_fails_when_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`skills add` should fail if skill exists and --force is not set."""
    monkeypatch.chdir(tmp_path)
    first = runner.invoke(skills_app, ["add", "--dest", "my-skills"])
    second = runner.invoke(skills_app, ["add", "--dest", "my-skills"])

    assert first.exit_code == 0
    assert second.exit_code == 1


def test_skills_add_for_codex_creates_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`skills add --codex` should install central skill and codex symlink."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(skills_app, ["add", "--codex"])

    assert result.exit_code == 0
    central_skill = Path(".agents/skills/hf-cloud/SKILL.md")
    codex_link = Path(".codex/skills/hf-cloud")

    assert central_skill.exists()
    assert codex_link.exists()
