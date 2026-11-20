"""Literal env-file parsing and deterministic override order without shell execution."""

import pytest

from hf_cloud.core.exceptions import ConfigurationError
from hf_cloud.utils.environment import load_environment, merge_environment


def test_files_then_cli_last_wins_and_defaults_survive(tmp_path):
    first, second = tmp_path / "first.env", tmp_path / "second.env"
    first.write_text("# comment\nFIRST=1\nDUP=first\nEMPTY=\n", encoding="utf-8")
    second.write_text("DUP=second\nexport QUOTED='a # b = c'\nPLAIN=a=b # comment\n", encoding="utf-8")
    custom = load_environment(["DUP=third", "DUP=last", "SPACE= a  b "], [first, second])
    assert merge_environment({"DEFAULT": "kept", "DUP": "default"}, custom) == {
        "DEFAULT": "kept",
        "FIRST": "1",
        "DUP": "last",
        "EMPTY": "",
        "QUOTED": "a # b = c",
        "PLAIN": "a=b",
        "SPACE": " a  b ",
    }


def test_env_file_quotes_bom_unicode_and_no_expansion(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        '\ufeffDOUBLE="one\\ntwo\\t\\"three\\"" # comment\r\n'
        "SINGLE='one\\ntwo'\nLITERAL=$(touch sentinel) ${HOME}\nHASH=abc#def\nUNICODE=España\n",
        encoding="utf-8",
    )
    assert load_environment(files=[path]) == {
        "DOUBLE": 'one\ntwo\t"three"',
        "SINGLE": r"one\ntwo",
        "LITERAL": "$(touch sentinel) ${HOME}",
        "HASH": "abc#def",
        "UNICODE": "España",
    }


@pytest.mark.parametrize(
    "assignment",
    [
        "MISSING_EQUALS",
        "=private-value",
        "BAD-NAME=private-value",
        "1BAD=private-value",
        "GOOD=private-value\0",
    ],
)
def test_invalid_assignments_never_echo_values(assignment):
    with pytest.raises(ConfigurationError) as error:
        load_environment([assignment])
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("line", ['KEY="private-value', 'KEY="private-value" trailing', "KEY='private-value"])
def test_invalid_file_quotes_have_line_number_without_value(tmp_path, line):
    path = tmp_path / "bad.env"
    path.write_text("# comment\n" + line, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="line 2") as error:
        load_environment(files=[path])
    assert "private-value" not in str(error.value)


def test_missing_or_non_utf8_file_is_actionable(tmp_path):
    path = tmp_path / "bad.env"
    with pytest.raises(ConfigurationError, match="Cannot read UTF-8"):
        load_environment(files=[path])
    path.write_bytes(b"KEY=\xff")
    with pytest.raises(ConfigurationError, match="Cannot read UTF-8"):
        load_environment(files=[path])
