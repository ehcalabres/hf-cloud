"""Literal deployment environment values; files are parsed without executing a shell."""

import re
from pathlib import Path
from typing import Any, Optional

from hf_cloud.core.exceptions import ConfigurationError

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def validate_environment(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ConfigurationError("Environment variables must be a mapping of names to strings")
    for key, item in value.items():
        if not isinstance(key, str) or not NAME.fullmatch(key):
            raise ConfigurationError(
                "Environment names must contain letters, digits or underscores and not start with a digit"
            )
        if not isinstance(item, str) or "\0" in item:
            raise ConfigurationError(f"Environment variable {key} must be a string without NUL characters")
    return dict(value)


def _assignment(text: str) -> tuple[str, str]:
    key, separator, value = text.partition("=")
    if not separator:
        raise ConfigurationError("Expected an environment assignment in KEY=VALUE format")
    key = key.strip()
    validate_environment({key: value})
    return key, value


def _file_value(value: str) -> str:
    value = value.strip()
    if not value.startswith(("'", '"')):
        return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
    quote, chars, index = value[0], [], 1
    escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
    while index < len(value):
        char = value[index]
        if char == quote:
            suffix = value[index + 1 :].strip()
            if suffix and not suffix.startswith("#"):
                raise ConfigurationError("Unexpected text after a quoted environment value")
            return "".join(chars)
        if quote == '"' and char == "\\" and index + 1 < len(value) and value[index + 1] in escapes:
            index += 1
            char = escapes[value[index]]
        chars.append(char)
        index += 1
    raise ConfigurationError("Unterminated quoted environment value (multiline values are not supported)")


def load_environment(
    assignments: Optional[list[str]] = None,
    files: Optional[list[Path]] = None,
) -> dict[str, str]:
    """Merge files in order, then CLI values. No variable or command expansion."""
    result = {}
    for path in files or []:
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as error:
            raise ConfigurationError(f"Cannot read UTF-8 environment file {path}") from error
        for number, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            try:
                key, value = _assignment(line)
                result[key] = _file_value(value)
                validate_environment({key: result[key]})
            except ConfigurationError as error:
                raise ConfigurationError(
                    f"Invalid environment file {path}, line {number}", details=str(error)
                ) from error
    for assignment in assignments or []:
        key, value = _assignment(assignment)
        result[key] = value
    return result


def merge_environment(defaults: dict[str, str], custom: Optional[dict[str, str]] = None) -> dict[str, str]:
    return {**defaults, **validate_environment(custom or {})}
