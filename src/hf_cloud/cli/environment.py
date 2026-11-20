"""CLI adapter for shared literal environment-file parsing."""

from pathlib import Path
from typing import Optional

import typer

from hf_cloud.core.exceptions import ConfigurationError
from hf_cloud.utils.environment import load_environment


def deployment_environment(env: Optional[list[str]], env_file: Optional[list[Path]]) -> dict[str, str]:
    try:
        return load_environment(env, env_file)
    except ConfigurationError as error:
        raise typer.BadParameter(str(error), param_hint="--env / --env-file") from error
