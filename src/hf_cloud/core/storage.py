"""Shared persistence helpers for local JSON files."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Replace a JSON file only after its complete contents have been written.

    The temporary file lives beside the destination so replacement is atomic.
    This prevents partial files; it does not coordinate concurrent writers.
    """
    # Preserve symlinks used to manage configuration in a separate directory.
    path = path.resolve()
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(data, temporary_file, indent=2)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
