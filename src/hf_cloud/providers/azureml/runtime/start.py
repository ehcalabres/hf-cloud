"""Start the pinned model server without shell expansion or logging secrets."""

import json
import os
import sys


def command(environ=None):
    environ = os.environ if environ is None else environ
    arguments = json.loads(environ["HF_CLOUD_VLLM_ARGS"])
    if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
        raise ValueError("HF_CLOUD_VLLM_ARGS must be a JSON string array")
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *arguments]


if __name__ == "__main__":
    os.execv(sys.executable, command())
