"""Best-effort SDK release checks without importing cloud SDKs or changing dependencies."""

import json
import os
import platform
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Thread
from typing import Optional
from urllib.request import Request, urlopen

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from rich.console import Console
from rich.text import Text

from hf_cloud.core.storage import write_json_atomic

PROVIDER_SDKS = {
    "sagemaker": ("sagemaker", "boto3"),
    "vertex": ("google-cloud-aiplatform", "google-cloud-logging"),
}
CACHE_DIR = Path.home() / ".hf-cloud" / "cache" / "sdk-versions"
CACHE_TTL = 24 * 60 * 60
REQUEST_TIMEOUT = 2
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class SdkVersion:
    provider: str
    package: str
    installed: Optional[str]
    latest: Optional[str] = None
    requires_python: Optional[str] = None

    @property
    def status(self) -> str:
        if self.installed is None:
            return "not-installed"
        if self.latest is None:
            return "check-unavailable"
        try:
            if Version(self.latest) <= Version(self.installed):
                return "up-to-date"
            if self.requires_python and not SpecifierSet(self.requires_python).contains(
                platform.python_version()
            ):
                return "requires-newer-python"
            return "update-available"
        except ValueError:
            return "check-unavailable"


def _latest_release(package: str) -> dict:
    request = Request(
        f"https://pypi.org/pypi/{package}/json",
        headers={"Accept": "application/json", "User-Agent": "hf-cloud SDK update check"},
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("Package metadata exceeds size limit")
    payload = json.loads(data)
    latest = Version(payload["info"]["version"])
    if (
        latest.is_prerelease
        or latest.is_devrelease
        or not any(not artifact.get("yanked", False) for artifact in payload["urls"])
    ):
        raise ValueError("No stable, non-yanked release available")
    requires_python = payload["info"].get("requires_python") or None
    if requires_python:
        SpecifierSet(requires_python)
    return {"latest": str(latest), "requires_python": requires_python}


def _cached_release(package: str) -> Optional[dict]:
    try:
        data = json.loads((CACHE_DIR / f"{package}.json").read_text())
        age = time.time() - data["checked_at"]
        if not 0 <= age < CACHE_TTL:
            return None
        latest, requires_python = data["latest"], data.get("requires_python")
        if latest is not None:
            parsed = Version(latest)
            if parsed.is_prerelease or parsed.is_devrelease:
                return None
        if requires_python:
            SpecifierSet(requires_python)
        return {"latest": latest, "requires_python": requires_python}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _release(package: str, *, refresh: bool, cached_only: bool) -> dict:
    cached = None if refresh else _cached_release(package)
    if cached is not None:
        return cached
    if cached_only:
        return {}
    try:
        release = _latest_release(package)
    except Exception:
        # Advisory checks must never interrupt deployment, including when offline.
        release = {"latest": None, "requires_python": None}
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        write_json_atomic(CACHE_DIR / f"{package}.json", {"checked_at": time.time(), **release})
    except OSError:
        pass
    return release


def check_sdk_versions(
    providers: Optional[list[str]] = None, *, refresh: bool = False, cached_only: bool = False
) -> list[SdkVersion]:
    """Read installed distributions and stable PyPI versions; never import an SDK."""
    results = []
    for provider in providers if providers is not None else PROVIDER_SDKS:
        for package in PROVIDER_SDKS[provider]:
            try:
                installed = version(package)
            except PackageNotFoundError:
                installed = None
            release = _release(package, refresh=refresh, cached_only=cached_only) if installed else {}
            results.append(SdkVersion(provider, package, installed, **release))
    return results


class SdkUpdateCheck:
    """Use cached results immediately and refresh on a daemon without delaying deployment.

    Only the calling thread prints. If a deployment ends before the check finishes,
    the next invocation can use the cached result. No worker is joined on exit.
    """

    def __init__(self, provider: str):
        self.results: list[SdkVersion] = []
        if os.environ.get("HF_CLOUD_NO_UPDATE_CHECK", "").lower() in {"1", "true", "yes"}:
            return
        try:
            self.results = check_sdk_versions([provider], cached_only=True)
            if any(result.installed and _cached_release(result.package) is None for result in self.results):
                Thread(target=self._refresh, args=(provider,), daemon=True).start()
        except Exception:
            pass

    def _refresh(self, provider: str) -> None:
        try:
            self.results = check_sdk_versions([provider])
        except Exception:
            pass

    def notify(self, console: Console) -> None:
        for result in self.results:
            if result.status in {"update-available", "requires-newer-python"}:
                message = (
                    f"SDK update available: {result.package} {result.installed} → {result.latest} (PyPI). "
                )
                if result.status == "requires-newer-python":
                    message += f"Requires Python {result.requires_python}. "
                message += "Review the release notes before upgrading."
                console.print(Text(message, style="cyan"))
