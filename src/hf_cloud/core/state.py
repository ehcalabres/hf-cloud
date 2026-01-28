"""State management for HF-Cloud deployments."""

import json
from pathlib import Path
from typing import Any

from .deployment import Deployment


class StateManager:
    """Manages local state of deployments."""

    DEFAULT_STATE_DIR = Path.home() / ".hf-cloud"
    DEFAULT_STATE_FILE = DEFAULT_STATE_DIR / "state.json"

    def __init__(self, state_file: Path | None = None):
        """Initialize state manager.

        Args:
            state_file: Path to state file. Defaults to ~/.hf-cloud/state.json
        """
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                self._state = json.load(f)
        else:
            self._state = {"deployments": {}}

    def _save_state(self) -> None:
        """Save state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def add_deployment(self, deployment: Deployment) -> None:
        """Add or update a deployment in state.

        Args:
            deployment: Deployment object to store
        """
        self._state["deployments"][deployment.deployment_id] = deployment.to_dict()
        self._save_state()

    def get_deployment(self, deployment_id: str) -> dict[str, Any] | None:
        """Get deployment from state.

        Args:
            deployment_id: Unique deployment identifier

        Returns:
            Deployment dictionary or None if not found
        """
        return self._state["deployments"].get(deployment_id)

    def get_deployment_object(self, deployment_id: str) -> Deployment | None:
        """Get deployment as a Deployment object.

        Args:
            deployment_id: Unique deployment identifier

        Returns:
            Deployment object or None if not found
        """
        data = self.get_deployment(deployment_id)
        if data:
            return Deployment.from_dict(data)
        return None

    def remove_deployment(self, deployment_id: str) -> bool:
        """Remove deployment from state.

        Args:
            deployment_id: Unique deployment identifier

        Returns:
            True if deployment was removed, False if not found
        """
        if deployment_id in self._state["deployments"]:
            del self._state["deployments"][deployment_id]
            self._save_state()
            return True
        return False

    def list_deployments(self, provider: str | None = None) -> list[dict[str, Any]]:
        """List all deployments, optionally filtered by provider.

        Args:
            provider: Optional provider name to filter by

        Returns:
            List of deployment dictionaries
        """
        deployments = list(self._state["deployments"].values())
        if provider:
            deployments = [d for d in deployments if d.get("provider") == provider]
        return deployments

    def list_deployment_objects(self, provider: str | None = None) -> list[Deployment]:
        """List all deployments as Deployment objects.

        Args:
            provider: Optional provider name to filter by

        Returns:
            List of Deployment objects
        """
        deployments = self.list_deployments(provider)
        return [Deployment.from_dict(d) for d in deployments]

    def update_deployment_status(self, deployment_id: str, status: str) -> bool:
        """Update the status of a deployment.

        Args:
            deployment_id: Unique deployment identifier
            status: New status value

        Returns:
            True if updated, False if deployment not found
        """
        if deployment_id in self._state["deployments"]:
            self._state["deployments"][deployment_id]["status"] = status
            self._save_state()
            return True
        return False

    def clear(self) -> None:
        """Clear all deployments from state."""
        self._state = {"deployments": {}}
        self._save_state()
