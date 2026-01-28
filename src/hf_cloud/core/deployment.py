"""Deployment data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DeploymentStatus(str, Enum):
    """Deployment status enumeration."""

    CREATING = "creating"
    RUNNING = "running"
    UPDATING = "updating"
    DELETING = "deleting"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Deployment:
    """Deployment information data class."""

    # Core identification
    deployment_id: str
    deployment_name: str
    provider: str
    model_id: str

    # Status
    status: DeploymentStatus

    # Configuration
    config: dict[str, Any]

    # Endpoints
    endpoint_url: str | None = None

    # Metadata
    created_at: datetime | None = None
    updated_at: datetime | None = None
    region: str | None = None

    # Resource information
    instance_type: str | None = None
    instance_count: int | None = None

    # Cost tracking
    estimated_cost_per_hour: float | None = None

    # Tags
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "deployment_id": self.deployment_id,
            "deployment_name": self.deployment_name,
            "provider": self.provider,
            "model_id": self.model_id,
            "status": self.status.value,
            "config": self.config,
            "endpoint_url": self.endpoint_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "region": self.region,
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "estimated_cost_per_hour": self.estimated_cost_per_hour,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Deployment":
        """Create a Deployment instance from a dictionary."""
        # Handle datetime fields
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        # Handle status
        status = data.get("status")
        if isinstance(status, str):
            status = DeploymentStatus(status)

        return cls(
            deployment_id=data["deployment_id"],
            deployment_name=data["deployment_name"],
            provider=data["provider"],
            model_id=data["model_id"],
            status=status,
            config=data.get("config", {}),
            endpoint_url=data.get("endpoint_url"),
            created_at=created_at,
            updated_at=updated_at,
            region=data.get("region"),
            instance_type=data.get("instance_type"),
            instance_count=data.get("instance_count"),
            estimated_cost_per_hour=data.get("estimated_cost_per_hour"),
            tags=data.get("tags", {}),
        )
