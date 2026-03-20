
"""Custom exceptions for HF-Cloud."""

from typing import Optional


class HFCloudError(Exception):
    """Base exception for all HF-Cloud errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ProviderError(HFCloudError):
    """Error related to cloud provider operations."""

    def __init__(self, provider: str, message: str, details: Optional[str] = None):
        self.provider = provider
        super().__init__(f"[{provider}] {message}", details)


class DeploymentError(HFCloudError):
    """Error during deployment operations."""

    def __init__(
        self,
        message: str,
        deployment_id: Optional[str] = None,
        details: Optional[str] = None,
    ):
        self.deployment_id = deployment_id
        prefix = f"[{deployment_id}] " if deployment_id else ""
        super().__init__(f"{prefix}{message}", details)


class ConfigurationError(HFCloudError):
    """Error related to configuration."""

    pass


class AuthenticationError(HFCloudError):
    """Error related to authentication."""

    def __init__(self, provider: str, message: str, details: Optional[str] = None):
        self.provider = provider
        super().__init__(f"[{provider}] Authentication failed: {message}", details)


class DeploymentNotFoundError(DeploymentError):
    """Deployment not found error."""

    def __init__(self, deployment_id: str):
        super().__init__("Deployment not found", deployment_id=deployment_id)


class ProviderNotFoundError(ProviderError):
    """Provider not found error."""

    def __init__(self, provider: str):
        super().__init__(provider, f"Provider '{provider}' is not registered or available")
