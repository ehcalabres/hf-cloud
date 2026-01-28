"""Provider registry for managing cloud provider instances."""

from typing import TYPE_CHECKING

from ..core.exceptions import ProviderNotFoundError

if TYPE_CHECKING:
    from .base import CloudProvider


class ProviderRegistry:
    """Central registry for cloud providers."""

    _providers: dict[str, "CloudProvider"] = {}
    _available_providers: set[str] = {"sagemaker", "azure", "gcp"}

    @classmethod
    def register_provider(cls, name: str, provider: "CloudProvider") -> None:
        """Register a provider instance.

        Args:
            name: Provider name
            provider: Provider instance
        """
        cls._providers[name] = provider

    @classmethod
    def get_provider(cls, name: str) -> "CloudProvider":
        """Get a provider instance by name.

        Args:
            name: Provider name

        Returns:
            CloudProvider instance

        Raises:
            ProviderNotFoundError: If provider is not found
        """
        if name not in cls._providers:
            cls._load_provider(name)
        return cls._providers[name]

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all available provider names.

        Returns:
            List of provider names
        """
        return list(cls._available_providers)

    @classmethod
    def list_registered_providers(cls) -> list[str]:
        """List all currently registered (loaded) providers.

        Returns:
            List of registered provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def is_provider_available(cls, name: str) -> bool:
        """Check if a provider is available.

        Args:
            name: Provider name

        Returns:
            True if provider is available
        """
        return name in cls._available_providers

    @classmethod
    def _load_provider(cls, name: str) -> None:
        """Lazy load a provider.

        Args:
            name: Provider name

        Raises:
            ProviderNotFoundError: If provider is not found
        """
        if name == "sagemaker":
            try:
                from .sagemaker import SageMakerProvider

                cls._providers[name] = SageMakerProvider()
            except ImportError as e:
                raise ProviderNotFoundError(name) from e
        elif name == "azure":
            try:
                from .azure import AzureProvider

                cls._providers[name] = AzureProvider()
            except ImportError as e:
                raise ProviderNotFoundError(name) from e
        elif name == "gcp":
            try:
                from .gcp import GCPProvider

                cls._providers[name] = GCPProvider()
            except ImportError as e:
                raise ProviderNotFoundError(name) from e
        else:
            raise ProviderNotFoundError(name)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (useful for testing)."""
        cls._providers.clear()
