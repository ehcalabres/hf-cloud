"""Configuration management for HF-Cloud."""

import json
from pathlib import Path
from typing import Any

from hf_cloud.core.exceptions import ConfigurationError


class Config:
    """Configuration manager for HF-Cloud."""

    DEFAULT_CONFIG_DIR = Path.home() / ".hf-cloud"
    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"

    def __init__(self, config_file: Path | None = None):
        """Initialize configuration manager.

        Args:
            config_file: Path to config file. Defaults to ~/.hf-cloud/config.json
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Invalid configuration file: {e}")
        return self._default_config()

    def _default_config(self) -> dict[str, Any]:
        """Return default configuration."""
        return {
            "providers": {},
            "defaults": {
                "provider": None,
            },
        }

    def _save_config(self) -> None:
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(self._config, f, indent=2)

    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get configuration for a specific provider.

        Args:
            provider: Provider name (e.g., 'sagemaker', 'azure', 'vertex')

        Returns:
            Provider configuration dictionary
        """
        return self._config.get("providers", {}).get(provider, {})

    def set_provider_config(self, provider: str, config: dict[str, Any]) -> None:
        """Set configuration for a specific provider.

        Args:
            provider: Provider name
            config: Configuration dictionary
        """
        if "providers" not in self._config:
            self._config["providers"] = {}
        self._config["providers"][provider] = config
        self._save_config()

    def update_provider_config(self, provider: str, updates: dict[str, Any]) -> None:
        """Update configuration for a specific provider.

        Args:
            provider: Provider name
            updates: Configuration updates to merge
        """
        current = self.get_provider_config(provider)
        current.update(updates)
        self.set_provider_config(provider, current)

    def get_default_provider(self) -> str | None:
        """Get the default provider."""
        return self._config.get("defaults", {}).get("provider")

    def set_default_provider(self, provider: str) -> None:
        """Set the default provider.

        Args:
            provider: Provider name
        """
        if "defaults" not in self._config:
            self._config["defaults"] = {}
        self._config["defaults"]["provider"] = provider
        self._save_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot notation, e.g., 'providers.sagemaker.region')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config()
