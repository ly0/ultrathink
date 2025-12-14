"""Configuration management for Ultrathink.

This module handles global and project-specific configuration,
including API keys, model settings, and user preferences.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ProviderType(str, Enum):
    """Supported AI model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"

    @classmethod
    def from_string(cls, value: str) -> "ProviderType":
        """Parse provider from string, with alias support."""
        aliases = {
            "claude": cls.ANTHROPIC,
            "gpt": cls.OPENAI,
            "google": cls.GEMINI,
            "ds": cls.DEEPSEEK,
        }
        normalized = value.strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


class ModelProfile(BaseModel):
    """Configuration for a specific AI model."""

    model_config = {"protected_namespaces": ()}

    provider: ProviderType = ProviderType.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 16384
    temperature: float = 0.7
    context_window: Optional[int] = None
    # Pricing (USD per 1M tokens)
    input_cost_per_million_tokens: float = 0.0
    output_cost_per_million_tokens: float = 0.0


class ModelPointers(BaseModel):
    """Pointers to different model profiles for different purposes.

    Built-in aliases:
        - main: Primary model for main agent interactions
        - task: Model for subagent/task execution
        - quick: Model for quick operations (e.g., summarization)

    Custom aliases can be added via the 'custom' dict and used in
    subagent or tool configurations.
    """

    # Built-in aliases
    main: str = "default"
    task: str = "default"
    quick: str = "default"

    # Custom aliases (user-defined)
    custom: Dict[str, str] = Field(default_factory=dict)

    def get(self, alias: str) -> Optional[str]:
        """Get profile name for an alias.

        Args:
            alias: The alias name (builtin or custom)

        Returns:
            Profile name or None if not found
        """
        # Check builtin aliases first
        if alias in ("main", "task", "quick"):
            return getattr(self, alias)
        # Check custom aliases
        return self.custom.get(alias)

    def set(self, alias: str, profile_name: str) -> None:
        """Set an alias to point to a profile.

        Args:
            alias: The alias name
            profile_name: The profile name to point to
        """
        if alias in ("main", "task", "quick"):
            setattr(self, alias, profile_name)
        else:
            self.custom[alias] = profile_name

    def remove(self, alias: str) -> bool:
        """Remove a custom alias.

        Args:
            alias: The alias name to remove

        Returns:
            True if removed, False if not found or is builtin
        """
        if alias in ("main", "task", "quick"):
            return False  # Cannot remove builtin
        if alias in self.custom:
            del self.custom[alias]
            return True
        return False

    def all_aliases(self) -> Dict[str, str]:
        """Get all aliases (builtin + custom)."""
        result = {
            "main": self.main,
            "task": self.task,
            "quick": self.quick,
        }
        result.update(self.custom)
        return result


class GlobalConfig(BaseModel):
    """Global configuration stored in ~/.ultrathink.json"""

    model_config = {"protected_namespaces": ()}

    # Model configuration
    model_profiles: Dict[str, ModelProfile] = Field(default_factory=dict)
    model_pointers: ModelPointers = Field(default_factory=ModelPointers)

    # User preferences
    theme: str = "dark"
    verbose: bool = False
    safe_mode: bool = True

    # Onboarding
    has_completed_onboarding: bool = False


class ProjectConfig(BaseModel):
    """Project-specific configuration stored in .ultrathink/config.json"""

    # Tool permissions
    allowed_tools: List[str] = Field(default_factory=list)
    bash_allow_rules: List[str] = Field(default_factory=list)
    bash_deny_rules: List[str] = Field(default_factory=list)

    # Context
    context: Dict[str, str] = Field(default_factory=dict)
    context_files: List[str] = Field(default_factory=list)

    # Trust
    has_trust_dialog_accepted: bool = False


class EnvSettings(BaseSettings):
    """Environment-based settings."""

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None

    # Base URLs
    anthropic_base_url: Optional[str] = None
    openai_base_url: Optional[str] = None
    gemini_base_url: Optional[str] = None
    deepseek_base_url: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class ConfigManager:
    """Manages global and project-specific configuration."""

    def __init__(self) -> None:
        self.global_config_path = Path.home() / ".ultrathink.json"
        self.current_project_path: Optional[Path] = None
        self._global_config: Optional[GlobalConfig] = None
        self._project_config: Optional[ProjectConfig] = None
        self._env_settings: Optional[EnvSettings] = None

    def get_env_settings(self) -> EnvSettings:
        """Load environment settings."""
        if self._env_settings is None:
            self._env_settings = EnvSettings()
        return self._env_settings

    def get_global_config(self) -> GlobalConfig:
        """Load and return global configuration."""
        if self._global_config is None:
            if self.global_config_path.exists():
                try:
                    data = json.loads(self.global_config_path.read_text())
                    self._global_config = GlobalConfig(**data)
                except json.JSONDecodeError as e:
                    import sys
                    print(f"Warning: Invalid JSON in {self.global_config_path}: {e}", file=sys.stderr)
                    self._global_config = GlobalConfig()
                except Exception as e:
                    import sys
                    print(f"Warning: Error loading config from {self.global_config_path}: {e}", file=sys.stderr)
                    self._global_config = GlobalConfig()
            else:
                self._global_config = GlobalConfig()
        return self._global_config

    def save_global_config(self, config: GlobalConfig) -> None:
        """Save global configuration."""
        self._global_config = config
        self.global_config_path.write_text(config.model_dump_json(indent=2))

    def get_project_config(self, project_path: Optional[Path] = None) -> ProjectConfig:
        """Load and return project configuration."""
        if project_path is not None:
            if self.current_project_path != project_path:
                self._project_config = None
            self.current_project_path = project_path

        if self.current_project_path is None:
            return ProjectConfig()

        config_path = self.current_project_path / ".ultrathink" / "config.json"

        if self._project_config is None:
            if config_path.exists():
                try:
                    data = json.loads(config_path.read_text())
                    self._project_config = ProjectConfig(**data)
                except Exception:
                    self._project_config = ProjectConfig()
            else:
                self._project_config = ProjectConfig()

        return self._project_config

    def save_project_config(
        self, config: ProjectConfig, project_path: Optional[Path] = None
    ) -> None:
        """Save project configuration."""
        if project_path is not None:
            self.current_project_path = project_path

        if self.current_project_path is None:
            return

        config_dir = self.current_project_path / ".ultrathink"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "config.json"
        self._project_config = config
        config_path.write_text(config.model_dump_json(indent=2))

    def get_api_key(self, provider: ProviderType) -> Optional[str]:
        """Get API key for a provider from environment or config."""
        env_settings = self.get_env_settings()

        # Check environment first
        if provider == ProviderType.ANTHROPIC:
            if env_settings.anthropic_api_key:
                return env_settings.anthropic_api_key
            if os.environ.get("ANTHROPIC_API_KEY"):
                return os.environ["ANTHROPIC_API_KEY"]
        elif provider == ProviderType.OPENAI:
            if env_settings.openai_api_key:
                return env_settings.openai_api_key
            if os.environ.get("OPENAI_API_KEY"):
                return os.environ["OPENAI_API_KEY"]
        elif provider == ProviderType.GEMINI:
            if env_settings.gemini_api_key:
                return env_settings.gemini_api_key
            if env_settings.google_api_key:
                return env_settings.google_api_key
            if os.environ.get("GEMINI_API_KEY"):
                return os.environ["GEMINI_API_KEY"]
            if os.environ.get("GOOGLE_API_KEY"):
                return os.environ["GOOGLE_API_KEY"]
        elif provider == ProviderType.DEEPSEEK:
            if env_settings.deepseek_api_key:
                return env_settings.deepseek_api_key
            if os.environ.get("DEEPSEEK_API_KEY"):
                return os.environ["DEEPSEEK_API_KEY"]

        # Check global config
        global_config = self.get_global_config()
        for profile in global_config.model_profiles.values():
            if profile.provider == provider and profile.api_key:
                return profile.api_key

        return None

    def get_base_url(self, provider: ProviderType) -> Optional[str]:
        """Get base URL for a provider from environment or config."""
        env_settings = self.get_env_settings()

        # Check environment first
        if provider == ProviderType.ANTHROPIC:
            if env_settings.anthropic_base_url:
                return env_settings.anthropic_base_url
            if os.environ.get("ANTHROPIC_BASE_URL"):
                return os.environ["ANTHROPIC_BASE_URL"]
            if os.environ.get("ANTHROPIC_API_BASE"):
                return os.environ["ANTHROPIC_API_BASE"]
        elif provider == ProviderType.OPENAI:
            if env_settings.openai_base_url:
                return env_settings.openai_base_url
            if os.environ.get("OPENAI_BASE_URL"):
                return os.environ["OPENAI_BASE_URL"]
            if os.environ.get("OPENAI_API_BASE"):
                return os.environ["OPENAI_API_BASE"]
        elif provider == ProviderType.GEMINI:
            if env_settings.gemini_base_url:
                return env_settings.gemini_base_url
            if os.environ.get("GEMINI_BASE_URL"):
                return os.environ["GEMINI_BASE_URL"]
            if os.environ.get("GOOGLE_API_BASE"):
                return os.environ["GOOGLE_API_BASE"]
        elif provider == ProviderType.DEEPSEEK:
            if env_settings.deepseek_base_url:
                return env_settings.deepseek_base_url
            if os.environ.get("DEEPSEEK_BASE_URL"):
                return os.environ["DEEPSEEK_BASE_URL"]
            # Default DeepSeek base URL
            return "https://api.deepseek.com/v1"

        # Check global config
        global_config = self.get_global_config()
        for profile in global_config.model_profiles.values():
            if profile.provider == provider and profile.api_base:
                return profile.api_base

        return None

    def get_model_profile(self, alias: str = "main") -> Optional[ModelProfile]:
        """Get the model profile for a given alias.

        Args:
            alias: Alias name (builtin: main/task/quick, or custom)

        Returns:
            ModelProfile or None if not found
        """
        global_config = self.get_global_config()
        profile_name = global_config.model_pointers.get(alias)
        if profile_name is None:
            # Fallback: treat alias as direct profile name
            profile_name = alias
        return global_config.model_profiles.get(profile_name)

    def get_model_string(self, alias: str = "main") -> str:
        """Get model string in format 'provider:model' for langchain.

        Args:
            alias: Alias name or direct profile name

        Returns:
            Model string like 'anthropic:claude-sonnet-4-20250514'
        """
        profile = self.get_model_profile(alias)
        if profile:
            return f"{profile.provider.value}:{profile.model}"
        # Default to Claude Sonnet
        return "anthropic:claude-sonnet-4-20250514"

    def set_alias(self, alias: str, profile_name: str) -> None:
        """Set an alias to point to a profile.

        Args:
            alias: Alias name
            profile_name: Profile name to point to

        Raises:
            ValueError: If the profile doesn't exist
        """
        config = self.get_global_config()
        if profile_name not in config.model_profiles:
            raise ValueError(f"Profile '{profile_name}' does not exist")
        config.model_pointers.set(alias, profile_name)
        self.save_global_config(config)

    def remove_alias(self, alias: str) -> bool:
        """Remove a custom alias.

        Args:
            alias: Alias name to remove

        Returns:
            True if removed, False if not found or is builtin
        """
        config = self.get_global_config()
        if config.model_pointers.remove(alias):
            self.save_global_config(config)
            return True
        return False

    def update_profile(
        self,
        name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> bool:
        """Update an existing model profile.

        Args:
            name: Profile name to update
            api_key: New API key (None to keep existing)
            api_base: New base URL (None to keep existing)
            model: New model name (None to keep existing)
            temperature: New temperature (None to keep existing)

        Returns:
            True if updated, False if not found
        """
        config = self.get_global_config()
        if name not in config.model_profiles:
            return False

        profile = config.model_profiles[name]

        if api_key is not None:
            profile.api_key = api_key
        if api_base is not None:
            profile.api_base = api_base
        if model is not None:
            profile.model = model
        if temperature is not None:
            profile.temperature = temperature

        self.save_global_config(config)
        return True

    def remove_profile(self, name: str) -> bool:
        """Remove a model profile.

        Args:
            name: Profile name to remove

        Returns:
            True if removed, False if not found
        """
        config = self.get_global_config()
        if name not in config.model_profiles:
            return False

        # Check if any alias points to this profile
        aliases = config.model_pointers.all_aliases()
        pointing_aliases = [a for a, p in aliases.items() if p == name]
        if pointing_aliases:
            raise ValueError(
                f"Cannot remove profile '{name}': aliases {pointing_aliases} point to it"
            )

        del config.model_profiles[name]
        self.save_global_config(config)
        return True

    def list_profiles(self) -> Dict[str, ModelProfile]:
        """List all model profiles."""
        return self.get_global_config().model_profiles.copy()

    def list_aliases(self) -> Dict[str, str]:
        """List all aliases (builtin + custom)."""
        return self.get_global_config().model_pointers.all_aliases()

    def add_model_profile(
        self,
        name: str,
        profile: ModelProfile,
        overwrite: bool = False,
        set_as_main: bool = False,
        set_alias: Optional[str] = None,
    ) -> GlobalConfig:
        """Add or replace a model profile.

        Args:
            name: Profile name
            profile: The ModelProfile configuration
            overwrite: If True, overwrite existing profile
            set_as_main: If True, set as the main alias
            set_alias: Optional custom alias to set pointing to this profile

        Returns:
            Updated GlobalConfig
        """
        config = self.get_global_config()
        if not overwrite and name in config.model_profiles:
            raise ValueError(f"Model profile '{name}' already exists.")

        config.model_profiles[name] = profile
        if set_as_main or not config.model_pointers.main:
            config.model_pointers.main = name
        if set_alias:
            config.model_pointers.set(set_alias, name)
        self.save_global_config(config)
        return config

    def ensure_default_profile(self) -> None:
        """Ensure a default model profile exists."""
        config = self.get_global_config()
        if "default" not in config.model_profiles:
            default_profile = ModelProfile(
                provider=ProviderType.ANTHROPIC,
                model="claude-sonnet-4-20250514",
            )
            config.model_profiles["default"] = default_profile
            self.save_global_config(config)


# Global instance
config_manager = ConfigManager()


def get_global_config() -> GlobalConfig:
    """Get global configuration."""
    return config_manager.get_global_config()


def save_global_config(config: GlobalConfig) -> None:
    """Save global configuration."""
    config_manager.save_global_config(config)


def get_project_config(project_path: Optional[Path] = None) -> ProjectConfig:
    """Get project configuration."""
    return config_manager.get_project_config(project_path)


def save_project_config(config: ProjectConfig, project_path: Optional[Path] = None) -> None:
    """Save project configuration."""
    config_manager.save_project_config(config, project_path)


def get_api_key(provider: ProviderType) -> Optional[str]:
    """Get API key for a provider."""
    return config_manager.get_api_key(provider)


def get_base_url(provider: ProviderType) -> Optional[str]:
    """Get base URL for a provider."""
    return config_manager.get_base_url(provider)


def get_model_string(alias: str = "main") -> str:
    """Get model string for langchain."""
    return config_manager.get_model_string(alias)


def get_model_profile(alias: str = "main") -> Optional[ModelProfile]:
    """Get model profile by alias or profile name."""
    return config_manager.get_model_profile(alias)


def add_model_profile(
    name: str,
    profile: ModelProfile,
    overwrite: bool = False,
    set_as_main: bool = False,
    set_alias: Optional[str] = None,
) -> GlobalConfig:
    """Add or replace a model profile."""
    return config_manager.add_model_profile(
        name, profile, overwrite, set_as_main, set_alias
    )


def set_alias(alias: str, profile_name: str) -> None:
    """Set an alias to point to a profile."""
    config_manager.set_alias(alias, profile_name)


def remove_alias(alias: str) -> bool:
    """Remove a custom alias."""
    return config_manager.remove_alias(alias)


def update_profile(
    name: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> bool:
    """Update an existing model profile."""
    return config_manager.update_profile(name, api_key, api_base, model, temperature)


def remove_profile(name: str) -> bool:
    """Remove a model profile."""
    return config_manager.remove_profile(name)


def list_profiles() -> Dict[str, ModelProfile]:
    """List all model profiles."""
    return config_manager.list_profiles()


def list_aliases() -> Dict[str, str]:
    """List all aliases (builtin + custom)."""
    return config_manager.list_aliases()


def ensure_onboarding() -> bool:
    """Check if onboarding is complete, run if needed."""
    config = get_global_config()
    if config.has_completed_onboarding:
        return True

    # Check if we have at least one API key
    has_key = any(
        get_api_key(provider) for provider in ProviderType
    )

    if has_key:
        config.has_completed_onboarding = True
        config_manager.ensure_default_profile()
        save_global_config(config)
        return True

    # Need to run onboarding
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel(
        "[bold red]No API key found![/bold red]\n\n"
        "Please set one of the following environment variables:\n"
        "  - ANTHROPIC_API_KEY (for Claude models)\n"
        "  - OPENAI_API_KEY (for GPT models)\n"
        "  - GEMINI_API_KEY or GOOGLE_API_KEY (for Gemini models)\n\n"
        "Or create a .env file in your project directory.",
        title="Ultrathink Setup",
        border_style="red",
    ))
    return False
