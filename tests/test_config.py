"""Tests for configuration module."""

import json
import tempfile
from pathlib import Path

import pytest

from ultrathink.core.config import (
    ConfigManager,
    GlobalConfig,
    ModelProfile,
    ModelPointers,
    ProjectConfig,
    ProviderType,
)


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_from_string_anthropic(self):
        assert ProviderType.from_string("anthropic") == ProviderType.ANTHROPIC

    def test_from_string_openai(self):
        assert ProviderType.from_string("openai") == ProviderType.OPENAI

    def test_from_string_gemini(self):
        assert ProviderType.from_string("gemini") == ProviderType.GEMINI

    def test_from_string_alias_claude(self):
        assert ProviderType.from_string("claude") == ProviderType.ANTHROPIC

    def test_from_string_alias_gpt(self):
        assert ProviderType.from_string("gpt") == ProviderType.OPENAI

    def test_from_string_alias_google(self):
        assert ProviderType.from_string("google") == ProviderType.GEMINI


class TestModelProfile:
    """Tests for ModelProfile."""

    def test_default_values(self):
        profile = ModelProfile()
        assert profile.provider == ProviderType.ANTHROPIC
        assert profile.model == "claude-sonnet-4-20250514"
        assert profile.max_tokens == 16384
        assert profile.temperature == 0.7

    def test_custom_values(self):
        profile = ModelProfile(
            provider=ProviderType.OPENAI,
            model="gpt-4",
            max_tokens=8192,
            temperature=0.5,
        )
        assert profile.provider == ProviderType.OPENAI
        assert profile.model == "gpt-4"
        assert profile.max_tokens == 8192
        assert profile.temperature == 0.5


class TestGlobalConfig:
    """Tests for GlobalConfig."""

    def test_default_values(self):
        config = GlobalConfig()
        assert config.model_profiles == {}
        assert config.safe_mode is True
        assert config.verbose is False
        assert config.has_completed_onboarding is False


class TestProjectConfig:
    """Tests for ProjectConfig."""

    def test_default_values(self):
        config = ProjectConfig()
        assert config.allowed_tools == []
        assert config.bash_allow_rules == []
        assert config.has_trust_dialog_accepted is False


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_get_global_config_creates_default(self):
        manager = ConfigManager()
        # Use a temp path
        manager.global_config_path = Path(tempfile.gettempdir()) / "test_ultrathink.json"
        if manager.global_config_path.exists():
            manager.global_config_path.unlink()

        config = manager.get_global_config()
        assert isinstance(config, GlobalConfig)

    def test_save_and_load_global_config(self):
        manager = ConfigManager()
        manager.global_config_path = Path(tempfile.gettempdir()) / "test_ultrathink.json"

        # Save config
        config = GlobalConfig(safe_mode=False, verbose=True)
        manager.save_global_config(config)

        # Clear cache and reload
        manager._global_config = None
        loaded = manager.get_global_config()

        assert loaded.safe_mode is False
        assert loaded.verbose is True

        # Cleanup
        if manager.global_config_path.exists():
            manager.global_config_path.unlink()

    def test_get_project_config_creates_default(self):
        manager = ConfigManager()
        config = manager.get_project_config(Path("/nonexistent"))
        assert isinstance(config, ProjectConfig)

    def test_get_model_string_default(self):
        manager = ConfigManager()
        manager._global_config = GlobalConfig()
        model_string = manager.get_model_string()
        assert model_string == "anthropic:claude-sonnet-4-20250514"

    def test_get_model_string_with_profile(self):
        manager = ConfigManager()
        manager._global_config = GlobalConfig(
            model_profiles={
                "custom": ModelProfile(
                    provider=ProviderType.OPENAI,
                    model="gpt-4",
                )
            },
            model_pointers=ModelPointers(main="custom"),
        )
        model_string = manager.get_model_string()
        assert model_string == "openai:gpt-4"
