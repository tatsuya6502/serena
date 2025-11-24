import pytest

from solidlsp.language_servers.rust_analyzer import RustAnalyzer, RustAnalyzerProfile


@pytest.mark.rust
class TestRustAnalyzerProfiles:
    def test_profile_enum_values(self) -> None:
        """Test that the profile enum has the expected values."""
        assert RustAnalyzerProfile.PERFORMANCE.value == "performance"
        assert RustAnalyzerProfile.BALANCED.value == "balanced"
        assert RustAnalyzerProfile.LOW_MEMORY.value == "low-memory"

    def test_performance_profile_settings(self) -> None:
        """Test that performance profile returns empty dict (uses defaults)."""
        settings = RustAnalyzer.get_profile_settings("performance")
        assert settings == {}

    def test_balanced_profile_settings(self) -> None:
        """Test that balanced profile returns correct settings."""
        settings = RustAnalyzer.get_profile_settings("balanced")
        assert "cachePriming" in settings
        assert settings["cachePriming"]["enable"] is False
        assert settings["lru"]["capacity"] == 64
        assert settings["checkOnSave"] is False
        assert settings["check"]["allTargets"] is False

    def test_low_memory_profile_settings(self) -> None:
        """Test that low-memory profile returns correct settings."""
        settings = RustAnalyzer.get_profile_settings("low-memory")
        assert "cachePriming" in settings
        assert settings["cachePriming"]["enable"] is False
        assert settings["lru"]["capacity"] == 32
        assert settings["cargo"]["buildScripts"]["enable"] is False
        assert settings["procMacro"]["enable"] is False
        assert settings["checkOnSave"] is False
        assert settings["check"]["allTargets"] is False

    def test_invalid_profile_returns_empty_dict(self) -> None:
        """Test that an invalid profile name returns empty dict (falls back to defaults)."""
        settings = RustAnalyzer.get_profile_settings("invalid-profile")
        assert settings == {}

    def test_deep_merge_dict(self) -> None:
        """Test the deep merge functionality."""
        base = {"a": 1, "b": {"c": 2, "d": 3}, "e": 5}
        override = {"b": {"d": 4}, "f": 6}
        result = RustAnalyzer._deep_merge_dict(base, override)
        assert result == {"a": 1, "b": {"c": 2, "d": 4}, "e": 5, "f": 6}

    def test_deep_merge_nested(self) -> None:
        """Test deep merge with nested dictionaries."""
        base = {"cargo": {"buildScripts": {"enable": True, "other": "value"}}}
        override = {"cargo": {"buildScripts": {"enable": False}}}
        result = RustAnalyzer._deep_merge_dict(base, override)
        assert result["cargo"]["buildScripts"]["enable"] is False
        assert result["cargo"]["buildScripts"]["other"] == "value"
