import pytest

from csfd.settings import load_settings


def test_load_settings_with_default_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = load_settings(default_path="config/default.yaml", profile=None)
    assert s.pipeline.version == "0.1.0"
    assert s.phase1.problem_count == 100
    assert s.agents["generator"].provider == "anthropic"
    assert s.phase2.voting_policy == "strict_all_pass"


def test_load_settings_with_dev_profile_overrides_problem_count() -> None:
    s = load_settings(default_path="config/default.yaml", profile="dev")
    assert s.phase1.problem_count == 5
    assert s.pipeline.budget.max_usd_per_run == 1.0


def test_load_settings_with_local_only_profile_switches_providers() -> None:
    s = load_settings(default_path="config/default.yaml", profile="local-only")
    assert s.agents["generator"].provider == "openai_compat"
    assert "11434" in (s.agents["generator"].base_url or "")


def test_load_settings_unknown_profile_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(default_path="config/default.yaml", profile="does-not-exist")


def test_env_var_overrides_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    s = load_settings(default_path="config/default.yaml", profile=None)
    assert s.env.anthropic_api_key == "test-key"
