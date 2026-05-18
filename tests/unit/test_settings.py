"""Tests for ``csfd.settings.load_settings``.

These tests exercise loader *behavior* (YAML parse, profile deep-merge,
env-var precedence, error paths) using synthetic configs written into
``tmp_path``. They intentionally avoid asserting numeric values from the
real ``config/default.yaml`` so that legitimate config tuning does not
produce spurious test failures. A single smoke test at the bottom loads
the real default config and asserts only structural invariants.
"""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import dedent

import pytest

from csfd.settings import load_settings

# ---- Synthetic YAML helpers --------------------------------------------------


_MINIMAL_DEFAULT = dedent(
    """
    pipeline:
      version: "9.9.9"
      run_seed: 1
      budget:
        max_tokens_per_run: 1000
        max_usd_per_run: 2.5
        max_retries_per_artifact: 1
    agents:
      generator:
        provider: anthropic
        model: test-model
        temperature: 0.5
        max_tokens: 100
        timeout_s: 5
    problem_database:
      count: 7
      complexity_proportions: {simple: 1.0}
    tickets:
      total: 42
      type_proportions: {l1: 1.0}
      assignment_strategy: uniform
      turns_per_type: {l1: 2}
      tier_proportions: {standard: 1.0}
      tone_proportions_per_type:
        l1: {neutral: 1.0}
    validation: {enabled: true, max_retries: 0}
    observability:
      structlog_json: false
      langsmith_enabled: false
      langsmith_project: test
      persist_stream_updates: false
    storage:
      sqlite_path: ":memory:"
      exports_dir: /tmp/exports
      exports_format: [jsonl]
    """
).strip()


def _write_default(tmp_path: Path, body: str = _MINIMAL_DEFAULT) -> Path:
    path = tmp_path / "default.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _write_profile(tmp_path: Path, name: str, body: str) -> Path:
    """Profile paths are resolved relative to CWD as ``config/profiles/<name>.yaml``."""
    profiles_dir = tmp_path / "config" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profiles_dir / f"{name}.yaml"
    path.write_text(dedent(body).strip(), encoding="utf-8")
    return path


# ---- Behavior: default-only load --------------------------------------------


def test_load_settings_parses_default_yaml_into_appsettings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    default_path = _write_default(tmp_path)

    s = load_settings(default_path=str(default_path), profile=None)

    # Round-trips arbitrary scalar/string/dict values from YAML.
    assert s.pipeline.version == "9.9.9"
    assert s.agents["generator"].provider == "anthropic"
    assert s.tickets.total == 42
    assert s.problem_database.count == 7


def test_load_settings_rejects_missing_required_sections(tmp_path: Path) -> None:
    # ``tickets`` is non-Optional in AppSettings -> omission must fail at load time.
    truncated = dedent(
        """
        pipeline: {version: "0.0.0"}
        agents:
          generator: {provider: anthropic, model: m}
        problem_database: {count: 1, complexity_proportions: {simple: 1.0}}
        validation: {enabled: true}
        observability:
          structlog_json: false
          langsmith_enabled: false
          langsmith_project: x
          persist_stream_updates: false
        storage: {sqlite_path: ":memory:", exports_dir: /tmp, exports_format: [jsonl]}
        """
    ).strip()
    path = _write_default(tmp_path, truncated)

    with pytest.raises(Exception):  # noqa: B017
        load_settings(default_path=str(path), profile=None)


# ---- Behavior: profile overlay (deep-merge) ---------------------------------


def test_profile_overlay_deep_merges_nested_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_path = _write_default(tmp_path)
    _write_profile(
        tmp_path,
        "tiny",
        """
        pipeline:
          budget:
            max_usd_per_run: 0.25
        tickets:
          total: 3
        """,
    )
    monkeypatch.chdir(tmp_path)

    s = load_settings(default_path=str(default_path), profile="tiny")

    # Overridden keys take overlay values.
    assert s.pipeline.budget.max_usd_per_run == 0.25
    assert s.tickets.total == 3
    # Sibling keys inside the same nested section are preserved from defaults.
    assert s.pipeline.budget.max_tokens_per_run == 1000
    assert s.pipeline.version == "9.9.9"
    # Unrelated top-level sections untouched.
    assert s.problem_database.count == 7


def test_unknown_profile_raises_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_path = _write_default(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        load_settings(default_path=str(default_path), profile="does-not-exist")


# ---- Behavior: env-var precedence -------------------------------------------


def test_anthropic_api_key_env_var_populates_env_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-sentinel")
    default_path = _write_default(tmp_path)

    s = load_settings(default_path=str(default_path), profile=None)

    assert s.env.anthropic_api_key == "sk-test-sentinel"


def test_missing_anthropic_api_key_leaves_env_secret_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    default_path = _write_default(tmp_path)

    s = load_settings(default_path=str(default_path), profile=None)

    assert s.env.anthropic_api_key in (None, "")


# ---- Smoke test against the real default config -----------------------------


def test_real_default_yaml_loads_and_satisfies_structural_invariants() -> None:
    """The shipped ``config/default.yaml`` must parse and remain internally consistent.

    We deliberately do *not* assert specific numeric values here so that tuning
    defaults (e.g. ``tickets.total``) does not break this test.
    """
    s = load_settings(default_path="config/default.yaml", profile=None)

    # Required sections present.
    assert s.pipeline.version
    assert "generator" in s.agents
    assert s.tickets.type_proportions
    assert s.tickets.turns_per_type
    assert s.tickets.tier_proportions

    # Proportions are well-formed (sum to ~1.0).
    assert math.isclose(sum(s.tickets.type_proportions.values()), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(s.tickets.tier_proportions.values()), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(s.problem_database.complexity_proportions.values()), 1.0, abs_tol=1e-6)
    for ticket_type, tones in s.tickets.tone_proportions_per_type.items():
        assert math.isclose(sum(tones.values()), 1.0, abs_tol=1e-6), ticket_type

    # Every ticket type referenced in type_proportions has matching detail entries.
    types = set(s.tickets.type_proportions)
    assert types <= set(s.tickets.turns_per_type)
    assert types <= set(s.tickets.tone_proportions_per_type)
