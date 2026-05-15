from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from csfd.cli import app

runner = CliRunner()


@pytest.fixture
def fake_run_phase1() -> Iterator[None]:
    async def _fake(**_kwargs: Any) -> str:
        return "fake-phase1-run-id"

    with patch("csfd.cli.run_phase1", new=_fake):
        yield


@pytest.fixture
def fake_run_phase2() -> Iterator[None]:
    async def _fake(**_kwargs: Any) -> str:
        return "fake-phase2-run-id"

    with patch("csfd.cli.run_phase2", new=_fake):
        yield


def test_phase1_run_command_emits_run_id(
    fake_run_phase1: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        [
            "phase1",
            "run",
            "--seed",
            "1",
            "--problems",
            "1",
            "--company-seed",
            "tests/fixtures/tiny_company_seed.md",
            "--scenarios-seed",
            "tests/fixtures/tiny_scenarios_seed.md",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "fake-phase1-run-id" in result.output


def test_generate_command_chains_both_phases(
    fake_run_phase1: None,
    fake_run_phase2: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        [
            "generate",
            "--seed",
            "1",
            "--problems",
            "1",
            "--company-seed",
            "tests/fixtures/tiny_company_seed.md",
            "--scenarios-seed",
            "tests/fixtures/tiny_scenarios_seed.md",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "fake-phase1-run-id" in result.output
    assert "fake-phase2-run-id" in result.output
