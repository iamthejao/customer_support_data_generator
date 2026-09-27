from pathlib import Path

from typer.testing import CliRunner

from csfd.cli import app

runner = CliRunner()


def test_cli_init_scaffolds_directories(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "seeds").exists()
    assert (tmp_path / "data").exists()
    assert (tmp_path / ".env.example").exists()


def test_cli_db_migrate_creates_tables(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    result = runner.invoke(
        app, ["db-migrate", "--sqlite-path", str(tmp_path / "data" / "test.sqlite")]
    )
    assert result.exit_code == 0, result.output
    import sqlite3

    with sqlite3.connect(str(tmp_path / "data" / "test.sqlite")) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {r[0] for r in rows}
    for required in (
        "runs",
        "problems",
        "incoming_requests",
        "resolutions",
        "lineage",
        "agent_traces",
    ):
        assert required in table_names


def test_cli_init_scaffolds_a_company_seed_folder(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "seeds" / "my_company" / "company_seed.md").exists()
    assert (tmp_path / "seeds" / "my_company" / "scenarios_seed.md").exists()
    assert "--company my_company" in result.output


def test_seed_paths_follow_the_configured_company() -> None:
    import pytest
    import typer

    from csfd.cli import _resolve_seed_paths
    from csfd.settings import SeedsConfig

    company, scenarios = _resolve_seed_paths(SeedsConfig(company="norrholt"), None, None)
    assert company == Path("seeds/norrholt/company_seed.md")
    assert scenarios == Path("seeds/norrholt/scenarios_seed.md")
    with pytest.raises(typer.BadParameter, match="--company"):
        _resolve_seed_paths(SeedsConfig(company="no_such_company"), None, None)
