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
    assert "runs" in table_names
    assert "problems" in table_names
    assert "kb_articles" in table_names
    assert "tickets" in table_names
    assert "turns" in table_names
    assert "agent_traces" in table_names
