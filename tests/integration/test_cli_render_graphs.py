from pathlib import Path

import pytest
from typer.testing import CliRunner

from csfd.cli import app

runner = CliRunner()


def test_render_graphs_writes_mermaid_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["render-graphs", "--out", str(tmp_path / "docs/diagrams")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "docs/diagrams/phase1.mmd").exists()
    assert (tmp_path / "docs/diagrams/phase2.mmd").exists()
    text = (tmp_path / "docs/diagrams/phase1.mmd").read_text()
    # Mermaid contains nodes from the structural Phase 1 subgraph
    assert "problem" in text or "brainstorm" in text
