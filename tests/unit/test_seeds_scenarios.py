from pathlib import Path

from csfd.seeds.scenarios import ScenarioCatalogue, parse_scenarios_seed


def test_parse_scenarios_extracts_categories() -> None:
    cat: ScenarioCatalogue = parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md"))
    assert len(cat.scenarios) == 4
    cats = {s.category for s in cat.scenarios}
    assert {"Billing", "Authentication", "Database"} <= cats


def test_scenario_has_summary() -> None:
    cat = parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md"))
    s = next(x for x in cat.scenarios if "overage" in x.title.lower())
    assert s.category == "Billing"
    assert "overage" in s.title.lower()
    assert s.summary
