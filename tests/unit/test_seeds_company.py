from pathlib import Path

from csfd.seeds.company import CompanyProfile, parse_company_seed


def test_parse_company_seed_extracts_name_and_sections() -> None:
    profile: CompanyProfile = parse_company_seed(Path("tests/fixtures/tiny_company_seed.md"))
    assert profile.name == "Acme Cloud"
    assert "AcmeDB" in profile.raw_markdown
    assert "Products" in profile.sections
    assert "Tone of voice" in profile.sections


def test_parse_company_seed_preserves_source_path() -> None:
    profile = parse_company_seed(Path("tests/fixtures/tiny_company_seed.md"))
    assert profile.source_path.name == "tiny_company_seed.md"
