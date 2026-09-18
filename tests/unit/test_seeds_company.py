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


def test_parse_company_seed_drops_document_title_suffix(tmp_path: Path) -> None:
    seed = tmp_path / "company_seed.md"
    seed.write_text(
        "# CoolTherm Industrial Chillers — Company Profile (Seed File)\n\n## About\nx\n"
    )
    assert parse_company_seed(seed).name == "CoolTherm Industrial Chillers"


def test_shipped_company_seed_name_is_speakable() -> None:
    assert parse_company_seed(Path("seeds/company_seed.md")).name == "CoolTherm Industrial Chillers"
