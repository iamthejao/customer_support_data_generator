from pathlib import Path

import pytest

from csfd.seeds.company import (
    CompanyProfile,
    SeedDocument,
    SeedErrorCode,
    SeedPart,
    SeedSpec,
    parse_company_seed,
)


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
    seed.write_text("# Kalvora Dental — Company Profile (Seed File)\n\n## About\nx\n")
    assert parse_company_seed(seed).name == "Kalvora Dental"


SHIPPED = [
    ("kalvora", "Kalvora Dental", "CF-600"),
    ("norrholt", "Norrholt Glass Machinery", "FX-608"),
]


@pytest.mark.parametrize(("company", "name", "_model"), SHIPPED)
def test_shipped_company_seed_name_is_speakable(company: str, name: str, _model: str) -> None:
    assert parse_company_seed(Path("seeds") / company / "company_seed.md").name == name


def test_parse_company_seed_reads_case_facts_catalogue() -> None:
    catalogue = parse_company_seed(Path("tests/fixtures/tiny_company_seed.md")).case_facts
    assert [a.model for a in catalogue.assets] == ["AX-100", "AX-200"]
    assert catalogue.assets[0].serial_format == "AX100-#####-?"
    assert catalogue.assets[0].description == "Acme edge appliance"
    assert catalogue.caller_roles == ["database administrator", "IT operations lead"]
    assert catalogue.site_locales == ["en_GB"]


@pytest.mark.parametrize(("company", "_name", "model"), SHIPPED)
def test_shipped_company_seed_lists_its_models(company: str, _name: str, model: str) -> None:
    catalogue = parse_company_seed(Path("seeds") / company / "company_seed.md").case_facts
    assert model in [a.model for a in catalogue.assets]
    assert all(a.serial_format for a in catalogue.assets)
    assert catalogue.caller_roles


def test_parse_company_seed_without_case_facts_is_empty(tmp_path: Path) -> None:
    seed = tmp_path / "company_seed.md"
    seed.write_text("# Acme\n\n## About\nx\n")
    catalogue = parse_company_seed(seed).case_facts
    assert catalogue.assets == [] and catalogue.caller_roles == [] and catalogue.site_locales == []


def test_parse_company_seed_reads_registry_tables() -> None:
    catalogue = parse_company_seed(Path("tests/fixtures/registry_company_seed.md")).case_facts
    assert catalogue.parts == [
        SeedPart(
            model="AX-100",
            name="Power supply unit, 240 W",
            part_number="AC-PSU-001",
            item=4,
            serials="AX100-00001 to AX100-04999",
        ),
        SeedPart(model="AX-100", name="Fan tray, front"),
    ]
    assert catalogue.specs == [SeedSpec(model="AX-100", name="Rated power", value="240 W")]
    assert catalogue.error_codes == [
        SeedErrorCode(
            model="AX-100", meaning="Fan tray failure", code="F-01", action="Replace the fan tray"
        ),
        SeedErrorCode(model="AX-100", meaning="Disk nearly full", action="Free disk space"),
    ]
    assert catalogue.documents == [
        SeedDocument(
            model="AX-100",
            doc_type="service_manual",
            doc_number="AC-SVC-100",
            revisions=("A", "B", "C"),
        )
    ]
