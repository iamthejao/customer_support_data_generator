"""The identifier registry: seeded numbers, seed tables, and the unknown-identifier check."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from csfd.documents.registry import (
    NamedErrorCode,
    NamedSpec,
    ProductFacts,
    RegistryNamesOutput,
    build_registry,
    cited_models,
    company_prefix,
    missing_names,
    phase1_view,
    problem_model,
    register_parts,
    unknown_identifiers,
    unregistered_part_numbers,
)
from csfd.facts import draw_case_facts
from csfd.seeds.company import (
    AssetModel,
    CaseFactsCatalogue,
    CompanyProfile,
    SeedDocument,
    SeedPart,
    parse_company_seed,
)

START = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)


def _company() -> CompanyProfile:
    return CompanyProfile(
        name="Kalvora Dental",
        raw_markdown="# Kalvora Dental",
        case_facts=CaseFactsCatalogue(
            assets=[
                AssetModel("CF-600", "Ceramic firing furnace", "CF600-####-??"),
                AssetModel("SX-1500", "Sintering furnace", "SX1500-###-??"),
            ]
        ),
    )


def _names(prefix: str = "") -> RegistryNamesOutput:
    return RegistryNamesOutput(
        parts=[f"{prefix}Thermocouple, type S", f"{prefix}Door seal", f"{prefix}Fuse, 6.3 A"],
        error_codes=[
            NamedErrorCode(kind="error", meaning="Over-temperature", action="Switch off"),
            NamedErrorCode(kind="warning", meaning="Vacuum slow", action="Check the pump"),
        ],
        specs=[
            NamedSpec(name="Max. firing temperature", value="1200 °C"),
            NamedSpec(name="Rated power", value="1.6 kW"),
        ],
        menus=["Service > Calibration > Silver test"],
    )


def _registry(seed: int = 7, company: CompanyProfile | None = None) -> dict[str, ProductFacts]:
    company = company or _company()
    names = {a.model: _names() for a in company.case_facts.assets}
    return build_registry(company, seed=seed, calendar_start=START, names=names)


def _part_numbers(registry: dict[str, ProductFacts]) -> list[str]:
    return [p.part_number for f in registry.values() for p in f.parts]


def test_same_seed_gives_the_same_registry_and_another_seed_differs() -> None:
    assert _registry(7) == _registry(7)
    assert _part_numbers(_registry(7)) != _part_numbers(_registry(8))


def test_code_assigns_numbers_in_registry_formats() -> None:
    registry = _registry()
    assert list(registry) == ["CF-600", "SX-1500"]
    cf = registry["CF-600"]
    assert cf.source == "generated"
    assert re.fullmatch(r"KD-\d\d-####", cf.part_number_format)
    family = cf.part_number_format[:6]
    assert [p.name for p in cf.parts] == ["Thermocouple, type S", "Door seal", "Fuse, 6.3 A"]
    assert all(re.fullmatch(rf"{family}\d{{4}}", p.part_number) for p in cf.parts)
    assert [c.code[:2] for c in cf.error_codes] == ["E-", "W-"]
    assert cf.menus == ["Service > Calibration > Silver test"]
    assert cf.specs == {"Max. firing temperature": "1200 °C", "Rated power": "1.6 kW"}
    # Exploded-view item numbers run from 1; no serial range unless the seed gives one.
    assert [p.item for p in cf.parts] == [1, 2, 3]
    assert all(p.serials is None for p in cf.parts)
    # Firmware versions ascend; the last is current.
    versions = [tuple(int(x) for x in v.split(".")) for v in cf.firmware]
    assert versions == sorted(versions) and len(set(versions)) == len(versions)


def test_part_numbers_are_unique_across_the_company() -> None:
    registry = _registry()
    numbers = _part_numbers(registry)
    assert len(numbers) == len(set(numbers))
    families = {f.part_number_format for f in registry.values()}
    assert len(families) == 2


def test_library_documents_are_numbered_and_dated_before_the_first_contact() -> None:
    docs = _registry()["CF-600"].documents
    assert [d.doc_type for d in docs] == [
        "operating_manual",
        "service_manual",
        "datasheet",
        "parts_list",
    ]
    assert [d.doc_number for d in docs] == [
        "KD-OM-CF600-EN",
        "KD-SM-CF600-EN",
        "KD-DS-CF600-EN",
        "KD-PL-CF600-EN",
    ]
    for d in docs:
        assert d.revisions == [chr(ord("A") + i) for i in range(len(d.revisions))]
        assert len(d.issued) == len(d.revisions)
        assert d.issued == sorted(d.issued) and d.issued[-1] < START.date()


def test_a_named_entry_keeps_its_number_whatever_else_the_model_returns() -> None:
    company = _company()
    short = build_registry(company, seed=7, calendar_start=START, names={"CF-600": _names()})
    longer_names = _names()
    longer_names.parts.append("Heating element")
    longer = build_registry(company, seed=7, calendar_start=START, names={"CF-600": longer_names})
    assert longer["CF-600"].parts[:3] == short["CF-600"].parts


def test_seed_tables_win_and_the_model_fills_the_rest() -> None:
    company = parse_company_seed(Path("tests/fixtures/registry_company_seed.md"))
    assert missing_names(company.case_facts, "AX-100") == ["menus"]
    assert missing_names(company.case_facts, "AX-200") == [
        "parts",
        "error_codes",
        "specs",
        "menus",
    ]
    names = {"AX-100": _names("Model "), "AX-200": _names()}
    registry = build_registry(company, seed=3, calendar_start=START, names=names)

    ax100 = registry["AX-100"]
    assert ax100.source == "mixed"
    # The seed's parts and codes, not the model's; seed numbers kept, the rest drawn.
    assert [p.name for p in ax100.parts] == ["Power supply unit, 240 W", "Fan tray, front"]
    assert ax100.parts[0].part_number == "AC-PSU-001"
    assert re.fullmatch(r"AC-\d\d-\d{4}", ax100.parts[1].part_number)
    # The seed's item number and serial range are kept; code gives the lowest free item.
    assert (ax100.parts[0].item, ax100.parts[0].serials) == (4, "AX100-00001 to AX100-04999")
    assert (ax100.parts[1].item, ax100.parts[1].serials) == (1, None)
    assert ax100.specs == {"Rated power": "240 W"}
    assert ax100.error_codes[0].code == "F-01"
    assert re.fullmatch(r"E-\d\d", ax100.error_codes[1].code)
    service = next(d for d in ax100.documents if d.doc_type == "service_manual")
    assert (service.doc_number, service.revisions) == ("AC-SVC-100", ["A", "B", "C"])
    assert ax100.menus == ["Service > Calibration > Silver test"]

    ax200 = registry["AX-200"]
    assert ax200.source == "generated"
    assert [p.name for p in ax200.parts] == ["Thermocouple, type S", "Door seal", "Fuse, 6.3 A"]
    assert ax200.specs == {"Max. firing temperature": "1200 °C", "Rated power": "1.6 kW"}


def test_seed_tables_naming_an_unknown_model_or_type_fail() -> None:
    company = _company()
    company.case_facts.documents.append(SeedDocument(model="CF-600", doc_type="brochure"))
    with pytest.raises(ValueError, match="brochure"):
        _registry(company=company)
    company.case_facts.documents[:] = [SeedDocument(model="ZZ-1", doc_type="datasheet")]
    with pytest.raises(ValueError, match="ZZ-1"):
        _registry(company=company)


def test_company_prefix_uses_initials() -> None:
    assert company_prefix("Kalvora Dental") == "KD"
    assert company_prefix("Norrholt Glass Machinery") == "NGM"
    assert company_prefix("Acme") == "AC"


def test_phase1_view_lists_parts_and_codes_per_model() -> None:
    registry = _registry()
    view = phase1_view(registry)
    assert [v["model"] for v in view] == ["CF-600", "SX-1500"]
    part = registry["CF-600"].parts[0]
    assert view[0]["parts"][0] == {"part_number": part.part_number, "name": part.name}
    assert set(view[0]["error_codes"][0]) == {"code", "meaning"}


def test_unknown_identifiers_accepts_registry_ids_and_flags_invented_ones() -> None:
    registry = _registry()
    catalogue = _company().case_facts
    cf, sx = registry["CF-600"], registry["SX-1500"]
    good = f"Replace {cf.parts[0].part_number}; the display shows {cf.error_codes[0].code}."
    assert unknown_identifiers([good], registry, "CF-600", catalogue) == []

    family = cf.part_number_format[:6]
    taken = {p.part_number for p in cf.parts}
    invented = next(f"{family}{n:04d}" for n in range(10_000) if f"{family}{n:04d}" not in taken)
    foreign = sx.parts[0].part_number
    text = f"Order {invented} and {foreign}; error E-99 or {cf.error_codes[0].code}."
    issues = unknown_identifiers([text], registry, "CF-600", catalogue)
    flagged = " ".join(issues)
    assert invented in flagged and foreign in flagged and "belongs to the SX-1500" in flagged
    if "E-99" not in {c.code for c in cf.error_codes}:
        assert "E-99" in flagged
    # Without a known model, any model's identifiers are allowed.
    assert foreign not in " ".join(unknown_identifiers([text], registry, None, catalogue))
    # An empty registry (documents disabled) checks nothing.
    assert unknown_identifiers([text], {}, "CF-600", catalogue) == []


def test_unknown_identifiers_ignores_serial_numbers_and_plain_numbers() -> None:
    registry = _registry()
    catalogue = _company().case_facts
    assert (
        unknown_identifiers(["Serial CF600-1234-AB, 2024-001"], registry, "CF-600", catalogue) == []
    )


def test_register_parts_canonicalises_and_numbers_new_parts() -> None:
    registry = _registry()
    cf = registry["CF-600"]
    first = cf.parts[0]
    entries = [first.part_number, "door seal", "Heating element, 230 V", "KD-00-0000 bogus"]
    updated, canonical = register_parts(registry, "CF-600", entries, seed=7)

    new = updated["CF-600"].parts[-1]
    assert new.name == "Heating element, 230 V"
    assert new.part_number not in {p.part_number for p in cf.parts}
    assert new.item == len(cf.parts) + 1
    assert re.fullmatch(rf"{cf.part_number_format[:6]}\d{{4}}", new.part_number)
    assert canonical == [first.entry, cf.parts[1].entry, new.entry, "KD-00-0000 bogus"]
    assert updated["SX-1500"] == registry["SX-1500"]
    # Registering is deterministic, and a registered part is found by name next time.
    again, _ = register_parts(registry, "CF-600", entries, seed=7)
    assert again == updated
    _, reuse = register_parts(updated, "CF-600", ["heating element, 230 V"], seed=7)
    assert reuse == [new.entry]


def test_register_parts_without_a_known_model_changes_nothing() -> None:
    registry = _registry()
    assert register_parts(registry, None, ["Fan"], seed=7) == (registry, ["Fan"])


def test_near_miss_part_numbers_are_reported_and_never_registered() -> None:
    registry = _registry()
    cf = registry["CF-600"]
    near_miss = "KD-600-2204 Thermocouple, type S"
    assert not unknown_identifiers([near_miss], registry, "CF-600", _company().case_facts)
    issues = unregistered_part_numbers([near_miss, cf.parts[0].entry], registry, "CF-600")
    assert len(issues) == 1 and "KD-600-2204" in issues[0]

    entries = [near_miss, "KD-60-22045 Fan", "Fuse, 6.3 A", "O-ring, door"]
    updated, canonical = register_parts(registry, "CF-600", entries, seed=7)
    assert canonical[:3] == [near_miss, "KD-60-22045 Fan", cf.parts[2].entry]
    # Only the plain name is registered; entries led by a near-miss number are not.
    assert [p.name for p in updated["CF-600"].parts[len(cf.parts) :]] == ["O-ring, door"]
    assert not unregistered_part_numbers(entries[2:], registry, "CF-600")


@pytest.mark.parametrize(
    "name",
    [
        "3-way solenoid valve",
        "2-stage vacuum pump",
        "5-axis spindle motor",
        "12-section manifold",
        "Pt1000 sensor",
    ],
)
def test_quantity_prefixed_part_names_are_registered_as_new_parts(name: str) -> None:
    registry = _registry()
    cf = registry["CF-600"]
    assert unregistered_part_numbers([name], registry, "CF-600") == []
    updated, canonical = register_parts(registry, "CF-600", [name], seed=7)
    new = updated["CF-600"].parts[-1]
    assert new.name == name and new.part_number not in {p.part_number for p in cf.parts}
    assert canonical == [new.entry]


def test_an_all_digit_seed_part_number_is_matched_not_registered() -> None:
    company = _company()
    company.case_facts.parts.append(SeedPart("CF-600", "Fan", part_number="1234567"))
    registry = _registry(company=company)
    fan = registry["CF-600"].parts[0]
    assert fan.part_number == "1234567"
    updated, canonical = register_parts(registry, "CF-600", ["1234567 Fan"], seed=7)
    assert canonical == [fan.entry] and updated == registry
    assert unregistered_part_numbers(["1234567 Fan"], registry, "CF-600") == []
    assert unregistered_part_numbers(["7654321 Fan"], registry, "CF-600") != []


def test_problem_model_is_the_named_model_or_the_single_owner_of_cited_ids() -> None:
    registry = _registry()
    catalogue = _company().case_facts
    cf, sx = registry["CF-600"], registry["SX-1500"]
    no_model = "Furnace does not reach firing temperature"
    # Named model wins; otherwise the one model owning every cited identifier.
    assert problem_model(f"Our SX-1500 {no_model}", [cf.parts[0].entry], registry, catalogue) == (
        "SX-1500"
    )
    assert problem_model(no_model, [cf.parts[0].entry], registry, catalogue) == "CF-600"
    assert cited_models([no_model, cf.parts[0].entry], registry, catalogue) == {"CF-600"}
    # Identifiers of two models, or none: no model.
    both = [cf.parts[0].entry, sx.parts[0].entry]
    assert cited_models([no_model, *both], registry, catalogue) == set()
    assert problem_model(no_model, both, registry, catalogue) is None
    assert cited_models([no_model, "Heating element"], registry, catalogue) is None
    assert problem_model(no_model, ["Heating element"], registry, catalogue) is None
    # Without a registry only a named model counts.
    assert problem_model(no_model, [cf.parts[0].entry], {}, catalogue) is None


def test_draw_case_facts_takes_the_known_asset_model() -> None:
    catalogue = _company().case_facts
    for slot in range(20):
        facts = draw_case_facts(catalogue, seed=7, slot_index=slot, asset_model="SX-1500")
        assert facts.asset_model == "SX-1500"
        assert facts.asset_serial is not None and facts.asset_serial.startswith("SX1500")
    # Without one, the draw is unchanged.
    assert draw_case_facts(catalogue, seed=7, slot_index=3) == draw_case_facts(
        catalogue, seed=7, slot_index=3, asset_model=None
    )
