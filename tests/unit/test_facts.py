from pathlib import Path

import pytest

from csfd.facts import (
    CaseFacts,
    crm_view,
    customer_view,
    draw_case_facts,
    identifier_mismatches,
)
from csfd.seeds.company import AssetModel, CaseFactsCatalogue, parse_company_seed

CATALOGUE = CaseFactsCatalogue(
    assets=[
        AssetModel("FX-608", "8-section IS machine", "FX608-####-??"),
        AssetModel("FX-612", "12-section IS machine", "FX612-####-??"),
    ],
    caller_roles=["maintenance technician"],
    site_locales=["en_GB"],
)


def _facts() -> CaseFacts:
    return CaseFacts(
        asset_model="FX-608",
        asset_description="8-section IS machine",
        asset_serial="FX608-0862-YR",
        customer_company="Wells-Byrne",
        site_city="North Paulaville",
        site_country="United Kingdom",
        caller_role="maintenance technician",
    )


def test_draw_is_deterministic_per_seed_and_slot() -> None:
    a = draw_case_facts(CATALOGUE, seed=42, slot_index=3)
    assert a == draw_case_facts(CATALOGUE, seed=42, slot_index=3)
    others = [draw_case_facts(CATALOGUE, seed=42, slot_index=i) for i in range(4, 10)]
    assert any(o.asset_serial != a.asset_serial for o in others)
    assert draw_case_facts(CATALOGUE, seed=43, slot_index=3) != a


def test_draw_uses_the_catalogue() -> None:
    facts = draw_case_facts(CATALOGUE, seed=1, slot_index=1)
    assert facts.asset_model in {"FX-608", "FX-612"}
    assert facts.asset_serial is not None
    assert facts.asset_serial.startswith(facts.asset_model.replace("-", "") + "-")
    assert len(facts.asset_serial) == len("FX608-0000-AA")
    assert facts.asset_serial[-2:].isupper()
    assert facts.caller_role == "maintenance technician"
    assert facts.site_country == "United Kingdom"
    assert facts.customer_company and facts.site_city


def test_draw_prefers_the_model_the_problem_names() -> None:
    for slot in range(10):
        facts = draw_case_facts(
            CATALOGUE, seed=7, slot_index=slot, problem_text="The FX-612 trips its alarm."
        )
        assert facts.asset_model == "FX-612"


def test_draw_matches_unhyphenated_model_names_in_the_problem() -> None:
    for text in ("The FX612 unit trips.", "Our fx 612 trips."):
        for slot in range(10):
            facts = draw_case_facts(CATALOGUE, seed=1, slot_index=slot, problem_text=text)
            assert facts.asset_model == "FX-612"
            assert identifier_mismatches([text], facts, CATALOGUE) == []


def test_draw_without_catalogue_still_has_a_site() -> None:
    facts = draw_case_facts(CaseFactsCatalogue(), seed=7, slot_index=1)
    assert facts.asset_model is None and facts.asset_serial is None
    assert facts.caller_role is None
    assert facts.customer_company and facts.site_country


def test_views_split_customer_and_crm_facts() -> None:
    facts = _facts()
    assert customer_view(facts) == {
        "role": "maintenance technician",
        "company": "Wells-Byrne",
        "site": "North Paulaville, United Kingdom",
        "asset_model": "FX-608",
        "asset_serial": "FX608-0862-YR",
    }
    crm = crm_view(facts)
    assert crm["account"] == "Wells-Byrne"
    assert crm["contact_role"] == "maintenance technician"
    assert crm["asset_description"] == "8-section IS machine"


def test_matching_identifiers_pass() -> None:
    texts = [
        "Our FX-608 keeps tripping. Two FX-608s on site, actually.",
        "Serial is FX608-0862-YR, or fx608 0862 yr as I read it.",
        "Thanks, I see the FX 608 on your account.",
    ]
    assert identifier_mismatches(texts, _facts(), CATALOGUE) == []


def test_invented_model_and_serial_are_reported() -> None:
    texts = [
        "It's the FX-4400X, serial FX608-9999-ZZ.",
        "Right, the FX-612. And again the FX-4400X.",
    ]
    issues = identifier_mismatches(texts, _facts(), CATALOGUE)
    assert len(issues) == 3
    assert "FX608-9999-ZZ" in issues[0] and "FX608-0862-YR" in issues[0]
    assert "FX-4400X" in issues[1] and "FX-608" in issues[1]
    assert "FX-612" in issues[2]


def test_short_family_tokens_are_not_read_as_machines() -> None:
    texts = [
        "I'll call back at 3pm, FX 15 minutes from now.",
        "Replace part FX-12 on the sensor board; check FX1 and FX2 on the mechanism.",
    ]
    assert identifier_mismatches(texts, _facts(), CATALOGUE) == []


def test_no_asset_means_nothing_to_check() -> None:
    facts = _facts().model_copy(update={"asset_model": None, "asset_serial": None})
    assert identifier_mismatches(["the FX-4400X"], facts, CATALOGUE) == []


@pytest.mark.parametrize(
    ("company", "model", "prose"),
    [
        (
            "kalvora",
            "CF-600",
            "The glaze firing at 770 degrees came out glossy; the silver sample melts at 961. "
            "We fired 12 crowns on tray 2, program P-2025.2, lot V24117, twice at 3 pm.",
        ),
        (
            "norrholt",
            "FX-608",
            "Section 5 throws checks, 8% rejects, gob weight 385 g at 1,150 degrees. "
            "Job 4471 is the 330 ml bottle; the takeout on angle moved by 4 degrees on shift 2.",
        ),
    ],
)
def test_shipped_seeds_draw_their_own_facts(company: str, model: str, prose: str) -> None:
    catalogue = parse_company_seed(Path("seeds") / company / "company_seed.md").case_facts
    facts = draw_case_facts(catalogue, seed=42, slot_index=1, problem_text=f"{model} alarm")
    assert facts.asset_model == model
    said = f"My {model}, serial {facts.asset_serial}. {prose}"
    assert identifier_mismatches([said], facts, catalogue) == []
