from pathlib import Path

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
        AssetModel("CT-500", "Mid-size chiller", "CT500-####-??"),
        AssetModel("CT-800", "Large chiller", "CT800-####-??"),
    ],
    caller_roles=["maintenance technician"],
    site_locales=["en_GB"],
)


def _facts() -> CaseFacts:
    return CaseFacts(
        asset_model="CT-500",
        asset_description="Mid-size chiller",
        asset_serial="CT500-0862-YR",
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
    assert facts.asset_model in {"CT-500", "CT-800"}
    assert facts.asset_serial is not None
    assert facts.asset_serial.startswith(facts.asset_model.replace("-", "") + "-")
    assert len(facts.asset_serial) == len("CT500-0000-AA")
    assert facts.asset_serial[-2:].isupper()
    assert facts.caller_role == "maintenance technician"
    assert facts.site_country == "United Kingdom"
    assert facts.customer_company and facts.site_city


def test_draw_prefers_the_model_the_problem_names() -> None:
    for slot in range(10):
        facts = draw_case_facts(
            CATALOGUE, seed=7, slot_index=slot, problem_text="The CT-800 trips its alarm."
        )
        assert facts.asset_model == "CT-800"


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
        "asset_model": "CT-500",
        "asset_serial": "CT500-0862-YR",
    }
    crm = crm_view(facts)
    assert crm["account"] == "Wells-Byrne"
    assert crm["contact_role"] == "maintenance technician"
    assert crm["asset_description"] == "Mid-size chiller"


def test_matching_identifiers_pass() -> None:
    texts = [
        "Our CT-500 keeps tripping. Two CT-500s on site, actually.",
        "Serial is CT500-0862-YR, or ct500 0862 yr as I read it.",
        "Thanks, I see the CT 500 on your account.",
    ]
    assert identifier_mismatches(texts, _facts(), CATALOGUE) == []


def test_invented_model_and_serial_are_reported() -> None:
    texts = [
        "It's the CT-4400X, serial CT500-9999-ZZ.",
        "Right, the CT-800. And again the CT-4400X.",
    ]
    issues = identifier_mismatches(texts, _facts(), CATALOGUE)
    assert len(issues) == 3
    assert "CT500-9999-ZZ" in issues[0] and "CT500-0862-YR" in issues[0]
    assert "CT-4400X" in issues[1] and "CT-500" in issues[1]
    assert "CT-800" in issues[2]


def test_no_asset_means_nothing_to_check() -> None:
    facts = _facts().model_copy(update={"asset_model": None, "asset_serial": None})
    assert identifier_mismatches(["the CT-4400X"], facts, CATALOGUE) == []


def test_shipped_seed_draws_cooltherm_facts() -> None:
    catalogue = parse_company_seed(Path("seeds/company_seed.md")).case_facts
    facts = draw_case_facts(catalogue, seed=42, slot_index=1, problem_text="CT-500 alarm")
    assert facts.asset_model == "CT-500"
    assert (
        identifier_mismatches([f"My CT-500, serial {facts.asset_serial}"], facts, catalogue) == []
    )
