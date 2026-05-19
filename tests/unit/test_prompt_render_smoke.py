"""Render smoke tests for all four agent prompts.

Each test renders a template for a representative `inputs` payload and
asserts that:
  - rendering succeeds (no Jinja error),
  - a branch-specific marker phrase appears in the output.

Marker phrases are short, distinctive strings that the rewritten
templates MUST contain in the corresponding conditional branch. They
double as a contract between this test and the templates.
"""

from pathlib import Path

import pytest

from csfd.prompts.registry import PromptRegistry


@pytest.fixture(scope="module")
def registry() -> PromptRegistry:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return reg


def _base_problem_inputs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "company_name": "CoolTherm",
        "target_complexity": "medium",
        "company_profile": "Industrial chiller manufacturer.",
        "scenarios": "Standard CS scenarios.",
    }
    base.update(overrides)
    return base


def _base_resolution_inputs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ticket_type": "l1",
        "customer_tier": "standard",
        "customer_tone": "neutral",
        "target_turn_count": 3,
        "customer_name": "Alice",
        "agent_name": "Bob",
        "problem": {
            "title": "Chiller intermittently shuts off",
            "summary": "Unit power-cycles every 20 minutes.",
        },
        "resolution_hint": "Check ambient temperature sensor.",
    }
    base.update(overrides)
    return base


# ---------- Phase 1: problem_brainstorm_v2 ----------


@pytest.mark.parametrize("complexity", ["simple", "medium", "complex"])
def test_problem_generator_complexity_branches(registry: PromptRegistry, complexity: str) -> None:
    handle = registry.get("phase1.problem_brainstorm_v2")
    rendered = handle.template.render(inputs=_base_problem_inputs(target_complexity=complexity))
    assert "target_complexity" in rendered
    assert complexity in rendered
    markers = {
        "simple": "single self-contained issue",
        "medium": "multi-step troubleshooting",
        "complex": "deep domain expertise",
    }
    assert markers[complexity] in rendered


def test_problem_generator_includes_rule_ids(registry: PromptRegistry) -> None:
    handle = registry.get("phase1.problem_brainstorm_v2")
    rendered = handle.template.render(inputs=_base_problem_inputs())
    for rid in (
        "complexity_mismatch",
        "background_implausible",
        "scenario_unrealistic",
        "resolution_hints_invalid",
        "category_off_topic",
        "complexity_voice_mismatch",
    ):
        assert rid in rendered, f"generator missing rule id {rid}"


# ---------- Phase 1: problem_combined_check ----------


def test_problem_checker_rule_ids_match_generator(registry: PromptRegistry) -> None:
    handle = registry.get("phase1.problem_combined_check")
    rendered = handle.template.render(inputs=_base_problem_inputs())
    for rid in (
        "complexity_mismatch",
        "background_implausible",
        "scenario_unrealistic",
        "resolution_hints_invalid",
        "category_off_topic",
        "complexity_voice_mismatch",
    ):
        assert rid in rendered, f"checker missing rule id {rid}"


# ---------- Phase 2: resolution_generator ----------


@pytest.mark.parametrize(
    "ticket_type,marker",
    [
        ("docs_request", "documentation pointer"),
        ("l1", "first-line"),
        ("l2", "methodical troubleshooting"),
        ("l3", "hypothesis-driven"),
    ],
)
def test_resolution_generator_ticket_type_branches(
    registry: PromptRegistry, ticket_type: str, marker: str
) -> None:
    handle = registry.get("phase2.resolution_generator")
    rendered = handle.template.render(inputs=_base_resolution_inputs(ticket_type=ticket_type))
    assert marker in rendered


@pytest.mark.parametrize(
    "tier,marker",
    [
        ("standard", "standard entitlements"),
        ("premium", "premium entitlements"),
        ("enterprise", "enterprise entitlements"),
    ],
)
def test_resolution_generator_tier_branches(
    registry: PromptRegistry, tier: str, marker: str
) -> None:
    handle = registry.get("phase2.resolution_generator")
    rendered = handle.template.render(inputs=_base_resolution_inputs(customer_tier=tier))
    assert marker in rendered


@pytest.mark.parametrize(
    "tone,marker",
    [
        ("neutral", "matter-of-fact"),
        ("polite", "courteous"),
        ("frustrated", "visibly frustrated"),
        ("urgent", "time pressure"),
    ],
)
def test_resolution_generator_tone_branches(
    registry: PromptRegistry, tone: str, marker: str
) -> None:
    handle = registry.get("phase2.resolution_generator")
    rendered = handle.template.render(inputs=_base_resolution_inputs(customer_tone=tone))
    assert marker in rendered


def test_resolution_generator_unknown_value_falls_through(
    registry: PromptRegistry,
) -> None:
    """Unknown tone must trigger the fallthrough branch, not blow up."""
    handle = registry.get("phase2.resolution_generator")
    rendered = handle.template.render(inputs=_base_resolution_inputs(customer_tone="sardonic"))
    assert "sardonic" in rendered
    assert "match the named tone" in rendered


def test_resolution_generator_includes_rule_ids(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.resolution_generator")
    rendered = handle.template.render(inputs=_base_resolution_inputs())
    for rid in (
        "turn_count",
        "alternation",
        "first_turn_parity",
        "persona_match",
        "tier_tone_consistency",
        "on_topic_resolution",
        "naming",
        "length_band",
        "agent_overreach",
    ):
        assert rid in rendered, f"generator missing rule id {rid}"


# ---------- Phase 2: resolution_combined_check ----------


def test_resolution_checker_rule_ids_match_generator(
    registry: PromptRegistry,
) -> None:
    handle = registry.get("phase2.resolution_combined_check")
    rendered = handle.template.render(inputs=_base_resolution_inputs())
    for rid in (
        "turn_count",
        "alternation",
        "first_turn_parity",
        "persona_match",
        "tier_tone_consistency",
        "on_topic_resolution",
        "naming",
        "length_band",
        "agent_overreach",
    ):
        assert rid in rendered, f"checker missing rule id {rid}"
