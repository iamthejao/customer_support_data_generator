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


# ---------- Phase 2: turn-based dialogue prompts ----------


def test_incoming_request_renders_symptoms_only(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.incoming_request")
    rendered = handle.template.render(
        inputs={
            "company_name": "CoolTherm",
            "customer_name": "Alice",
            "customer_tier": "standard",
            "customer_tone": "neutral",
            "ticket_type": "l1",
            "category": "cooling",
            "symptoms": ["unit power-cycles every 20 minutes"],
            "customer_impact": "degraded",
        }
    )
    assert "unit power-cycles every 20 minutes" in rendered


def test_customer_turn_renders_history(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.customer_turn")
    rendered = handle.template.render(
        inputs={
            "company_name": "CoolTherm",
            "customer_tone": "frustrated",
            "symptoms": ["fan rattles"],
            "customer_impact": "degraded",
            "conversation_so_far": [{"speaker": "agent", "content": "Can you describe the noise?"}],
            "turn_index": 3,
            "turn_cap": 20,
        }
    )
    assert "Can you describe the noise?" in rendered


def test_agent_turn_renders_root_cause(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.agent_turn")
    rendered = handle.template.render(
        inputs={
            "company_name": "CoolTherm",
            "agent_name": "Bob",
            "ticket_type": "l2",
            "problem": {
                "title": "Brownout",
                "summary": "PSU brownout",
                "background": "loose connector",
                "category": "power",
                "fault_domain": "hardware",
                "root_cause": ["loose PSU connector"],
                "resolution_hint": "reseat the connector",
            },
            "conversation_so_far": [{"speaker": "customer", "content": "It keeps restarting."}],
            "turn_index": 2,
            "turn_cap": 20,
        },
        prior_issues=[],
    )
    assert "loose PSU connector" in rendered


def test_agent_turn_renders_prior_issues_on_retry(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.agent_turn")
    rendered = handle.template.render(
        inputs={
            "company_name": "CoolTherm",
            "agent_name": "Bob",
            "ticket_type": "l1",
            "problem": {
                "title": "t",
                "summary": "s",
                "background": "b",
                "category": "c",
                "fault_domain": "software",
                "root_cause": ["x"],
                "resolution_hint": "y",
            },
            "conversation_so_far": [],
            "turn_index": 2,
            "turn_cap": 20,
            # _generate_turn passes prior_issues inside `inputs`, not as a
            # top-level template variable.
            "prior_issues": ["customer leaked root cause"],
        },
    )
    assert "customer leaked root cause" in rendered


def test_consistency_check_renders_complexity(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.conversation_consistency_check")
    rendered = handle.template.render(
        inputs={
            "company_name": "CoolTherm",
            "customer_tone": "neutral",
            "problem": {
                "title": "T",
                "complexity": "simple",
                "symptoms": ["x"],
                "root_cause": ["y"],
                "resolution_hint": "z",
            },
            "candidate": {
                "subject": "s",
                "body": "b",
                "end_reason": "agent_done",
                "turns": [
                    {"speaker": "customer", "content": "b", "done": False, "done_reason": None}
                ],
            },
        }
    )
    assert "simple" in rendered


# ---------- Phase 2: phone-call prompts ----------


def _phone_customer_inputs(disfluency: str) -> dict[str, object]:
    return {
        "company_name": "CoolTherm",
        "agent_greeting": "Thank you for calling CoolTherm support, this is Agent-l1-0001.",
        "customer_name": "Customer-standard-0001",
        "customer_tier": "standard",
        "customer_tone": "frustrated",
        "ticket_type": "l1",
        "category": "cooling",
        "symptoms": ["unit power-cycles every 20 minutes"],
        "customer_impact": "degraded",
        "disfluency": disfluency,
        "conversation_so_far": [
            {"speaker": "agent", "content": "Can I get the serial number?"},
        ],
        "turn_index": 3,
        "turn_cap": 20,
    }


@pytest.mark.parametrize("prompt", ["phase2.phone_incoming_request", "phase2.phone_customer_turn"])
def test_phone_customer_prompts_render_disfluency_branch(
    registry: PromptRegistry, prompt: str
) -> None:
    template = registry.get(prompt).template
    renders = {
        level: template.render(inputs=_phone_customer_inputs(level))
        for level in ("none", "light", "moderate")
    }
    assert len(set(renders.values())) == 3
    for rendered in renders.values():
        assert "unit power-cycles every 20 minutes" in rendered
        assert "Customer-standard-0001" in rendered


def test_phone_incoming_request_shows_greeting(registry: PromptRegistry) -> None:
    rendered = registry.get("phase2.phone_incoming_request").template.render(
        inputs=_phone_customer_inputs("light")
    )
    assert "AGENT: Thank you for calling CoolTherm support, this is Agent-l1-0001." in rendered


def test_phone_customer_turn_renders_uppercase_speakers(registry: PromptRegistry) -> None:
    rendered = registry.get("phase2.phone_customer_turn").template.render(
        inputs=_phone_customer_inputs("light")
    )
    assert "AGENT: Can I get the serial number?" in rendered


def _phone_agent_inputs(disfluency: str) -> dict[str, object]:
    return {
        "company_name": "CoolTherm",
        "agent_name": "Agent-l2-0004",
        "ticket_type": "l2",
        "problem": {
            "title": "t",
            "summary": "s",
            "background": "b",
            "category": "c",
            "fault_domain": "hardware",
            "root_cause": ["loose PSU connector"],
            "resolution_hint": "reseat",
        },
        "disfluency": disfluency,
        "conversation_so_far": [{"speaker": "customer", "content": "It keeps tripping."}],
        "turn_index": 3,
        "turn_cap": 20,
        "prior_issues": ["agent read the root cause aloud"],
    }


def test_phone_agent_turn_renders_call_flow(registry: PromptRegistry) -> None:
    template = registry.get("phase2.phone_agent_turn").template
    renders = {
        level: template.render(inputs=_phone_agent_inputs(level))
        for level in ("none", "light", "moderate")
    }
    assert len(set(renders.values())) == 3
    for rendered in renders.values():
        assert "loose PSU connector" in rendered
        assert "CUSTOMER: It keeps tripping." in rendered
        assert "agent read the root cause aloud" in rendered


def test_consistency_check_phone_block_only_for_phone(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.conversation_consistency_check")
    base = {
        "customer_tone": "neutral",
        "problem": {"symptoms": [], "root_cause": []},
        "candidate": {"subject": "s", "body": "b", "end_reason": "agent_done", "turns": []},
    }
    phone = handle.template.render(inputs={**base, "channel": "phone"})
    email = handle.template.render(inputs={**base, "channel": "email"})
    assert phone != email
    assert email == handle.template.render(inputs=base)
