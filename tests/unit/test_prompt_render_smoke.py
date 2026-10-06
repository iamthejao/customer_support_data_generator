"""Render smoke tests for all four agent prompts.

Each test renders a template for a representative `inputs` payload and
asserts that:
  - rendering succeeds (no Jinja error),
  - a branch-specific marker phrase appears in the output.

Marker phrases are short, distinctive strings that the rewritten
templates MUST contain in the corresponding conditional branch. They
double as a contract between this test and the templates.
"""

from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

from csfd.diagnosis import (
    CandidateCause,
    ContactBeat,
    DiagnosisPlan,
    DiagnosticCheck,
    agent_guide,
    customer_findings,
)
from csfd.graph.phase2_graph import EndedLabel
from csfd.outcomes import ProblemState
from csfd.prompts.registry import PromptRegistry


@pytest.fixture(scope="module")
def registry() -> PromptRegistry:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return reg


def _base_problem_inputs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "company_name": "Kalvora",
        "target_complexity": "medium",
        "company_profile": "Dental equipment manufacturer.",
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
            "company_name": "Kalvora",
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
            "company_name": "Kalvora",
            "customer_tone": "frustrated",
            "symptoms": ["fan rattles"],
            "customer_impact": "degraded",
            "conversation_so_far": [{"speaker": "agent", "content": "Can you describe the noise?"}],
            "turn_index": 3,
            "turn_cap": 20,
        }
    )
    assert "Can you describe the noise?" in rendered


_PLAN = DiagnosisPlan(
    candidate_causes=[
        CandidateCause(
            cause="loose PSU connector",
            is_root_cause=True,
            resolution_steps=["Reseat the PSU connector until it clicks"],
            parts=[],
            verification="Run the unit for an hour",
            verification_finding="no restarts in an hour",
        ),
        CandidateCause(
            cause="failing PSU",
            resolution_steps=["Replace the PSU module"],
            parts=["PSU module"],
            verification="Run the unit for an hour",
        ),
        CandidateCause(
            cause="mains brownout",
            resolution_steps=["Move the unit to a UPS-backed socket"],
            workaround="Run the unit from a portable UPS",
        ),
    ],
    checks=[
        DiagnosticCheck(
            check="Mains check",
            how_to_check="Plug a lamp into the same socket and watch it",
            finding="the lamp stays steady",
            rules_out=["mains brownout"],
        ),
        DiagnosticCheck(
            check="Connector check",
            how_to_check="Wiggle the PSU plug at the back of the unit",
            finding="the display flickers when the plug moves",
            rules_out=["failing PSU"],
            confirms_cause=True,
        ),
    ],
)


def _guide(state: ProblemState, **beat: object) -> dict[str, object]:
    return agent_guide(
        _PLAN,
        ContactBeat(problem_state=state, **beat),
        done_checks=[],
        seed_label="t",
    )


def test_agent_turn_renders_the_guide_not_the_root_cause(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.agent_turn")
    inputs = {
        "company_name": "Kalvora",
        "agent_name": "Bob",
        "ticket_type": "l2",
        "conversation_so_far": [{"speaker": "customer", "content": "It keeps restarting."}],
        "turn_index": 2,
        "turn_cap": 20,
    }
    early = handle.template.render(
        inputs={
            **inputs,
            "guide": _guide(ProblemState.PENDING_CUSTOMER_TEST, checks=[0], next_check=1),
            "planned": {
                "problem_state": "pending_customer_test",
                "ending": "agreed_next_step",
                "cause_confirmed": False,
                "final": False,
                "revealed": True,
            },
        },
        prior_issues=[],
    )
    assert "Wiggle the PSU plug at the back of the unit" in early
    assert "the display flickers" not in early  # the customer's finding, not the agent's
    # Every candidate cause comes with its fix, so a fix does not mark the true cause.
    for fix in ("Reseat the PSU connector", "Replace the PSU module", "UPS-backed socket"):
        assert fix in early
    assert "A workaround that keeps the customer working meanwhile: Run the unit" in early
    assert "Work through check 1 of the guide" in early
    assert "runs check 2 (Connector check)" in early
    final = handle.template.render(
        inputs={
            **inputs,
            "guide": _guide(ProblemState.FIXED_VERIFIED, checks=[1], cause_confirmed=True),
            "planned": {
                "problem_state": "fixed_verified",
                "ending": "customer_satisfied",
                "cause_confirmed": True,
                "final": True,
                "revealed": True,
            },
        },
        prior_issues=[],
    )
    assert "Reseat the PSU connector until it clicks" in final
    assert "How to confirm the fix worked: Run the unit for an hour" in final
    assert 'done_reason="resolved"' in final
    # Before the diagnosis is done, the agent is not told how the contact ends.
    hidden = handle.template.render(
        inputs={
            **inputs,
            "guide": _guide(ProblemState.PENDING_PART, checks=[0, 1], cause_confirmed=True),
            "planned": {"final": True, "revealed": False},
        },
        prior_issues=[],
    )
    assert "set `diagnosis_done=true`" in hidden
    assert "part" not in hidden.split("Plan for this contact")[1].split("Conversation so far")[0]


def test_customer_turn_renders_findings_and_planned_ending(registry: PromptRegistry) -> None:
    rendered = registry.get("phase2.customer_turn").template.render(
        inputs={
            "company_name": "Kalvora",
            "customer_tone": "neutral",
            "customer_impact": "degraded",
            "symptoms": ["it restarts"],
            "conversation_so_far": [],
            "turn_index": 3,
            "turn_cap": 20,
            "diagnosis": customer_findings(
                _PLAN,
                ContactBeat(checks=[0, 1], problem_state=ProblemState.PENDING_PART),
                done_checks=[],
            ),
            "planned": {
                "problem_state": "pending_part",
                "ending": "agreed_next_step",
                "cause_confirmed": True,
                "final": True,
            },
        }
    )
    assert "you find: the display flickers when the plug moves" in rendered
    assert "loose PSU connector" not in rendered
    assert 'done_reason="follow_up"' in rendered


def test_agent_turn_renders_prior_issues_on_retry(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.agent_turn")
    rendered = handle.template.render(
        inputs={
            "company_name": "Kalvora",
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
            "company_name": "Kalvora",
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
        "company_name": "Kalvora",
        "agent_greeting": "Thank you for calling Kalvora support, this is Agent-l1-0001.",
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
    assert "AGENT: Thank you for calling Kalvora support, this is Agent-l1-0001." in rendered


def test_phone_customer_turn_renders_uppercase_speakers(registry: PromptRegistry) -> None:
    rendered = registry.get("phase2.phone_customer_turn").template.render(
        inputs=_phone_customer_inputs("light")
    )
    assert "AGENT: Can I get the serial number?" in rendered


def _phone_agent_inputs(disfluency: str) -> dict[str, object]:
    return {
        "company_name": "Kalvora",
        "agent_name": "Agent-l2-0004",
        "ticket_type": "l2",
        "guide": _guide(ProblemState.FIXED_VERIFIED, checks=[0, 1], cause_confirmed=True),
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
        assert "Wiggle the PSU plug at the back of the unit" in rendered
        assert "the display flickers" not in rendered
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


# ---------- Phase 2: how an earlier contact of the same case ended ----------

_ENDED_LABELS = ("dropped", "cap_hit", "frustrated", "follow_up", "unresolved")


def _email_customer_inputs() -> dict[str, object]:
    return {
        "company_name": "Kalvora",
        "customer_name": "Customer-standard-0001",
        "customer_tier": "standard",
        "customer_tone": "neutral",
        "ticket_type": "l1",
        "category": "cooling",
        "symptoms": ["unit power-cycles every 20 minutes"],
        "customer_impact": "degraded",
        "conversation_so_far": [{"speaker": "agent", "content": "Can you describe the noise?"}],
        "turn_index": 3,
        "turn_cap": 20,
    }


def _email_agent_inputs() -> dict[str, object]:
    return {
        "company_name": "Kalvora",
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
        "conversation_so_far": [{"speaker": "customer", "content": "It keeps tripping."}],
        "turn_index": 3,
        "turn_cap": 20,
        "prior_issues": [],
    }


def _checker_inputs() -> dict[str, object]:
    return {
        "company_name": "Kalvora",
        "customer_tone": "neutral",
        "problem": {"symptoms": [], "root_cause": []},
        "candidate": {"subject": "s", "body": "b", "end_reason": "agent_done", "turns": []},
    }


_CASE_HISTORY_PROMPTS: dict[str, Callable[[], dict[str, object]]] = {
    "phase2.incoming_request": _email_customer_inputs,
    "phase2.customer_turn": _email_customer_inputs,
    "phase2.agent_turn": _email_agent_inputs,
    "phase2.phone_incoming_request": lambda: _phone_customer_inputs("light"),
    "phase2.phone_customer_turn": lambda: _phone_customer_inputs("light"),
    "phase2.phone_agent_turn": lambda: _phone_agent_inputs("light"),
    "phase2.conversation_consistency_check": _checker_inputs,
}


def _with_case_history(base: dict[str, object], ended: str) -> dict[str, object]:
    return {
        **base,
        "round": {"sequence": 2, "count": 2, "end_mode": "final", "since_previous": "about a day"},
        "case_history": [
            {
                "sequence": 1,
                "when": "Mon 05 Jan 2026, 09:00",
                "ended": ended,
                "turns": [{"speaker": "customer", "content": "It keeps restarting."}],
            }
        ],
    }


def test_case_history_labels_cover_every_value_the_graph_emits() -> None:
    assert set(get_args(EndedLabel)) == set(_ENDED_LABELS)


@pytest.mark.parametrize("prompt", sorted(_CASE_HISTORY_PROMPTS))
def test_case_history_end_labels_render_distinctly(registry: PromptRegistry, prompt: str) -> None:
    """Each way an earlier contact can end reaches the prompt as its own description."""
    template = registry.get(prompt).template
    base = _CASE_HISTORY_PROMPTS[prompt]()
    renders = {
        label: template.render(inputs=_with_case_history(base, label)) for label in _ENDED_LABELS
    }
    assert len(set(renders.values())) == len(_ENDED_LABELS)
    for rendered in renders.values():
        assert "It keeps restarting." in rendered


@pytest.mark.parametrize("prompt", sorted(_CASE_HISTORY_PROMPTS))
def test_single_contact_render_has_no_case_history(registry: PromptRegistry, prompt: str) -> None:
    """A case with one contact renders exactly as it did before rounds existed."""
    template = registry.get(prompt).template
    base = _CASE_HISTORY_PROMPTS[prompt]()
    assert template.render(inputs=base) == template.render(
        inputs={**base, "round": None, "case_history": []}
    )


# ---------- Phase 2: case facts ----------

_CUSTOMER_FACTS = {
    "role": "maintenance technician",
    "company": "Wells-Byrne",
    "site": "North Paulaville, United Kingdom",
    "asset_model": "CF-600",
    "asset_serial": "CF600-0862-YR",
}
_CRM_FACTS = {
    "account": "Wells-Byrne",
    "site": "North Paulaville, United Kingdom",
    "contact_role": "maintenance technician",
    "asset_model": "CF-600",
    "asset_description": "Ceramic firing furnace",
    "asset_serial": "CF600-0862-YR",
}
_CUSTOMER_PROMPTS = [
    "phase2.incoming_request",
    "phase2.customer_turn",
    "phase2.phone_incoming_request",
    "phase2.phone_customer_turn",
]


@pytest.mark.parametrize("prompt", _CUSTOMER_PROMPTS)
def test_customer_prompts_show_the_case_facts(registry: PromptRegistry, prompt: str) -> None:
    template = registry.get(prompt).template
    base = _phone_customer_inputs("light")
    rendered = template.render(
        inputs={**base, "facts": _CUSTOMER_FACTS, "fact_issues": ["named the CT-4400X"]}
    )
    assert "You are the maintenance technician at Wells-Byrne" in rendered
    assert "CF-600, serial number CF600-0862-YR" in rendered
    assert "named the CT-4400X" in rendered
    assert "you may make up" not in rendered
    # Without facts (a hand-built slot) the prompt renders as before.
    assert "About you and your machine" not in template.render(inputs=base)


@pytest.mark.parametrize("prompt", ["phase2.agent_turn", "phase2.phone_agent_turn"])
def test_agent_prompts_show_the_crm_record(registry: PromptRegistry, prompt: str) -> None:
    template = registry.get(prompt).template
    base = _phone_agent_inputs("light")
    rendered = template.render(inputs={**base, "facts": _CRM_FACTS})
    assert "Account: Wells-Byrne (North Paulaville, United Kingdom)" in rendered
    assert (
        "Installed machine: CF-600 (Ceramic firing furnace), serial number CF600-0862-YR"
        in rendered
    )
    assert "CRM record" not in template.render(inputs=base)


def test_consistency_check_sees_background_and_case_record(registry: PromptRegistry) -> None:
    rendered = registry.get("phase2.conversation_consistency_check").template.render(
        inputs={
            "customer_tone": "neutral",
            "problem": {
                "title": "t",
                "complexity": "simple",
                "background": "Relocated six months ago.",
                "symptoms": ["s"],
                "root_cause": ["r"],
                "resolution_hint": "h",
            },
            "facts": {
                "asset_model": "CF-600",
                "asset_serial": "CF600-0862-YR",
                "customer_company": "Wells-Byrne",
                "site_city": "North Paulaville",
                "site_country": "United Kingdom",
                "caller_role": "plant engineer",
            },
            "candidate": {"subject": "s", "body": "b", "end_reason": "agent_done", "turns": []},
        }
    )
    assert "Relocated six months ago." in rendered
    assert "Machine: CF-600, serial number CF600-0862-YR" in rendered
    assert "plant engineer at Wells-Byrne, North Paulaville, United Kingdom" in rendered
    assert "(l) Identifiers, dates and history" in rendered


def test_consistency_check_shows_the_plan_and_asks_for_the_state(registry: PromptRegistry) -> None:
    rendered = registry.get("phase2.conversation_consistency_check").template.render(
        inputs={
            "customer_tone": "neutral",
            "problem": {"symptoms": [], "root_cause": ["loose PSU connector"]},
            "diagnosis": {
                "plan": _PLAN.model_dump(mode="json"),
                "confirming_step": 2,
                "done_steps": [],
                "beat_steps": [1, 2],
                "next_step": None,
            },
            "planned": {"problem_state": "pending_part", "ending": "agreed_next_step"},
            "candidate": {
                "subject": "s",
                "body": "b",
                "end_reason": "agent_done",
                "turns": [
                    {
                        "speaker": "agent",
                        "content": "I'll ship a cable.",
                        "done": True,
                        "done_reason": "follow_up",
                        "commitments": [
                            {"who": "agent", "what": "ship a cable", "due": "tomorrow"}
                        ],
                    }
                ],
            },
        }
    )
    assert "customer finds: the display flickers when the plug moves" in rendered
    assert "[confirms the root cause]" in rendered
    assert "problem state `pending_part`" in rendered
    assert "[recorded commitments: agent — ship a cable (due: tomorrow)]" in rendered
    assert "- (o) Report in `problem_state`" in rendered


# ---------- Identifier registry ----------


def test_problem_generator_registry_block_only_with_a_registry(registry: PromptRegistry) -> None:
    handle = registry.get("phase1.problem_brainstorm_v2")
    plain = handle.template.render(inputs=_base_problem_inputs())
    assert "identifier registry" not in plain and "unknown_identifier" not in plain
    view = [
        {
            "model": "CF-600",
            "parts": [{"part_number": "KD-60-2204", "name": "Thermocouple, type S"}],
            "error_codes": [{"code": "E-21", "meaning": "Over-temperature"}],
        }
    ]
    rendered = handle.template.render(inputs=_base_problem_inputs(registry=view))
    assert "identifier registry" in rendered
    assert '"KD-60-2204 Thermocouple, type S"' in rendered
    assert "`unknown_identifier`" in rendered


def test_registry_names_asks_only_for_what_the_seed_lacks(registry: PromptRegistry) -> None:
    handle = registry.get("documents.registry_names")
    inputs = {
        "company_name": "Kalvora Dental",
        "model": "CF-600",
        "description": "Ceramic firing furnace",
        "need": ["menus"],
        "seed_parts": ["Door seal"],
        "seed_error_codes": [],
    }
    rendered = handle.template.render(inputs=inputs)
    assert "- `menus`:" in rendered
    assert "- `parts`:" not in rendered and "- `error_codes`:" not in rendered
    assert "already list some of this model's parts" in rendered
    full = handle.template.render(
        inputs={**inputs, "need": ["parts", "error_codes", "menus"], "seed_parts": []}
    )
    assert "- `parts`:" in full and "- `error_codes`:" in full
    assert "already list" not in full
