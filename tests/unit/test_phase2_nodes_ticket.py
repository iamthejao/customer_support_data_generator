from uuid import uuid4

import pytest

from csfd.phases.phase1_kb.state import CommittedProblem, ProblemDraft
from csfd.phases.phase2_cases.nodes import ticket_init_node, ticket_sampler_node
from csfd.phases.phase2_cases.state import TicketState
from csfd.seeds.company import CompanyProfile
from csfd.settings import Phase2Config, TicketsPerProblem, TicketTypeWeights


def _phase2_cfg() -> Phase2Config:
    return Phase2Config(
        tickets_per_problem=TicketsPerProblem(has_kb=(2, 2), no_kb=(1, 1)),
        ticket_type_weights=TicketTypeWeights(
            has_kb={"docs_request": 0.0, "l1": 1.0, "l2": 0.0, "l3": 0.0},
            no_kb={"l3": 1.0},
        ),
        creative_noise_probability=0.0,
        noise_type_weights={"x": 1.0},
        min_turns_per_ticket=2,
        max_turns_per_ticket=4,
    )


def _state(problems: list[CommittedProblem]) -> TicketState:
    return TicketState(
        run_id=str(uuid4()),
        parent_run_id=str(uuid4()),
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        problem_pool=problems,
    )


@pytest.mark.asyncio
async def test_ticket_sampler_picks_l3_when_no_kb() -> None:
    p_no = CommittedProblem(
        id="p-no",
        draft=ProblemDraft(title="t", description="d", category="c", severity="low"),
        has_kb=False,
        coverage_reasoning="",
        coverage_confidence="low",
    )
    state = _state([p_no])
    update = await ticket_sampler_node(state, cfg=_phase2_cfg())
    assert "current_problem_id" in update
    assert update["current_problem_id"] == "p-no"
    assert update["ticket_type"] == "l3"


@pytest.mark.asyncio
async def test_ticket_sampler_picks_l1_when_has_kb_under_l1_weights() -> None:
    p_yes = CommittedProblem(
        id="p-yes",
        draft=ProblemDraft(title="t", description="d", category="c", severity="low"),
        has_kb=True,
        coverage_reasoning="r",
        coverage_confidence="high",
    )
    state = _state([p_yes])
    update = await ticket_sampler_node(state, cfg=_phase2_cfg())
    assert update["ticket_type"] == "l1"


@pytest.mark.asyncio
async def test_ticket_init_node_builds_ticket_draft_with_personas() -> None:
    p_yes = CommittedProblem(
        id="p-yes",
        draft=ProblemDraft(
            title="Cannot log in", description="d", category="auth", severity="medium"
        ),
        has_kb=True,
        coverage_reasoning="r",
        coverage_confidence="high",
    )
    state = _state([p_yes])
    update = await ticket_init_node(
        state,
        problem=p_yes,
        ticket_type="l1",
        kb_article_id="a-1",
    )
    assert "current_ticket" in update
    ticket = update["current_ticket"]
    assert ticket.draft.problem_id == "p-yes"
    assert ticket.draft.ticket_type == "l1"
    assert ticket.draft.kb_article_id == "a-1"
    assert ticket.draft.customer_persona.tier in {"standard", "premium", "enterprise"}
    assert ticket.draft.agent_persona.tier == "l1"
    assert ticket.draft.subject
