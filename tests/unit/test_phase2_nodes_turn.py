from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBArticleDraft,
    ProblemDraft,
)
from csfd.phases.phase2_cases.nodes import turn_writer_node
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CustomerPersona,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.settings import AgentLLMConfig


def _factory(canned: TurnDraft) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={TurnDraft: canned}),
        agent_configs={
            "turn_writer": AgentLLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5",
                temperature=0.7,
            ),
        },
    )


def _state(with_kb: bool) -> TicketState:
    pid = "p-x"
    problem = CommittedProblem(
        id=pid,
        draft=ProblemDraft(
            title="Login fails", description="401 returned", category="auth", severity="medium"
        ),
        has_kb=with_kb,
        coverage_reasoning="r",
        coverage_confidence="high",
    )
    kbs: dict[str, CommittedArticle] = {}
    if with_kb:
        kbs[pid] = CommittedArticle(
            id="a-1",
            problem_id=pid,
            draft=KBArticleDraft(
                title="Recover login",
                content_markdown="## Step 1",
                troubleshooting_steps=[{"step": "x", "expected_result": "y"}],
            ),
        )
    ticket = CommittedTicket(
        id="t-1",
        draft=TicketDraft(
            problem_id=pid,
            kb_article_id="a-1" if with_kb else None,
            ticket_type="l3" if not with_kb else "l1",
            priority="medium",
            subject="Login fails",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="neutral"),
            agent_persona=AgentPersona(name="A", tier="l3" if not with_kb else "l1", expertise=""),
        ),
    )
    return TicketState(
        run_id=str(uuid4()),
        parent_run_id=str(uuid4()),
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        problem_pool=[problem],
        kb_by_problem=kbs,
        current_ticket=ticket,
        turn_index=0,
    )


@pytest.mark.asyncio
async def test_turn_writer_emits_draft_into_state_with_kb() -> None:
    canned = TurnDraft(
        speaker="agent",
        content="Hi, please reset via email link.",
        intent="resolution",
    )
    factory = _factory(canned)
    state = _state(with_kb=True)
    update = await turn_writer_node(state, factory=factory)
    assert "current_turn_draft" in update
    draft = update["current_turn_draft"]
    assert isinstance(draft, TurnDraft)
    assert draft.speaker == "agent"


@pytest.mark.asyncio
async def test_turn_writer_runs_for_l3_no_kb_path() -> None:
    canned = TurnDraft(
        speaker="agent",
        content="Let me diagnose. Can you share recent logs?",
        intent="clarification",
    )
    factory = _factory(canned)
    state = _state(with_kb=False)
    update = await turn_writer_node(state, factory=factory)
    assert update["current_turn_draft"].intent == "clarification"
