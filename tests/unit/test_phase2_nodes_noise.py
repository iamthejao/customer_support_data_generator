from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.creative_noise import TurnModification
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase2_cases.nodes import (
    creative_noise_gate,
    creative_noise_node,
)
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


def _state(seed: int, *, with_turn: bool = True) -> TicketState:
    draft = TurnDraft(speaker="customer", content="help me", intent="question")
    ticket = CommittedTicket(
        id="t-1",
        draft=TicketDraft(
            problem_id="p-1",
            kb_article_id=None,
            ticket_type="l3",
            priority="medium",
            subject="x",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="n"),
            agent_persona=AgentPersona(name="A", tier="l3", expertise=""),
        ),
    )
    return TicketState(
        run_id=str(uuid4()),
        parent_run_id=str(uuid4()),
        run_seed=seed,
        company=CompanyProfile(name="Acme", raw_markdown="# A"),
        current_ticket=ticket,
        current_turn_draft=draft if with_turn else None,
        turn_index=0,
    )


def test_creative_noise_gate_returns_noise_when_under_probability() -> None:
    state = _state(seed=1)
    decision = creative_noise_gate(state, probability=1.0)
    assert decision == "apply_noise"


def test_creative_noise_gate_returns_skip_when_zero_probability() -> None:
    state = _state(seed=1)
    decision = creative_noise_gate(state, probability=0.0)
    assert decision == "skip_noise"


def test_creative_noise_gate_is_deterministic_under_same_seed() -> None:
    s1 = _state(seed=42)
    s2 = _state(seed=42)
    assert creative_noise_gate(s1, probability=0.5) == creative_noise_gate(s2, probability=0.5)


@pytest.mark.asyncio
async def test_creative_noise_node_applies_modification() -> None:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    canned = TurnModification(
        modified_content="help me!!!1 i waited so long :(",
        noise_type="typos_informal_phrasing",
        rationale="customer frustration",
    )
    factory = AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={TurnModification: canned}),
        agent_configs={
            "creative_noise": AgentLLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5",
                temperature=0.95,
            ),
        },
    )
    state = _state(seed=1)
    update = await creative_noise_node(
        state,
        factory=factory,
        noise_type_weights={"typos_informal_phrasing": 1.0},
    )
    assert update["noise_applied"] is True
    assert update["noise_type"] == "typos_informal_phrasing"
    assert update["current_turn_draft"].content == canned.modified_content
    assert update["current_turn_draft"].noise_applied is True
    assert update["current_turn_draft"].noise_type == "typos_informal_phrasing"
