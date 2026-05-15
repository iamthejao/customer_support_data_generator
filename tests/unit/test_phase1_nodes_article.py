from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import article_writer_node
from csfd.phases.phase1_kb.state import (
    CommittedProblem,
    KBArticleDraft,
    KBState,
    ProblemDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.settings import AgentLLMConfig


@pytest.mark.asyncio
async def test_article_writer_produces_draft_for_kb_problem() -> None:
    pc = CommittedProblem(
        id="abc",
        draft=ProblemDraft(
            title="Login fails",
            description="401 error",
            category="auth",
            severity="medium",
        ),
        has_kb=True,
        coverage_reasoning="ok",
        coverage_confidence="high",
    )
    state = KBState(
        run_id=str(uuid4()),
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        scenarios=ScenarioCatalogue(),
        problems_committed=[pc],
        current_problem_draft=pc.draft,
    )
    canned_article = KBArticleDraft(
        title="How to recover login",
        content_markdown="## Step 1\n...",
        troubleshooting_steps=[{"step": "do X", "expected_result": "Y"}],
    )
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    factory = AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={KBArticleDraft: canned_article}),
        agent_configs={
            "article_writer": AgentLLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5",
                temperature=0.3,
            ),
        },
    )
    update = await article_writer_node(state, factory=factory, problem=pc)
    assert "current_article_draft" in update
    draft = update["current_article_draft"]
    assert isinstance(draft, KBArticleDraft)
    assert draft.title == "How to recover login"
