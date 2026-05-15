import jinja2
import pytest
from pydantic import BaseModel

from csfd.agents.base import AgentContext, Issue, Verdict
from csfd.agents.checker import Checker
from csfd.agents.generator import Generator
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle


class _Problem(BaseModel):
    title: str
    description: str


def _h(src: str) -> PromptHandle:
    env = jinja2.Environment(autoescape=False)
    return PromptHandle(name="x", template=env.from_string(src), source=src, version="hhhhhhhhhhhh")


@pytest.mark.asyncio
async def test_full_feedback_loop_terminates_after_retry() -> None:
    """Generator → 3 parallel Checkers → aggregate → if any fail, retry Generator with issues."""
    bad = _Problem(title="x", description="Internal jira id INT-1234")
    good = _Problem(title="x", description="Customer-visible description")

    llm_gen = FakeChatModel(structured={_Problem: bad})
    gen = Generator(
        name="problem_brainstorm",
        prompt=_h(
            "Generate {{ inputs.x }}."
            "{% for v in prior_verdicts %}{% for i in v.issues %}"
            "| FIX: {{ i.explanation }}"
            "{% endfor %}{% endfor %}"
        ),
        output_schema=_Problem,
        llm=llm_gen,
    )

    consistency_fail = Verdict(
        checker="consistency",
        passed=False,
        issues=[
            Issue(
                severity="error",
                location="description",
                rule_violated="no_internal_ids",
                explanation="Mentions internal jira id INT-1234",
            )
        ],
    )
    consistency_pass = Verdict(checker="consistency", passed=True)
    bg_pass = Verdict(checker="background", passed=True)
    sc_pass = Verdict(checker="scenario", passed=True)

    llm_cons = FakeChatModel(structured={Verdict: consistency_fail})
    llm_bg = FakeChatModel(structured={Verdict: bg_pass})
    llm_sc = FakeChatModel(structured={Verdict: sc_pass})

    cons = Checker(name="consistency", prompt=_h("check {{ inputs.artifact }}"), llm=llm_cons)
    bg = Checker(name="background", prompt=_h("check"), llm=llm_bg)
    sc = Checker(name="scenario", prompt=_h("check"), llm=llm_sc)

    # First attempt
    ctx = AgentContext(inputs={"x": "billing problem"})
    attempt1: BaseModel = await gen.invoke(ctx)
    v_cons = await cons.invoke(AgentContext(inputs={"artifact": attempt1.model_dump()}))
    v_bg = await bg.invoke(AgentContext(inputs={"artifact": attempt1.model_dump()}))
    v_sc = await sc.invoke(AgentContext(inputs={"artifact": attempt1.model_dump()}))
    verdicts = [v_cons, v_bg, v_sc]
    passed = all(v.passed for v in verdicts)
    assert passed is False
    assert any("INT-1234" in i.explanation for v in verdicts for i in v.issues)

    # Retry: flip the generator to good, flip consistency to pass
    llm_gen.structured = {_Problem: good}
    llm_cons.structured = {Verdict: consistency_pass}
    retry_ctx = AgentContext(
        inputs={"x": "billing problem"},
        prior_verdicts=verdicts,
        retry_attempt=1,
    )
    attempt2: BaseModel = await gen.invoke(retry_ctx)
    # The retry prompt MUST have included the fix instruction
    assert "INT-1234" in (gen.last_rendered_prompt or "")
    v_cons2 = await cons.invoke(AgentContext(inputs={"artifact": attempt2.model_dump()}))
    assert v_cons2.passed is True
