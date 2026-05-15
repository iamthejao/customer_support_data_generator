from pathlib import Path

from pydantic import BaseModel

from csfd.agents.factory import AgentFactory
from csfd.agents.generator import Generator
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle, PromptRegistry
from csfd.settings import AgentLLMConfig


class _StubArtifact(BaseModel):
    summary: str = ""


def _registry(tmp_path: Path) -> PromptRegistry:
    (tmp_path / "phase1").mkdir()
    (tmp_path / "phase1" / "problem_generator.md.j2").write_text("Hi {{ inputs.x }}")
    reg = PromptRegistry(root=tmp_path)
    reg.load()
    return reg


def test_factory_builds_generator_with_bound_llm_and_prompt(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.5)
    factory = AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(canned={"default": "stub"}),
        agent_configs={"problem_brainstorm": cfg},
    )
    gen = factory.build_generator(
        name="problem_brainstorm",
        prompt_name="phase1.problem_generator",
        output_schema_factory=lambda: _StubArtifact,
    )
    assert isinstance(gen, Generator)
    assert gen.name == "problem_brainstorm"
    assert isinstance(gen.prompt, PromptHandle)
    assert gen.prompt.name == "phase1.problem_generator"
