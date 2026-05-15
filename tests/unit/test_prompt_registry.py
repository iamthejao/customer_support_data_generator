from pathlib import Path

import pytest

from csfd.prompts.registry import PromptRegistry


def test_registry_discovers_and_hashes_templates() -> None:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    handle = reg.get("phase1._test")
    assert handle.name == "phase1._test"
    assert len(handle.version) == 12
    rendered = handle.template.render(name="world")
    assert "world" in rendered


def test_registry_hashes_are_stable_across_loads() -> None:
    a = PromptRegistry(root=Path("prompts"))
    a.load()
    b = PromptRegistry(root=Path("prompts"))
    b.load()
    assert a.get("phase1._test").version == b.get("phase1._test").version


def test_registry_unknown_name_raises() -> None:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    with pytest.raises(KeyError):
        reg.get("nonexistent.template")
