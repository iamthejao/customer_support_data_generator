from csfd.embeddings.text import text_for_problem
from csfd.pipeline import ProblemBrainstormOutput
from csfd.ticket_types.definitions import ProblemComplexity


def _fixture() -> ProblemBrainstormOutput:
    return ProblemBrainstormOutput(
        title="Compressor surge alarm",
        summary="CT-500 trips surge alarm on cold start.",
        background="Detailed background paragraph.",
        category="hardware",
        complexity=ProblemComplexity.MEDIUM,
        resolution_hints={"docs_request": "x", "l1": "x", "l2": "x", "l3": "x"},
    )


def test_title_summary_template_joins_with_blank_line() -> None:
    text = text_for_problem(_fixture(), "title_summary")
    assert text == "Compressor surge alarm\n\nCT-500 trips surge alarm on cold start."


def test_title_summary_background_template_adds_background() -> None:
    text = text_for_problem(_fixture(), "title_summary_background")
    assert text == (
        "Compressor surge alarm\n\n"
        "CT-500 trips surge alarm on cold start.\n\n"
        "Detailed background paragraph."
    )


def test_unknown_template_raises_value_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown template"):
        text_for_problem(_fixture(), "nope")  # type: ignore[arg-type]


def _fixture_with_lists() -> ProblemBrainstormOutput:
    return ProblemBrainstormOutput(
        title="Compressor surge alarm",
        summary="CT-500 trips surge alarm on cold start.",
        background="Detailed background paragraph.",
        symptoms=["alarm at startup", "compressor cycles"],
        root_cause=["faulty pressure sensor"],
        category="hardware",
        complexity=ProblemComplexity.MEDIUM,
        resolution_hints={"docs_request": "x", "l1": "x", "l2": "x", "l3": "x"},
    )


def test_symptoms_root_cause_template_includes_lists() -> None:
    text = text_for_problem(_fixture_with_lists(), "title_summary_symptoms_root_cause")
    assert text == (
        "Compressor surge alarm\n\n"
        "CT-500 trips surge alarm on cold start.\n\n"
        "Symptoms:\n- alarm at startup\n- compressor cycles\n\n"
        "Root cause:\n- faulty pressure sensor"
    )


def test_symptoms_root_cause_template_omits_empty_sections() -> None:
    # _fixture() has empty symptoms/root_cause -> degrades to title+summary only.
    text = text_for_problem(_fixture(), "title_summary_symptoms_root_cause")
    assert text == "Compressor surge alarm\n\nCT-500 trips surge alarm on cold start."
