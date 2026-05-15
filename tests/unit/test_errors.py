from pydantic import BaseModel
from pydantic import ValidationError as PydValidationError

from csfd.errors import (
    BudgetExceededError,
    ConvergenceError,
    PipelineError,
    SchemaValidationError,
    TransportError,
)


class _DummyModel(BaseModel):
    n: int


def test_hierarchy_roots_at_pipeline_error() -> None:
    for cls in (TransportError, SchemaValidationError, BudgetExceededError, ConvergenceError):
        assert issubclass(cls, PipelineError)


def test_schema_validation_error_carries_pydantic_errors_and_raw() -> None:
    try:
        _DummyModel.model_validate({"n": "not-an-int"})
    except PydValidationError as e:
        # Cast to list[dict[str, Any]]; e.errors() returns list[ErrorDetails] which is compatible
        err = SchemaValidationError(
            errors=[dict(error_dict) for error_dict in e.errors()],
            raw_output='{"n":"not-an-int"}'
        )
    assert err.raw_output == '{"n":"not-an-int"}'
    assert len(err.errors) == 1
    assert "n" in err.errors[0]["loc"]


def test_budget_error_carries_stats() -> None:
    err = BudgetExceededError(stats={"tokens": 6_000_000, "usd": 12.5})
    assert err.stats["tokens"] == 6_000_000
    assert "tokens" in str(err)
