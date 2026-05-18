"""csfd — Customer Service Fake Data."""

from csfd.allocator import (
    AllocationPlan,
    ProblemRef,
    TicketSlot,
    build_allocation_plan,
)
from csfd.pipeline import run_pipeline
from csfd.settings import AppSettings, load_settings

__version__ = "0.2.0"

__all__ = [
    "AllocationPlan",
    "AppSettings",
    "ProblemRef",
    "TicketSlot",
    "__version__",
    "build_allocation_plan",
    "load_settings",
    "run_pipeline",
]
