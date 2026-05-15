"""LangGraph Studio entrypoint — pre-built graphs to avoid blockbuster errors.

``langgraph dev`` runs requests under the ``blockbuster`` library, which traps
synchronous I/O calls inside the asyncio event loop and raises
``BlockingError``. Our factory chain does three sync calls at startup:

1. ``load_settings()`` — YAML parse via ``open()``
2. ``PromptRegistry.load()`` — directory scan via ``rglob()``
3. ``Database(...)`` — ``mkdir(parents=True, exist_ok=True)``

If we let LangGraph invoke a factory per request, those calls execute inside
the event loop and trip blockbuster. Instead, we build the graphs **once at
module import time** — before the ASGI loop starts — and expose them as
module-level constants. ``langgraph.json`` references these constants directly,
so per-request handlers just return the cached compiled graph with no I/O.

NOTE: graph *execution* still does sync sqlite writes inside node functions
(``persist_problem_node`` et al.). Running graphs from Studio therefore still
requires ``langgraph dev --allow-blocking`` until the persistence layer is
async-ified (v2 backlog).
"""

from __future__ import annotations

from csfd.graph.compose import make_phase1_studio_graph, make_phase2_studio_graph

phase1 = make_phase1_studio_graph()
phase2 = make_phase2_studio_graph()

__all__ = ["phase1", "phase2"]
