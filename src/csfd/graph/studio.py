"""LangGraph Studio entrypoint — pre-built graph to avoid blockbuster errors.

``langgraph dev`` runs requests under the ``blockbuster`` library, which traps
synchronous I/O calls inside the asyncio event loop and raises
``BlockingError``. Our factory chain does three sync calls at startup:

1. ``load_settings()`` — YAML parse via ``open()``
2. ``PromptRegistry.load()`` — directory scan via ``rglob()``
3. ``Database(...)`` — ``mkdir(parents=True, exist_ok=True)``

If we let LangGraph invoke a factory per request, those calls execute inside
the event loop and trip blockbuster. Instead, we build the parent pipeline
graph **once at module import time** — before the ASGI loop starts — and
expose it as a module-level constant. ``langgraph.json`` references this
constant directly, so per-request handlers just return the cached compiled
graph with no I/O.

NOTE: graph *execution* still does sync sqlite writes inside node functions
(``commit_problem``, ``commit_resolution``, etc.). Running the graph from
Studio therefore still requires ``langgraph dev --allow-blocking`` until the
persistence layer is async-ified.
"""

from __future__ import annotations

from csfd.graph.pipeline_graph import make_pipeline_studio_graph

pipeline = make_pipeline_studio_graph()

__all__ = ["pipeline"]
