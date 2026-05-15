"""Imperative orchestrators that loop the runtime graphs over problems / tickets.

The runtime graphs in ``runtime_phase1`` and ``runtime_phase2`` each process a
*single* unit (one problem / one turn). The orchestrators here own the outer
loops:

* ``run_phase1`` creates the run row, then invokes the Phase 1 graph
  ``problem_count`` times.
* ``run_phase2`` loads the parent KB once, then for each problem samples a
  ticket count and for each ticket runs the per-turn graph until
  ``max_turns_per_ticket`` is reached.

Both functions update the run status to ``completed`` on a normal exit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from csfd.agents.factory import AgentFactory
from csfd.graph.runtime_phase1 import build_runtime_phase1_graph
from csfd.graph.runtime_phase2 import build_runtime_phase2_graph
from csfd.phases.phase1_kb.state import KBState
from csfd.phases.phase2_cases.nodes import (
    load_kb_node,
    ticket_init_node,
    ticket_sampler_node,
)
from csfd.phases.phase2_cases.sampler import sample_ticket_count
from csfd.phases.phase2_cases.state import TicketState
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.settings import Phase1Config, Phase2Config
from csfd.storage.db import Database
from csfd.storage.repository import RunRecord, RunRepo
from csfd.utils.rng import derive_rng


async def run_phase1(
    *,
    factory: AgentFactory,
    db: Database,
    phase1_cfg: Phase1Config,
    run_seed: int,
    max_retries: int,
    company: CompanyProfile,
    scenarios: ScenarioCatalogue,
    pipeline_version: str,
) -> str:
    """Orchestrate Phase 1: loop the runtime graph for ``problem_count`` iterations.

    Returns the newly-minted ``run_id`` so callers can chain a Phase 2 run.
    """
    run_id = str(uuid4())
    RunRepo(db).create(
        RunRecord(
            id=run_id,
            phase="phase1",
            parent_run_id=None,
            status="running",
            started_at=datetime.now(UTC),
            completed_at=None,
            run_seed=run_seed,
            pipeline_version=pipeline_version,
            git_sha=None,
            config_snapshot_json=phase1_cfg.model_dump_json(),
            stats_json=None,
            error_summary=None,
        )
    )
    graph = build_runtime_phase1_graph(
        factory=factory,
        db=db,
        max_retries=max_retries,
        kb_target_rate=phase1_cfg.kb_coverage_target_rate,
    )
    for _ in range(phase1_cfg.problem_count):
        state = KBState(
            run_id=run_id,
            run_seed=run_seed,
            company=company,
            scenarios=scenarios,
        )
        await graph.ainvoke(state)
    RunRepo(db).update_status(run_id, status="completed", completed=True)
    return run_id


async def run_phase2(
    *,
    factory: AgentFactory,
    db: Database,
    phase2_cfg: Phase2Config,
    run_seed: int,
    max_retries: int,
    parent_run_id: str,
    company: CompanyProfile,
    pipeline_version: str,
) -> str:
    """Orchestrate Phase 2.

    1. Create the Phase 2 run row.
    2. Load the parent Phase 1 KB (problems + articles) once.
    3. For each problem, sample a ticket count and for each ticket sample
       a ticket type, initialise the ticket draft, then loop the per-turn
       runtime graph until ``max_turns_per_ticket`` turns are committed.
    """
    run_id = str(uuid4())
    RunRepo(db).create(
        RunRecord(
            id=run_id,
            phase="phase2",
            parent_run_id=parent_run_id,
            status="running",
            started_at=datetime.now(UTC),
            completed_at=None,
            run_seed=run_seed,
            pipeline_version=pipeline_version,
            git_sha=None,
            config_snapshot_json=phase2_cfg.model_dump_json(),
            stats_json=None,
            error_summary=None,
        )
    )

    # Seed state used only to load the parent KB.
    seed_state = TicketState(
        run_id=run_id,
        parent_run_id=parent_run_id,
        run_seed=run_seed,
        company=company,
    )
    seed_update = await load_kb_node(seed_state, db=db)
    pool = seed_update["problem_pool"]
    kb_by_problem = seed_update["kb_by_problem"]

    graph = build_runtime_phase2_graph(
        factory=factory,
        db=db,
        max_retries=max_retries,
        noise_probability=phase2_cfg.creative_noise_probability,
        noise_type_weights=phase2_cfg.noise_type_weights,
    )

    for problem in pool:
        count_rng = derive_rng(run_seed, f"ticket_count:{problem.id}")
        range_inclusive = (
            phase2_cfg.tickets_per_problem.has_kb
            if problem.has_kb
            else phase2_cfg.tickets_per_problem.no_kb
        )
        n_tickets = sample_ticket_count(count_rng, range_inclusive=range_inclusive)
        for _ticket_i in range(n_tickets):
            base_state = TicketState(
                run_id=run_id,
                parent_run_id=parent_run_id,
                run_seed=run_seed,
                company=company,
                problem_pool=[problem],
                kb_by_problem=(
                    {problem.id: kb_by_problem[problem.id]} if problem.id in kb_by_problem else {}
                ),
            )
            sample = await ticket_sampler_node(base_state, cfg=phase2_cfg)
            tt = sample["ticket_type"]
            kb_id = kb_by_problem[problem.id].id if problem.id in kb_by_problem else None
            init_update = await ticket_init_node(
                base_state,
                problem=problem,
                ticket_type=tt,
                kb_article_id=kb_id,
            )
            state = base_state.model_copy(update=init_update)

            # Inner turn loop: each ``ainvoke`` produces one committed turn
            # (the runtime Phase 2 graph has its own ``ensure_ticket`` entry
            # node, so the ticket gets idempotently persisted on first turn).
            for _ in range(phase2_cfg.max_turns_per_ticket):
                result = await graph.ainvoke(state)
                # ``ainvoke`` returns a dict-shaped state — re-validate to
                # ``TicketState`` for the next iteration.
                state = TicketState.model_validate(result)
                if len(state.turns_committed) >= phase2_cfg.max_turns_per_ticket:
                    break
                # Reset per-turn fields before the next iteration so the
                # writer + checkers see a clean slate.
                state = state.model_copy(
                    update={
                        "current_turn_draft": None,
                        "verdicts": [],
                        "retry_attempt": 0,
                    }
                )

    RunRepo(db).update_status(run_id, status="completed", completed=True)
    return run_id
