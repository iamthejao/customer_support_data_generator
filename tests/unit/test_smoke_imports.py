def test_all_top_level_modules_import() -> None:
    import csfd
    import csfd.budget.breaker
    import csfd.budget.tracker
    import csfd.errors
    import csfd.models.fake
    import csfd.models.registry
    import csfd.observability.langsmith
    import csfd.observability.logging
    import csfd.prompts.registry
    import csfd.seeds.company
    import csfd.seeds.scenarios
    import csfd.settings
    import csfd.storage.db
    import csfd.storage.exporters
    import csfd.storage.migrations.runner
    import csfd.storage.repository
    import csfd.ticket_types.definitions
    import csfd.utils.hashing
    import csfd.utils.retry
    import csfd.utils.rng

    assert csfd is not None
