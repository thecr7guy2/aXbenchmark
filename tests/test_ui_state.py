from axbench.ui.state import RunUIState


def test_run_ui_state_tracks_counts_and_events():
    state = RunUIState.create(
        model="test-model",
        base_url="http://localhost:8000/v1",
        total_tasks=2,
        run_id="run-123",
    )
    state.initialize_tasks(["task_a", "task_b"], ["task_c"])

    assert state.queued_tasks == 2
    assert state.current_phase == "startup"

    state.set_phase("benchmarking", "Benchmarking started")
    state.mark_task_started("task_a")
    state.mark_task_completed("task_a", duration_ms=125.0)
    state.mark_task_started("task_b")
    state.mark_task_failed("task_b", duration_ms=250.0)
    state.mark_task_skipped("task_c")

    assert state.completed_tasks == 1
    assert state.failed_tasks == 1
    assert state.skipped_tasks == 1
    assert state.running_tasks == 0
    assert state.queued_tasks == 0
    assert state.progress_fraction == 1.0
    assert state.recently_finished_task_ids == ["task_a", "task_b", "task_c"]
    assert "task_a" in state.completed_task_durations_ms
    assert state.task_statuses["task_a"].value == "completed"
    assert state.task_statuses["task_c"].value == "skipped"
    assert len(state.recent_events) >= 4


def test_run_ui_state_exposes_telemetry_and_perf_metrics():
    state = RunUIState.create(
        model="test-model",
        base_url="http://localhost:8000/v1",
        total_tasks=2,
        run_id="run-456",
    )
    state.benchmark_started_monotonic -= 10
    state.initialize_tasks(["task_a", "task_b"])
    state.mark_task_started("task_a")
    state.mark_task_completed("task_a", duration_ms=200.0)
    state.mark_task_started("task_b")
    state.mark_task_failed("task_b", duration_ms=400.0, error="timeout")
    state.record_performance_metrics(
        "performance_llama_benchy",
        {
            "ttft_ms": 320.0,
            "tg_tokens_per_sec": 75.0,
        },
    )

    assert state.elapsed_seconds >= 10
    assert state.average_task_duration_ms == 300.0
    assert state.completed_tasks_per_minute > 0
    assert state.rolling_throughput_tasks_per_sec > 0
    assert state.performance_metrics["ttft_ms"] == 320.0
    assert state.task_error_messages["task_b"] == "timeout"
