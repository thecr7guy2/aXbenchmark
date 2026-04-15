from rich.console import Console

from axbench.ui.dashboard import LiveDashboard
from axbench.ui.theme import resolve_theme
from axbench.ui.state import RunUIState


def test_dashboard_render_contains_panels_and_task_states():
    state = RunUIState.create(
        model="test-model",
        base_url="http://localhost:8000/v1",
        total_tasks=3,
        run_id="run-123",
    )
    dashboard = LiveDashboard(
        console=Console(record=True, width=140),
        state=state,
        selected_task_ids=["task_a", "task_b", "task_c"],
        skipped_task_ids=["task_d"],
    )
    state.initialize_tasks(["task_a", "task_b", "task_c"], ["task_d"])
    state.set_phase("benchmarking", "Benchmarking started")
    state.mark_task_started("task_a")
    state.mark_task_completed("task_a", duration_ms=150.0)
    state.mark_task_started("task_b")
    state.record_performance_metrics(
        "performance_llama_benchy",
        {
            "ttft_ms": 400.0,
            "tg_tokens_per_sec": 60.0,
        },
    )

    console = Console(record=True, width=140)
    console.print(dashboard.render())
    output = console.export_text()

    assert "AXBench Command Center" in output
    assert "Progress / Status" in output
    assert "Throughput & Timing" in output
    assert "Task Queue / Live Task List" in output
    assert "Recent Events" in output
    assert "task_a" in output
    assert "task_b" in output
    assert "Rolling TPS" in output
    assert "Tasks / min" in output
    assert "PERF" in output


def test_dashboard_render_completion_screen_and_ascii_theme():
    state = RunUIState.create(
        model="test-model",
        base_url="http://localhost:8000/v1",
        total_tasks=2,
        run_id="run-999",
    )
    state.initialize_tasks(["task_a", "task_b"])
    state.mark_task_started("task_a")
    state.mark_task_completed("task_a", duration_ms=200.0)
    state.mark_task_started("task_b")
    state.mark_task_failed("task_b", duration_ms=450.0, error="timeout")
    state.current_phase = "completed"
    state.overall_status = "degraded"

    dashboard = LiveDashboard(
        console=Console(record=True, width=120),
        state=state,
        selected_task_ids=["task_a", "task_b"],
        skipped_task_ids=[],
        theme=resolve_theme(unicode_ok=False, color_ok=False),
    )

    console = Console(record=True, width=120)
    console.print(dashboard.render())
    output = console.export_text()

    assert "AXBench Console" in output
    assert "Final Summary" in output
    assert "Top Slowest Tasks" in output
    assert "RUN COMPLETE WITH FAILURES" in output
