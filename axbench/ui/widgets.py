from __future__ import annotations

from datetime import datetime

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from axbench.ui.state import RunEvent, RunUIState, TaskLifecycle
from axbench.ui.theme import DEFAULT_THEME, UITheme

def render_metric_cards(state: RunUIState, theme: UITheme = DEFAULT_THEME) -> Columns:
    cards = [
        _metric_card("Complete", f"{state.finished_tasks}/{state.total_tasks}", theme.accent),
        _metric_card("Running", str(state.running_tasks), theme.accent_alt),
        _metric_card("Queued", str(state.queued_tasks), theme.muted),
        _metric_card("Failed", str(state.failed_tasks), theme.danger),
        _metric_card("Skipped", str(state.skipped_tasks), theme.warning),
        _metric_card("Success", f"{state.success_rate * 100:.1f}%", theme.success),
    ]
    return Columns(cards, equal=True, expand=True)


def render_progress_panel(state: RunUIState, theme: UITheme = DEFAULT_THEME) -> Panel:
    bar = _segmented_progress_bar(
        total=max(state.total_tasks, 1),
        completed=state.completed_tasks,
        failed=state.failed_tasks,
        running=state.running_tasks,
        queued=state.queued_tasks,
        width=44,
        theme=theme,
    )
    table = Table.grid(padding=(0, 1))
    table.add_row(
        Text("Progress", style="bold"),
        Text(f"{state.progress_fraction * 100:5.1f}%", style=f"bold {theme.accent}"),
        Text(f"{state.finished_tasks}/{state.total_tasks} completed", style=theme.muted),
    )
    table.add_row(bar)
    table.add_row(
        Text("States", style="bold"),
        Text(f"queued {state.queued_tasks}", style=theme.muted),
        Text(f"running {state.running_tasks}", style=theme.accent_alt),
        Text(f"passed {state.completed_tasks}", style=theme.success),
        Text(f"failed {state.failed_tasks}", style=theme.danger),
    )
    current = ", ".join(state.active_task_ids) if state.active_task_ids else "idle"
    recent = ", ".join(state.recent_completion_ids) if state.recent_completion_ids else "none yet"
    table.add_row(
        Text("Active", style="bold"),
        Text(current, style=f"bold {theme.accent_alt}"),
        Text(f"phase: {state.current_phase}", style=theme.muted),
    )
    table.add_row(
        Text("Recent", style="bold"),
        Text(recent, style=theme.success),
        Text(f"remaining: {state.remaining_tasks}", style=theme.muted),
    )
    return Panel(table, title="Progress / Status", border_style=theme.border)


def render_telemetry_panel(state: RunUIState, theme: UITheme = DEFAULT_THEME) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_row("Status", Text(state.overall_status.upper(), style=f"bold {theme.accent_alt}"))
    table.add_row("Elapsed", _format_seconds(state.elapsed_seconds))
    table.add_row("ETA", _format_seconds(state.estimated_remaining_seconds))
    table.add_row("Avg / task", f"{state.average_task_duration_ms:.0f} ms")
    table.add_row("Rolling TPS", f"{state.rolling_throughput_tasks_per_sec:.2f} tasks/s")
    table.add_row("Sustained TPS", f"{state.throughput_tasks_per_sec:.2f} tasks/s")
    table.add_row("Tasks / min", f"{state.completed_tasks_per_minute:.1f}")
    table.add_row("Started", _format_timestamp(state.benchmark_started_at))
    if state.slowest_tasks:
        task_id, duration_ms = state.slowest_tasks[0]
        table.add_row("Slowest", f"{task_id} ({duration_ms:.0f} ms)")
    performance_metrics = state.performance_metrics
    if performance_metrics:
        table.add_row("Perf TTFT", f"{float(performance_metrics.get('ttft_ms') or 0.0):.0f} ms")
        table.add_row("Perf Gen", f"{float(performance_metrics.get('tg_tokens_per_sec') or 0.0):.1f} tok/s")
    return Panel(table, title="Throughput & Timing", border_style=theme.border)


def render_task_panel(state: RunUIState, theme: UITheme = DEFAULT_THEME, max_rows: int = 10) -> Panel:
    table = Table(show_header=True, expand=True, box=None, pad_edge=False)
    table.add_column("Task", overflow="fold")
    table.add_column("State", no_wrap=True)
    table.add_column("When", no_wrap=True)
    table.add_column("Duration", no_wrap=True, justify="right")

    display_ids = _task_subset(state, max_rows=max_rows)
    for task_id in display_ids:
        status = state.task_statuses.get(task_id, TaskLifecycle.QUEUED)
        style = theme.status_style(status.value)
        glyph = theme.glyphs[status.value]
        label = Text(f"{glyph} {task_id}", style=style)
        if status is TaskLifecycle.RUNNING:
            label.stylize(f"bold {theme.accent_alt}")
            label.append("  <active>", style=f"bold {theme.accent_alt}")
        duration_ms = state.completed_task_durations_ms.get(task_id)
        duration = f"{duration_ms:.0f} ms" if duration_ms is not None else "..."
        if status is TaskLifecycle.SKIPPED:
            duration = "-"
        when = _task_time_label(state, task_id, status)
        table.add_row(
            label,
            Text(status.value.upper(), style=style),
            Text(when, style=theme.muted if when == "--" else style),
            Text(duration, style=theme.muted if duration == "..." else style),
        )
    hidden = max(len(state.task_order) - len(display_ids), 0)
    if hidden:
        table.add_row(
            Text(f"... {hidden} more tasks hidden", style=theme.muted),
            "",
            "",
            "",
        )
    return Panel(table, title="Task Queue / Live Task List", border_style=theme.border)


def render_events_panel(state: RunUIState, theme: UITheme = DEFAULT_THEME, max_rows: int = 8) -> Panel:
    table = Table(show_header=False, expand=True, box=None, pad_edge=False)
    table.add_column("Time", no_wrap=True, style=theme.muted)
    table.add_column("Type", no_wrap=True)
    table.add_column("Event", overflow="fold")

    recent_events = list(reversed(state.recent_events[-max_rows:]))
    if not recent_events:
        table.add_row("--:--:--", "--", Text("Waiting for benchmark events", style=theme.muted))
    for event in recent_events:
        table.add_row(
            _format_event_time(event),
            _event_badge(event, theme),
            _style_event_message(event, theme),
        )
    return Panel(table, title="Recent Events", border_style=theme.border)


def render_completion_summary(state: RunUIState, theme: UITheme = DEFAULT_THEME) -> Panel:
    title = "RUN COMPLETE" if state.failed_tasks == 0 else "RUN COMPLETE WITH FAILURES"
    color = theme.success if state.failed_tasks == 0 else theme.warning

    table = Table.grid(padding=(0, 2))
    table.add_row(
        Text(title, style=f"bold {color}"),
        Text(f"elapsed { _format_seconds(state.elapsed_seconds) }", style=theme.muted),
    )
    table.add_row(
        Text(f"success rate {state.success_rate * 100:.1f}%", style=f"bold {theme.accent}"),
        Text(f"completed {state.completed_tasks}", style=theme.success),
    )
    table.add_row(
        Text(f"failed {state.failed_tasks}", style=theme.danger),
        Text(f"skipped {state.skipped_tasks}", style=theme.warning),
    )
    table.add_row(
        Text(f"avg task {state.average_task_duration_ms:.0f} ms", style=theme.accent_alt),
        Text(f"throughput {state.throughput_tasks_per_sec:.2f} tasks/s", style=theme.accent_alt),
    )
    if state.performance_metrics:
        table.add_row(
            Text(
                f"perf TTFT {float(state.performance_metrics.get('ttft_ms') or 0.0):.0f} ms",
                style=theme.accent_alt,
            ),
            Text(
                f"perf gen {float(state.performance_metrics.get('tg_tokens_per_sec') or 0.0):.1f} tok/s",
                style=theme.accent_alt,
            ),
        )
    return Panel(table, title="Final Summary", border_style=color, padding=(1, 2))


def render_slowest_tasks_panel(state: RunUIState, theme: UITheme = DEFAULT_THEME) -> Panel:
    table = Table(show_header=True, expand=True, box=None, pad_edge=False)
    table.add_column("Task")
    table.add_column("Duration", justify="right", no_wrap=True)
    if not state.slowest_tasks:
        table.add_row(Text("No completed tasks yet", style=theme.muted), "")
    for task_id, duration_ms in state.slowest_tasks:
        table.add_row(Text(task_id, style=theme.warning), Text(f"{duration_ms:.0f} ms", style=theme.warning))
    return Panel(table, title="Top Slowest Tasks", border_style=theme.border)


def render_completion_banner(state: RunUIState, theme: UITheme = DEFAULT_THEME) -> Panel:
    headline = "AXBench run finished cleanly" if state.failed_tasks == 0 else "AXBench run finished with issues"
    subline = f"model {state.model}  |  endpoint {state.base_url}  |  run {state.run_id or '-'}"
    text = Text(justify="center")
    text.append(headline + "\n", style=f"bold {theme.success if state.failed_tasks == 0 else theme.warning}")
    text.append(subline, style=theme.muted)
    return Panel(text, border_style=theme.border, padding=(1, 2))


def render_footer(theme: UITheme = DEFAULT_THEME) -> Panel:
    footer = Text.assemble(
        ("Controls  ", f"bold {theme.muted}"),
        ("q", f"bold {theme.accent_alt}"),
        (" quit  ", theme.muted),
        ("no-ui", f"bold {theme.accent_alt}"),
        (" plain mode  ", theme.muted),
        ("AXBENCH_ASCII", f"bold {theme.accent_alt}"),
        (" ascii fallback  ", theme.muted),
        ("live dashboard", f"bold {theme.accent}"),
    )
    return Panel(footer, border_style=theme.border)


def _metric_card(title: str, value: str, color: str) -> Panel:
    text = Text(justify="center")
    text.append(f"{value}\n", style=f"bold {color}")
    text.append(title, style="dim")
    return Panel(text, border_style=color, padding=(0, 1))


def _segmented_progress_bar(
    total: int,
    completed: int,
    failed: int,
    running: int,
    queued: int,
    width: int,
    theme: UITheme,
) -> Text:
    allocations = [
        (completed, theme.success, "█"),
        (failed, theme.danger, "█"),
        (running, theme.accent_alt, "▓"),
        (queued, theme.muted, "░"),
    ]
    text = Text()
    used = 0
    for count, style, glyph in allocations:
        if count <= 0:
            continue
        remaining = max(width - used, 0)
        if remaining == 0:
            break
        blocks = min(max(round((count / total) * width), 1), remaining)
        used += blocks
        text.append(glyph * blocks, style=style)
    if used < width:
        text.append("░" * (width - used), style=theme.muted)
    return text


def _task_subset(state: RunUIState, max_rows: int) -> list[str]:
    prioritized = state.active_task_ids + list(reversed(state.recently_finished_task_ids))
    seen: set[str] = set()
    ordered: list[str] = []
    for task_id in prioritized + state.task_order:
        if task_id in seen:
            continue
        seen.add(task_id)
        ordered.append(task_id)
    return ordered[:max_rows]


def _format_seconds(value: float) -> str:
    seconds = max(int(value), 0)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _format_timestamp(iso_value: str) -> str:
    return iso_value.replace("T", " ")[:19]


def _format_event_time(event: RunEvent) -> str:
    try:
        return datetime.fromisoformat(event.timestamp).strftime("%H:%M:%S")
    except ValueError:
        return event.timestamp[:8]


def _style_event_message(event: RunEvent, theme: UITheme) -> Text:
    if event.kind in {"task_failed", "performance_failed"}:
        return Text(event.message, style=theme.danger)
    if event.kind in {"task_completed", "performance_completed"}:
        return Text(event.message, style=theme.success)
    if event.kind == "perf_output":
        return Text(event.message, style=theme.accent_alt)
    if event.kind == "task_started":
        return Text(event.message, style=theme.accent_alt)
    if event.kind == "phase_changed":
        return Text(event.message, style=theme.accent)
    return Text(event.message, style="default")


def _event_badge(event: RunEvent, theme: UITheme) -> Text:
    badge_map = {
        "task_started": ("START", theme.accent_alt),
        "task_completed": ("DONE", theme.success),
        "task_failed": ("FAIL", theme.danger),
        "task_skipped": ("SKIP", theme.warning),
        "phase_changed": ("PHASE", theme.accent),
        "perf_output": ("PERF", theme.accent_alt),
        "performance_completed": ("PERF", theme.success),
        "performance_failed": ("PERF", theme.danger),
        "run_started": ("BOOT", theme.accent),
    }
    label, color = badge_map.get(event.kind, ("INFO", theme.muted))
    return Text(label, style=f"bold {color}")


def _task_time_label(state: RunUIState, task_id: str, status: TaskLifecycle) -> str:
    if status is TaskLifecycle.RUNNING:
        started = state.task_started_at_iso.get(task_id)
        if started:
            return _format_timestamp(started)[11:19]
        return "now"
    if status in {TaskLifecycle.COMPLETED, TaskLifecycle.FAILED, TaskLifecycle.SKIPPED}:
        finished = state.task_finished_at_iso.get(task_id)
        if finished:
            return _format_timestamp(finished)[11:19]
    return "--"
