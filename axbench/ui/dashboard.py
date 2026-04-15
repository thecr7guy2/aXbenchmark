from __future__ import annotations

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from axbench.ui.state import RunUIState
from axbench.ui.theme import DEFAULT_THEME, UITheme
from axbench.ui.widgets import (
    render_completion_banner,
    render_completion_summary,
    render_events_panel,
    render_footer,
    render_metric_cards,
    render_progress_panel,
    render_slowest_tasks_panel,
    render_task_panel,
    render_telemetry_panel,
)


class LiveDashboard:
    def __init__(
        self,
        console,
        state: RunUIState,
        selected_task_ids: list[str],
        skipped_task_ids: list[str],
        theme: UITheme = DEFAULT_THEME,
    ):
        self.console = console
        self.state = state
        self.selected_task_ids = selected_task_ids
        self.skipped_task_ids = skipped_task_ids
        self.theme = theme
        self._live: Live | None = None

    def start(self) -> None:
        self.state.initialize_tasks(self.selected_task_ids, self.skipped_task_ids)
        self.state.overall_status = "running"
        self._live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=6,
            transient=False,
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        self.state.overall_status = "completed" if self.state.failed_tasks == 0 else "degraded"
        self.state.current_phase = "completed"
        self.refresh()
        if self._live is not None:
            self._live.stop()
            self._live = None

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.render(), refresh=True)

    def render(self):
        if self.state.current_phase == "completed":
            return self._render_completion()

        layout = Layout()
        layout.split_column(
            Layout(self._render_header(), name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(render_footer(self.theme), name="footer", size=3),
        )
        layout["body"].split_column(
            Layout(render_metric_cards(self.state, self.theme), name="summary", size=5),
            Layout(name="main", ratio=1),
        )
        layout["body"]["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        layout["body"]["main"]["left"].split_column(
            Layout(render_progress_panel(self.state, self.theme), name="progress", size=7),
            Layout(render_task_panel(self.state, self.theme), name="tasks", ratio=1),
        )
        layout["body"]["main"]["right"].split_column(
            Layout(render_telemetry_panel(self.state, self.theme), name="telemetry", size=9),
            Layout(render_events_panel(self.state, self.theme), name="events", ratio=1),
        )
        return layout

    def _render_completion(self):
        layout = Layout()
        layout.split_column(
            Layout(self._render_header(), name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(render_footer(self.theme), name="footer", size=3),
        )
        layout["body"].split_column(
            Layout(render_completion_banner(self.state, self.theme), name="banner", size=5),
            Layout(name="middle", ratio=1),
        )
        layout["body"]["middle"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        layout["body"]["middle"]["left"].split_column(
            Layout(render_completion_summary(self.state, self.theme), name="summary", size=10),
            Layout(render_slowest_tasks_panel(self.state, self.theme), name="slowest", size=8),
        )
        layout["body"]["middle"]["right"].split_column(
            Layout(render_telemetry_panel(self.state, self.theme), name="telemetry", size=12),
            Layout(render_events_panel(self.state, self.theme), name="events", ratio=1),
        )
        return layout

    def _render_header(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(ratio=2)
        table.add_column(ratio=2, justify="center")
        table.add_column(ratio=3, justify="right")
        table.add_row(
            Text(self.theme.title_text, style=f"bold {self.theme.accent}"),
            Text(f"phase: {self.state.current_phase}", style=f"bold {self.theme.accent_alt}"),
            Text(
                f"model {self.state.model}  |  endpoint {self.state.base_url}  |  run {self.state.run_id or '-'}",
                style=self.theme.muted,
            ),
        )
        return Panel(table, border_style=self.theme.border)
