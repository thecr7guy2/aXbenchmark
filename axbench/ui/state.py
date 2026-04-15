from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class TaskLifecycle(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RunEvent:
    kind: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_id: str | None = None
    level: str = "info"


@dataclass
class RunUIState:
    model: str
    base_url: str
    total_tasks: int
    run_id: str | None = None
    benchmark_started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    benchmark_started_monotonic: float = field(default_factory=time.monotonic, repr=False)
    current_phase: str = "startup"
    overall_status: str = "pending"
    queued_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    active_task_ids: list[str] = field(default_factory=list)
    recently_finished_task_ids: list[str] = field(default_factory=list)
    recent_events: list[RunEvent] = field(default_factory=list)
    started_task_timestamps: dict[str, float] = field(default_factory=dict)
    task_started_at_iso: dict[str, str] = field(default_factory=dict)
    task_finished_at_iso: dict[str, str] = field(default_factory=dict)
    completed_task_durations_ms: dict[str, float] = field(default_factory=dict)
    task_error_messages: dict[str, str] = field(default_factory=dict)
    task_order: list[str] = field(default_factory=list)
    task_statuses: dict[str, TaskLifecycle] = field(default_factory=dict)
    completion_samples_monotonic: list[float] = field(default_factory=list, repr=False)
    performance_metrics: dict[str, float | str | None] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        model: str,
        base_url: str,
        total_tasks: int,
        run_id: str | None = None,
        queued_tasks: int | None = None,
    ) -> "RunUIState":
        return cls(
            model=model,
            base_url=base_url,
            total_tasks=total_tasks,
            run_id=run_id,
            queued_tasks=total_tasks if queued_tasks is None else queued_tasks,
        )

    @property
    def accounted_tasks(self) -> int:
        return (
            self.queued_tasks
            + self.running_tasks
            + self.completed_tasks
            + self.failed_tasks
            + self.skipped_tasks
        )

    @property
    def finished_tasks(self) -> int:
        return self.completed_tasks + self.failed_tasks

    @property
    def remaining_tasks(self) -> int:
        return max(self.total_tasks - self.finished_tasks - self.running_tasks, 0)

    @property
    def progress_fraction(self) -> float:
        if self.total_tasks <= 0:
            return 0.0
        return min(self.finished_tasks / self.total_tasks, 1.0)

    @property
    def elapsed_seconds(self) -> float:
        return max(time.monotonic() - self.benchmark_started_monotonic, 0.0)

    @property
    def average_task_duration_ms(self) -> float:
        if not self.completed_task_durations_ms:
            return 0.0
        return sum(self.completed_task_durations_ms.values()) / len(self.completed_task_durations_ms)

    @property
    def throughput_tasks_per_sec(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.finished_tasks / elapsed

    @property
    def rolling_throughput_tasks_per_sec(self) -> float:
        if not self.completion_samples_monotonic:
            return 0.0
        now = time.monotonic()
        window_seconds = min(max(self.elapsed_seconds, 1.0), 30.0)
        recent = [sample for sample in self.completion_samples_monotonic if now - sample <= window_seconds]
        if not recent:
            return 0.0
        return len(recent) / window_seconds

    @property
    def completed_tasks_per_minute(self) -> float:
        return self.throughput_tasks_per_sec * 60.0

    @property
    def success_rate(self) -> float:
        total_finished = self.finished_tasks
        if total_finished <= 0:
            return 0.0
        return self.completed_tasks / total_finished

    @property
    def estimated_remaining_seconds(self) -> float:
        average_ms = self.average_task_duration_ms
        if average_ms <= 0 or self.remaining_tasks <= 0:
            return 0.0
        return (average_ms / 1000.0) * self.remaining_tasks

    @property
    def recent_completion_ids(self) -> list[str]:
        return self.recently_finished_task_ids[-3:]

    @property
    def slowest_tasks(self) -> list[tuple[str, float]]:
        return sorted(
            self.completed_task_durations_ms.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]

    def initialize_tasks(self, selected_task_ids: list[str], skipped_task_ids: list[str] | None = None) -> None:
        skipped_task_ids = skipped_task_ids or []
        self.task_order = list(selected_task_ids) + [task_id for task_id in skipped_task_ids if task_id not in selected_task_ids]
        self.task_statuses = {
            task_id: TaskLifecycle.QUEUED for task_id in selected_task_ids
        }
        for task_id in skipped_task_ids:
            self.task_statuses[task_id] = TaskLifecycle.SKIPPED
        self.skipped_tasks = len(skipped_task_ids)
        self.queued_tasks = len(selected_task_ids)
        for task_id in skipped_task_ids:
            self.add_event("task_skipped", f"Task skipped by filter: {task_id}", task_id=task_id, level="warning")

    def set_phase(self, phase: str, message: str | None = None) -> None:
        self.current_phase = phase
        if message:
            self.add_event("phase_changed", message)

    def add_event(
        self,
        kind: str,
        message: str,
        task_id: str | None = None,
        level: str = "info",
    ) -> None:
        self.recent_events.append(RunEvent(kind=kind, message=message, task_id=task_id, level=level))
        self.recent_events = self.recent_events[-12:]

    def mark_task_started(self, task_id: str) -> None:
        if self.queued_tasks > 0:
            self.queued_tasks -= 1
        self.running_tasks += 1
        if task_id not in self.active_task_ids:
            self.active_task_ids.append(task_id)
        self.task_statuses[task_id] = TaskLifecycle.RUNNING
        self.started_task_timestamps[task_id] = time.monotonic()
        self.task_started_at_iso[task_id] = datetime.now(timezone.utc).isoformat()
        self.add_event("task_started", f"Task started: {task_id}", task_id=task_id, level="active")
        self.overall_status = "running"

    def mark_task_completed(self, task_id: str, duration_ms: float | None = None) -> None:
        if self.running_tasks > 0:
            self.running_tasks -= 1
        self.completed_tasks += 1
        self.task_statuses[task_id] = TaskLifecycle.COMPLETED
        self._finish_task(task_id, duration_ms)
        self.add_event("task_completed", f"Task completed: {task_id}", task_id=task_id, level="success")

    def mark_task_failed(
        self,
        task_id: str,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        if self.running_tasks > 0:
            self.running_tasks -= 1
        self.failed_tasks += 1
        self.task_statuses[task_id] = TaskLifecycle.FAILED
        if error:
            self.task_error_messages[task_id] = error
        self._finish_task(task_id, duration_ms)
        self.add_event("task_failed", f"Task failed: {task_id}", task_id=task_id, level="error")
        self.overall_status = "degraded"

    def mark_task_skipped(self, task_id: str) -> None:
        self.task_statuses[task_id] = TaskLifecycle.SKIPPED
        self._finish_task(task_id, None)
        self.add_event("task_skipped", f"Task skipped: {task_id}", task_id=task_id, level="warning")

    def record_performance_metrics(
        self,
        task_id: str,
        metrics: dict[str, float | str | None],
        error: str | None = None,
    ) -> None:
        self.performance_metrics = {
            "task_id": task_id,
            **metrics,
            "error": error,
        }
        if error:
            self.add_event("performance_failed", f"Performance benchmark failed: {task_id}", task_id=task_id, level="error")
            self.overall_status = "degraded"
        else:
            self.add_event(
                "performance_completed",
                f"Performance benchmark completed: {task_id}",
                task_id=task_id,
                level="success",
            )

    def _finish_task(self, task_id: str, duration_ms: float | None) -> None:
        self.active_task_ids = [current for current in self.active_task_ids if current != task_id]
        self.recently_finished_task_ids.append(task_id)
        self.recently_finished_task_ids = self.recently_finished_task_ids[-8:]
        self.started_task_timestamps.pop(task_id, None)
        self.task_finished_at_iso[task_id] = datetime.now(timezone.utc).isoformat()
        if duration_ms is None:
            return
        self.completed_task_durations_ms[task_id] = duration_ms
        self.completion_samples_monotonic.append(time.monotonic())
        self.completion_samples_monotonic = self.completion_samples_monotonic[-32:]
