import time
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Callable

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from axbench.client import LLMClient
from axbench.evaluators import PILLAR_MAP, get_evaluator
from axbench.evaluators.base import TaskResult
from axbench.results import BenchmarkRun, RunMetadata


class Runner:
    def __init__(self, client: LLMClient):
        self.client = client

    def run_tasks(
        self,
        tasks: list[dict],
        show_progress: bool = True,
        event_callback: Callable[[str, dict], None] | None = None,
    ) -> BenchmarkRun:
        start = time.monotonic()
        results = []

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                transient=False,
            ) as progress:
                progress_task = progress.add_task("Running tasks...", total=len(tasks))
                for task in tasks:
                    progress.update(progress_task, description=f"[cyan]{task['id']}")
                    task_result = self._run_task(task, event_callback=event_callback)
                    results.append(task_result)
                    progress.advance(progress_task)
        else:
            for task in tasks:
                task_result = self._run_task(task, event_callback=event_callback)
                results.append(task_result)

        duration = time.monotonic() - start
        return BenchmarkRun(
            metadata=RunMetadata(
                model=self.client.model,
                base_url=self.client.base_url,
                timestamp=datetime.now(timezone.utc).isoformat(),
                axbench_version=version("axbench"),
                duration_seconds=round(duration, 2),
            ),
            tasks=results,
        )

    def _run_task(
        self,
        task: dict,
        event_callback: Callable[[str, dict], None] | None = None,
    ) -> TaskResult:
        latency_ms = 0.0
        if event_callback is not None:
            event_callback("task_started", {"task": task})
        try:
            evaluator = get_evaluator(task["evaluator"])
            messages = evaluator.build_prompt(task)
            raw_output, latency_ms = self.client.generate(messages)
            task_result = evaluator.evaluate(task, raw_output)
            task_result.latency_ms = latency_ms
        except Exception as exc:
            task_result = self._error_result(task, str(exc), latency_ms=latency_ms)

        if event_callback is not None:
            event_name = "task_failed" if task_result.error or not task_result.passed else "task_completed"
            event_callback(
                event_name,
                {
                    "task": task,
                    "result": task_result,
                },
            )
        return task_result

    def _error_result(self, task: dict, error: str, latency_ms: float) -> TaskResult:
        source = task.get("source", "general")
        if isinstance(source, str) and source.startswith("team/"):
            pillar = "team_real_world"
        else:
            pillar = PILLAR_MAP.get(task.get("evaluator"), "general_coding")

        return TaskResult(
            task_id=task.get("id", "<unknown-task>"),
            evaluator=task.get("evaluator", "unknown"),
            pillar=pillar,
            source=source,
            language=task.get("language", "unknown"),
            difficulty=task.get("difficulty", "unknown"),
            passed=False,
            score=0.0,
            raw_output="",
            extracted_code="",
            test_results=[],
            error=error,
            latency_ms=latency_ms,
        )
