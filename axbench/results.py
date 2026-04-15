import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from axbench.evaluators.base import TaskResult


@dataclass
class RunMetadata:
    model: str
    base_url: str
    timestamp: str
    axbench_version: str
    duration_seconds: float


@dataclass
class BenchmarkRun:
    metadata: RunMetadata
    tasks: list[TaskResult]
    selected_task_ids: list[str] = field(default_factory=list)
    skipped_task_ids: list[str] = field(default_factory=list)

    def overall_quality_score(self) -> float:
        quality_tasks = [t for t in self.tasks if t.pillar != "performance"]
        if not quality_tasks:
            return 0.0
        return sum(1 for t in quality_tasks if t.passed) / len(quality_tasks)

    def save(self, path: Path) -> None:
        path = Path(path)
        data = {
            "metadata": asdict(self.metadata),
            "summary": self._build_summary(),
            "selection": {
                "selected_task_ids": self.selected_task_ids,
                "skipped_task_ids": self.skipped_task_ids,
            },
            "tasks": [asdict(t) for t in self.tasks],
        }
        path.write_text(json.dumps(data, indent=2))

    def _build_summary(self) -> dict:
        quality_tasks = [t for t in self.tasks if t.pillar != "performance"]

        by_pillar: dict = {}
        for t in quality_tasks:
            by_pillar.setdefault(t.pillar, {"total": 0, "passed": 0})
            by_pillar[t.pillar]["total"] += 1
            if t.passed:
                by_pillar[t.pillar]["passed"] += 1
        for p in by_pillar.values():
            p["score"] = round(p["passed"] / p["total"], 3) if p["total"] else 0.0

        by_language: dict = {}
        for t in quality_tasks:
            by_language.setdefault(t.language, {"total": 0, "passed": 0})
            by_language[t.language]["total"] += 1
            if t.passed:
                by_language[t.language]["passed"] += 1
        for l in by_language.values():
            l["score"] = round(l["passed"] / l["total"], 3) if l["total"] else 0.0

        by_difficulty: dict = {}
        for t in quality_tasks:
            by_difficulty.setdefault(t.difficulty, {"total": 0, "passed": 0})
            by_difficulty[t.difficulty]["total"] += 1
            if t.passed:
                by_difficulty[t.difficulty]["passed"] += 1
        for d in by_difficulty.values():
            d["score"] = round(d["passed"] / d["total"], 3) if d["total"] else 0.0

        by_source: dict = {}
        for t in quality_tasks:
            by_source.setdefault(t.source, {"total": 0, "passed": 0})
            by_source[t.source]["total"] += 1
            if t.passed:
                by_source[t.source]["passed"] += 1
        for s in by_source.values():
            s["score"] = round(s["passed"] / s["total"], 3) if s["total"] else 0.0

        performance = None
        for task in self.tasks:
            if task.pillar != "performance":
                continue
            metrics = task.test_results[0] if task.test_results else {}
            performance = {
                "task_id": task.task_id,
                "source": task.source,
                "error": task.error,
                "pp_tokens_per_sec": metrics.get("pp_tokens_per_sec", 0.0),
                "tg_tokens_per_sec": metrics.get("tg_tokens_per_sec", 0.0),
                "peak_tg_tokens_per_sec": metrics.get("peak_tg_tokens_per_sec", 0.0),
                "ttft_ms": metrics.get("ttft_ms", 0.0),
            }
            break

        return {
            "overall_quality_score": round(self.overall_quality_score(), 3),
            "executed_tasks": len(self.tasks),
            "skipped_tasks": len(self.skipped_task_ids),
            "passed_tasks": sum(1 for t in self.tasks if t.passed),
            "failed_tasks": sum(1 for t in self.tasks if not t.passed),
            "errored_tasks": sum(1 for t in self.tasks if t.error),
            "by_pillar": by_pillar,
            "by_language": by_language,
            "by_difficulty": by_difficulty,
            "by_source": by_source,
            "performance": performance,
        }

    @classmethod
    def load(cls, path: Path) -> "BenchmarkRun":
        data = json.loads(Path(path).read_text())
        metadata = RunMetadata(**data["metadata"])
        tasks = [TaskResult(**t) for t in data["tasks"]]
        selection = data.get("selection", {})
        selected_task_ids = selection.get("selected_task_ids") or [task.task_id for task in tasks]
        skipped_task_ids = selection.get("skipped_task_ids") or []
        return cls(
            metadata=metadata,
            tasks=tasks,
            selected_task_ids=selected_task_ids,
            skipped_task_ids=skipped_task_ids,
        )
