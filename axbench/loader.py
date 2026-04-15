from pathlib import Path

import yaml

from axbench.evaluators import PILLAR_MAP


class TaskLoader:
    def __init__(self, tasks_dir: Path | str):
        self.tasks_dir = Path(tasks_dir)

    def _all_task_files(self) -> list[Path]:
        return sorted(self.tasks_dir.rglob("*.yaml"))

    def load(
        self,
        evaluator: str | None = None,
        language: str | None = None,
        difficulty: str | None = None,
        source: str | None = None,
        pillar: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        tasks = []
        for file_path in self._all_task_files():
            task = yaml.safe_load(file_path.read_text())
            if evaluator and task.get("evaluator") != evaluator:
                continue
            if language and task.get("language") != language:
                continue
            if difficulty and task.get("difficulty") != difficulty:
                continue
            if source and not task.get("source", "").startswith(source):
                continue
            if pillar and self._task_pillar(task) != pillar:
                continue
            if tags:
                task_tags = task.get("tags", [])
                if not any(tag in task_tags for tag in tags):
                    continue
            tasks.append(task)
        return tasks

    def load_one(self, task_id: str) -> dict:
        for file_path in self._all_task_files():
            task = yaml.safe_load(file_path.read_text())
            if task.get("id") == task_id:
                return task
        raise KeyError(f"Task not found: {task_id!r}")

    def list_tasks(self) -> list[dict]:
        result = []
        for task in self.load():
            result.append(
                {
                    "id": task.get("id"),
                    "evaluator": task.get("evaluator"),
                    "language": task.get("language"),
                    "difficulty": task.get("difficulty"),
                    "source": task.get("source"),
                    "tags": task.get("tags", []),
                    "pillar": self._task_pillar(task),
                }
            )
        return sorted(
            result,
            key=lambda task: (
                task["evaluator"] or "",
                task["language"] or "",
                task["id"] or "",
            ),
        )

    def _task_pillar(self, task: dict) -> str | None:
        source = task.get("source", "")
        if isinstance(source, str) and source.startswith("team/"):
            return "team_real_world"
        return PILLAR_MAP.get(task.get("evaluator"))
