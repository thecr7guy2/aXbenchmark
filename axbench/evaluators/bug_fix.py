from axbench.evaluators.base import BaseEvaluator, TaskResult
from axbench.evaluators.code_gen import CodeGenEvaluator


class BugFixEvaluator(BaseEvaluator):
    """Bug fix evaluation reuses CodeGenEvaluator scoring logic."""

    def __init__(self):
        self._code_gen = CodeGenEvaluator()

    def build_prompt(self, task: dict) -> list[dict]:
        content = task["prompt"].strip()
        content += "\n\nReturn only the corrected code, no explanation."
        return [{"role": "user", "content": content}]

    def evaluate(self, task: dict, model_output: str) -> TaskResult:
        result = self._code_gen.evaluate(task, model_output)
        result.evaluator = "bug_fix"
        return result
