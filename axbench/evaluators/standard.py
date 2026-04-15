from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from axbench.evaluators.base import BaseEvaluator, TaskResult
from axbench.extractor import extract_code


class StandardEvaluator(BaseEvaluator):
    def __init__(self, timeout_seconds: int = 15):
        self._timeout_seconds = timeout_seconds

    def build_prompt(self, task: dict) -> list[dict]:
        kind = self._get_kind(task)
        if kind == "humaneval":
            return self.build_humaneval_prompt(task)
        if kind == "mbpp":
            return self.build_mbpp_prompt(task)
        if kind == "mmlu":
            return self.build_mmlu_prompt(task)
        if kind == "gpqa":
            return self.build_gpqa_prompt(task)
        if kind == "livecodebench":
            return self.build_livecodebench_prompt(task)
        raise ValueError(f"Unknown standard benchmark kind: {kind!r}")

    def evaluate(self, task: dict, model_output: str) -> TaskResult:
        kind = self._get_kind(task)
        if kind == "humaneval":
            return self.evaluate_humaneval(task, model_output)
        if kind == "mbpp":
            return self.evaluate_mbpp(task, model_output)
        if kind == "mmlu":
            return self.evaluate_mmlu(task, model_output)
        if kind == "gpqa":
            return self.evaluate_gpqa(task, model_output)
        if kind == "livecodebench":
            return self.evaluate_livecodebench(task, model_output)
        raise ValueError(f"Unknown standard benchmark kind: {kind!r}")

    def build_humaneval_prompt(self, task: dict) -> list[dict]:
        content = (
            "Complete the following Python function. "
            "Return only the complete function implementation.\n\n"
            f"```python\n{task['prompt'].rstrip()}\n```"
        )
        return [{"role": "user", "content": content}]

    def evaluate_humaneval(self, task: dict, model_output: str) -> TaskResult:
        code = extract_code(model_output, "python")
        candidate = self._build_candidate_code(task, code)
        test_code = task["test"].rstrip()
        entry_point = task.get("entry_point")
        if entry_point and "check(candidate)" in test_code:
            test_code = f"{test_code}\n\ncheck({entry_point})"
        result = self._run_python_code(f"{candidate}\n\n{test_code}\n")
        return self._build_python_result(
            task_id=self._resolve_task_id(task, "humaneval"),
            source="standard/humaneval",
            difficulty="medium",
            model_output=model_output,
            code=code,
            execution=result,
        )

    def build_mbpp_prompt(self, task: dict) -> list[dict]:
        examples = "\n".join(f"  {tc}" for tc in task.get("test_list", [])[:3])
        content = (
            "Write a Python function for the following task.\n\n"
            f"Task: {task['text']}\n\n"
        )
        if examples:
            content += f"Your function should pass these tests:\n{examples}\n\n"
        content += "Return only the function, no explanation."
        return [{"role": "user", "content": content}]

    def evaluate_mbpp(self, task: dict, model_output: str) -> TaskResult:
        code = extract_code(model_output, "python")
        setup_code = task.get("test_setup_code", "").rstrip()
        tests = task.get("challenge_test_list") or task.get("test_list") or []
        test_code = "\n".join(tests)
        combined = f"{code}\n\n"
        if setup_code:
            combined += f"{setup_code}\n\n"
        combined += f"{test_code}\n"
        result = self._run_python_code(combined)
        return self._build_python_result(
            task_id=self._resolve_task_id(task, "mbpp"),
            source="standard/mbpp",
            difficulty="medium",
            model_output=model_output,
            code=code,
            execution=result,
        )

    def build_mmlu_prompt(self, task: dict) -> list[dict]:
        return self._build_multiple_choice_prompt(task)

    def evaluate_mmlu(self, task: dict, model_output: str) -> TaskResult:
        return self._evaluate_multiple_choice(
            task=task,
            model_output=model_output,
            source="standard/mmlu",
            default_prefix="mmlu",
            difficulty="medium",
        )

    def build_gpqa_prompt(self, task: dict) -> list[dict]:
        return self._build_multiple_choice_prompt(task)

    def evaluate_gpqa(self, task: dict, model_output: str) -> TaskResult:
        return self._evaluate_multiple_choice(
            task=task,
            model_output=model_output,
            source="standard/gpqa",
            default_prefix="gpqa",
            difficulty="hard",
        )

    def build_livecodebench_prompt(self, task: dict) -> list[dict]:
        prompt = (
            task.get("prompt")
            or task.get("question_content")
            or task.get("question")
            or task.get("text")
            or ""
        ).strip()
        starter_code = task.get("starter_code")
        content = prompt
        if starter_code:
            content += f"\n\nStarter code:\n```python\n{starter_code.rstrip()}\n```"
        if content:
            content += "\n\n"
        content += "Return only the Python solution code."
        return [{"role": "user", "content": content}]

    def evaluate_livecodebench(self, task: dict, model_output: str) -> TaskResult:
        code = extract_code(model_output, "python")
        candidate = self._build_candidate_code(task, code)
        test_code = self._resolve_livecodebench_test_code(task)
        entry_point = task.get("entry_point")
        if entry_point and "check(candidate)" in test_code:
            test_code = f"{test_code.rstrip()}\n\ncheck({entry_point})"
        result = self._run_python_code(f"{candidate}\n\n{test_code}\n")
        return self._build_python_result(
            task_id=self._resolve_task_id(task, "livecodebench"),
            source="standard/livecodebench",
            difficulty="hard",
            model_output=model_output,
            code=code,
            execution=result,
        )

    def _build_multiple_choice_prompt(self, task: dict) -> list[dict]:
        choices = self._get_choices(task)
        options = "\n".join(f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices))
        content = (
            f"Question: {task['question']}\n\n"
            f"{options}\n\n"
            "Answer with only the letter (A, B, C, or D)."
        )
        return [{"role": "user", "content": content}]

    def _evaluate_multiple_choice(
        self,
        task: dict,
        model_output: str,
        source: str,
        default_prefix: str,
        difficulty: str,
    ) -> TaskResult:
        choices = self._get_choices(task)
        expected = self._normalize_choice_answer(task, choices)
        predicted = self._extract_choice_letter(model_output)
        passed = predicted == expected
        task_id = str(task.get("id") or task.get("task_id") or self._make_question_id(default_prefix, task))
        return TaskResult(
            task_id=task_id,
            evaluator="standard",
            pillar="standard",
            source=source,
            language="text",
            difficulty=difficulty,
            passed=passed,
            score=1.0 if passed else 0.0,
            raw_output=model_output,
            extracted_code="",
            test_results=[
                {
                    "expected": expected,
                    "predicted": predicted,
                    "choices": choices,
                    "passed": passed,
                }
            ],
            error=None,
            latency_ms=0.0,
        )

    def _build_python_result(
        self,
        task_id: str,
        source: str,
        difficulty: str,
        model_output: str,
        code: str,
        execution: dict[str, Any],
    ) -> TaskResult:
        passed = execution["passed"]
        return TaskResult(
            task_id=task_id,
            evaluator="standard",
            pillar="standard",
            source=source,
            language="python",
            difficulty=difficulty,
            passed=passed,
            score=1.0 if passed else 0.0,
            raw_output=model_output,
            extracted_code=code,
            test_results=[execution],
            error=execution.get("error"),
            latency_ms=0.0,
        )

    def _build_candidate_code(self, task: dict, completion: str) -> str:
        entry_point = task.get("entry_point")
        starter_code = task.get("starter_code")
        prompt_code = starter_code or task.get("prompt", "")

        if entry_point and self._has_top_level_definition(completion, entry_point):
            prelude = self._extract_prompt_prelude(prompt_code, entry_point)
            return f"{prelude}\n{completion}".strip() if prelude else completion.strip()

        if starter_code:
            return f"{starter_code.rstrip()}\n{completion}".strip()
        if prompt_code:
            return f"{prompt_code.rstrip()}\n{completion}".strip()
        return completion.strip()

    def _resolve_livecodebench_test_code(self, task: dict) -> str:
        if "test" in task and isinstance(task["test"], str):
            return task["test"].rstrip()
        if "test_code" in task and isinstance(task["test_code"], str):
            return task["test_code"].rstrip()

        collected: list[str] = []
        for key in ("private_test_cases", "public_test_cases", "test_list"):
            value = task.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        collected.append(item)

        if collected:
            return "\n".join(collected).rstrip()
        raise ValueError("LiveCodeBench task is missing executable test code")

    def _run_python_code(self, code: str) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as handle:
            handle.write(code)
            tmp_path = handle.name

        try:
            proc = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
            if proc.returncode == 0:
                return {"passed": True, "error": None, "stdout": proc.stdout.strip()}
            return {
                "passed": False,
                "error": proc.stderr.strip() or proc.stdout.strip(),
                "stdout": proc.stdout.strip(),
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Timeout", "stdout": ""}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _get_kind(self, task: dict) -> str:
        kind = str(task.get("kind", "humaneval")).lower()
        aliases = {
            "gpqa_diamond": "gpqa",
            "live_code_bench": "livecodebench",
            "lcb": "livecodebench",
        }
        return aliases.get(kind, kind)

    def _get_choices(self, task: dict) -> list[str]:
        if "choices" in task:
            return [str(choice) for choice in task["choices"]]

        if "options" in task:
            return [str(choice) for choice in task["options"]]

        correct = task.get("correct_answer") or task.get("Correct Answer")
        incorrect = [
            task.get("incorrect_answer_1") or task.get("Incorrect Answer 1"),
            task.get("incorrect_answer_2") or task.get("Incorrect Answer 2"),
            task.get("incorrect_answer_3") or task.get("Incorrect Answer 3"),
        ]
        if correct and all(value is not None for value in incorrect):
            return [str(correct), *(str(value) for value in incorrect)]

        raise ValueError("Multiple-choice task is missing answer choices")

    def _normalize_choice_answer(self, task: dict, choices: list[str]) -> str:
        answer = task.get("answer")
        if isinstance(answer, int):
            return chr(65 + answer)
        if isinstance(answer, str):
            value = answer.strip().upper()
            if value in {"A", "B", "C", "D"}:
                return value
            if answer in choices:
                return chr(65 + choices.index(answer))

        correct_answer = task.get("correct_answer") or task.get("Correct Answer")
        if correct_answer is not None:
            correct_answer = str(correct_answer)
            if correct_answer in choices:
                return chr(65 + choices.index(correct_answer))

        raise ValueError("Unable to normalize expected multiple-choice answer")

    def _extract_choice_letter(self, model_output: str) -> str:
        match = re.search(r"\b([ABCD])\b", model_output.strip().upper())
        return match.group(1) if match else ""

    def _make_question_id(self, prefix: str, task: dict) -> str:
        subject = str(task.get("subject", "unknown"))
        digest = abs(hash(task["question"])) % 10000
        return f"{prefix}_{subject}_{digest:04d}"

    def _resolve_task_id(self, task: dict, fallback_prefix: str) -> str:
        for key in ("id", "question_id", "task_id"):
            value = task.get(key)
            if value is not None:
                return str(value)
        if "question" in task:
            return self._make_question_id(fallback_prefix, task)
        raise ValueError("Task is missing an identifier")

    def _has_top_level_definition(self, code: str, entry_point: str) -> bool:
        pattern = rf"(^|\n)\s*(async\s+def|def)\s+{re.escape(entry_point)}\s*\("
        return re.search(pattern, code) is not None

    def _extract_prompt_prelude(self, prompt_code: str, entry_point: str) -> str:
        pattern = re.compile(rf"(^|\n)\s*(async\s+def|def)\s+{re.escape(entry_point)}\s*\(")
        match = pattern.search(prompt_code)
        if not match:
            return ""
        return prompt_code[: match.start()].rstrip()
