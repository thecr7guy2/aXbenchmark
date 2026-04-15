from axbench.evaluators.base import BaseEvaluator, TaskResult
from axbench.extractor import extract_code
from axbench.sandbox import Sandbox


class CodeGenEvaluator(BaseEvaluator):
    def __init__(self):
        self._sandbox = Sandbox()

    def build_prompt(self, task: dict) -> list[dict]:
        content = task["prompt"].strip()
        if signature := task.get("function_signature"):
            content += f"\n\nFunction signature: `{signature}`"
        content += "\n\nReturn only the code, no explanation."
        return [{"role": "user", "content": content}]

    def evaluate(self, task: dict, model_output: str) -> TaskResult:
        language = task["language"]
        timeout = task.get("timeout_seconds", 10)
        code = extract_code(model_output, language)
        source = task.get("source", "general")
        pillar = "team_real_world" if source.startswith("team/") else "general_coding"

        if language == "cpp":
            test_results, passed_all = self._evaluate_cpp(task, code, timeout)
        elif language == "bash":
            test_results, passed_all = self._evaluate_bash(task, code, timeout)
        else:
            test_results, passed_all = self._evaluate_python(task, code, timeout)

        score = 0.0
        if test_results:
            passed_count = sum(1 for result in test_results if result["passed"])
            score = passed_count / len(test_results)

        error = next((result.get("error") for result in test_results if result.get("error")), None)

        return TaskResult(
            task_id=task["id"],
            evaluator="code_gen",
            pillar=pillar,
            source=source,
            language=language,
            difficulty=task.get("difficulty", "unknown"),
            passed=passed_all,
            score=round(score, 3),
            raw_output=model_output,
            extracted_code=code,
            test_results=test_results,
            error=error,
            latency_ms=0.0,
        )

    def _evaluate_python(
        self, task: dict, code: str, timeout: int
    ) -> tuple[list[dict], bool]:
        test_results: list[dict] = []
        for test_case in task["test_cases"]:
            result = self._sandbox.run_python(
                code=code,
                test_expression=test_case["input"],
                expected=test_case["expected"],
                timeout=timeout,
            )
            test_results.append(
                {
                    "input": test_case["input"],
                    "expected": test_case["expected"],
                    "actual": result.actual,
                    "passed": result.passed,
                    "error": result.error,
                }
            )
        return test_results, all(result["passed"] for result in test_results)

    def _evaluate_bash(
        self, task: dict, code: str, timeout: int
    ) -> tuple[list[dict], bool]:
        test_results: list[dict] = []
        for test_case in task["test_cases"]:
            result = self._sandbox.run_bash(
                script=code,
                expected_stdout=test_case["expected"],
                timeout=timeout,
                setup_script=test_case.get("input", ""),
                expected_exit_code=test_case.get("expected_exit_code", 0),
                post_check_script=test_case.get("post_check", ""),
                allow_stderr=test_case.get("allow_stderr", False),
            )
            test_results.append(
                {
                    "input": test_case["input"],
                    "expected": test_case["expected"],
                    "expected_exit_code": test_case.get("expected_exit_code", 0),
                    "actual": result.actual,
                    "passed": result.passed,
                    "error": result.error,
                }
            )
        return test_results, all(result["passed"] for result in test_results)

    def _evaluate_cpp(
        self, task: dict, code: str, timeout: int
    ) -> tuple[list[dict], bool]:
        harness = task["test_harness"].replace("{{GENERATED_CODE}}", code)
        result = self._sandbox.run_cpp(harness, timeout=timeout)
        test_results = [
            {
                "harness": "test_harness",
                "passed": result.passed,
                "actual": result.actual,
                "error": result.error,
            }
        ]
        return test_results, result.passed
