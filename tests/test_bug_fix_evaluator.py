from axbench.evaluators.bug_fix import BugFixEvaluator


BUG_TASK = {
    "id": "python_off_by_one",
    "evaluator": "bug_fix",
    "language": "python",
    "difficulty": "easy",
    "source": "general",
    "buggy_code": (
        "def sum_list(numbers):\n"
        "    total = 0\n"
        "    for i in range(1, len(numbers)):\n"
        "        total += numbers[i]\n"
        "    return total"
    ),
    "prompt": (
        "The following Python function has a bug. Find and fix it.\n"
        "Return only the corrected function.\n\n"
        "```python\n"
        "def sum_list(numbers):\n"
        "    total = 0\n"
        "    for i in range(1, len(numbers)):\n"
        "        total += numbers[i]\n"
        "    return total\n"
        "```"
    ),
    "test_cases": [
        {"input": "sum_list([1, 2, 3])", "expected": 6},
        {"input": "sum_list([10])", "expected": 10},
        {"input": "sum_list([])", "expected": 0},
    ],
    "timeout_seconds": 10,
}


def test_build_prompt_includes_buggy_code():
    ev = BugFixEvaluator()
    messages = ev.build_prompt(BUG_TASK)
    assert messages[0]["role"] == "user"
    assert "bug" in messages[0]["content"].lower()


def test_evaluate_fixed_code_passes():
    ev = BugFixEvaluator()
    fixed = "```python\ndef sum_list(numbers):\n    return sum(numbers)\n```"
    result = ev.evaluate(BUG_TASK, fixed)
    assert result.passed is True
    assert result.score == 1.0
    assert result.evaluator == "bug_fix"


def test_evaluate_unfixed_code_fails():
    ev = BugFixEvaluator()
    unfixed = (
        "```python\n"
        "def sum_list(numbers):\n"
        "    total = 0\n"
        "    for i in range(1, len(numbers)):\n"
        "        total += numbers[i]\n"
        "    return total\n"
        "```"
    )
    result = ev.evaluate(BUG_TASK, unfixed)
    assert result.passed is False
