from axbench.evaluators.code_gen import CodeGenEvaluator


PYTHON_TASK = {
    "id": "python_add",
    "evaluator": "code_gen",
    "language": "python",
    "difficulty": "easy",
    "source": "general",
    "prompt": "Write a Python function called `add` that returns the sum of two integers.",
    "test_cases": [
        {"input": "add(1, 2)", "expected": 3},
        {"input": "add(-1, 1)", "expected": 0},
    ],
    "timeout_seconds": 10,
}


def test_build_prompt_returns_user_message():
    ev = CodeGenEvaluator()
    messages = ev.build_prompt(PYTHON_TASK)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "add" in messages[0]["content"]


def test_evaluate_passing_code():
    ev = CodeGenEvaluator()
    output = "```python\ndef add(a, b):\n    return a + b\n```"
    result = ev.evaluate(PYTHON_TASK, output)
    assert result.passed is True
    assert result.score == 1.0
    assert result.task_id == "python_add"
    assert len(result.test_results) == 2


def test_evaluate_failing_code():
    ev = CodeGenEvaluator()
    output = "```python\ndef add(a, b):\n    return a - b\n```"
    result = ev.evaluate(PYTHON_TASK, output)
    assert result.passed is False
    assert result.score < 1.0


def test_evaluate_partial_pass():
    ev = CodeGenEvaluator()
    output = "```python\ndef add(a, b):\n    return abs(a) + abs(b)\n```"
    result = ev.evaluate(PYTHON_TASK, output)
    assert 0.0 < result.score < 1.0
    assert result.passed is False
