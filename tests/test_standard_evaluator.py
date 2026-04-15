import pytest

from axbench.evaluators.standard import StandardEvaluator


def test_humaneval_build_prompt():
    ev = StandardEvaluator()
    task = {
        "kind": "humaneval",
        "id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers, threshold):\n",
        "entry_point": "has_close_elements",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0], 0.1) is False",
    }
    messages = ev.build_prompt(task)
    assert messages[0]["role"] == "user"
    assert "has_close_elements" in messages[0]["content"]


def test_humaneval_evaluate_passing():
    ev = StandardEvaluator()
    task = {
        "kind": "humaneval",
        "id": "HumanEval/test",
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0",
    }
    output = "```python\ndef add(a, b):\n    return a + b\n```"
    result = ev.evaluate(task, output)
    assert result.passed is True
    assert result.task_id == "HumanEval/test"
    assert result.pillar == "standard"


def test_humaneval_evaluate_failing():
    ev = StandardEvaluator()
    task = {
        "kind": "humaneval",
        "id": "HumanEval/test",
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
    }
    output = "```python\ndef add(a, b):\n    return a - b\n```"
    result = ev.evaluate(task, output)
    assert result.passed is False


def test_mbpp_evaluate_uses_challenge_tests():
    ev = StandardEvaluator()
    task = {
        "kind": "mbpp",
        "task_id": 7,
        "text": "Write a function add(a, b).",
        "test_list": ["assert add(1, 2) == 3"],
        "challenge_test_list": ["assert add(-1, 1) == 0"],
    }
    output = "```python\ndef add(a, b):\n    return a + b\n```"
    result = ev.evaluate(task, output)
    assert result.passed is True
    assert result.task_id == "7"
    assert result.source == "standard/mbpp"


def test_mbpp_evaluate_runs_setup_code_before_tests():
    ev = StandardEvaluator()
    task = {
        "kind": "mbpp",
        "task_id": 8,
        "text": "Write a function add(a, b).",
        "test_setup_code": "value = 41",
        "test_list": ["assert add(value, 1) == 42"],
    }
    output = "```python\ndef add(a, b):\n    return a + b\n```"
    result = ev.evaluate(task, output)
    assert result.passed is True
    assert result.task_id == "8"


def test_mmlu_prompt_and_evaluation():
    ev = StandardEvaluator()
    task = {
        "kind": "mmlu",
        "id": "mmlu_math_1",
        "question": "What is 2 + 2?",
        "choices": ["1", "3", "4", "5"],
        "answer": 2,
        "subject": "math",
    }
    prompt = ev.build_prompt(task)
    assert "A) 1" in prompt[0]["content"]
    result = ev.evaluate(task, "C")
    assert result.passed is True
    assert result.source == "standard/mmlu"


def test_gpqa_normalizes_correct_answer_text():
    ev = StandardEvaluator()
    task = {
        "kind": "gpqa_diamond",
        "id": "gpqa_chem_1",
        "question": "What is the chemical formula for water?",
        "choices": ["CO2", "H2O", "O2", "NaCl"],
        "correct_answer": "H2O",
        "subject": "chemistry",
    }
    result = ev.evaluate(task, "Final answer: B")
    assert result.passed is True
    assert result.source == "standard/gpqa"
    assert result.difficulty == "hard"


def test_livecodebench_build_prompt_includes_starter_code():
    ev = StandardEvaluator()
    task = {
        "kind": "livecodebench",
        "id": "LCB/1",
        "prompt": "Complete the square function.",
        "starter_code": "def square(n):\n    pass",
        "entry_point": "square",
        "test": "assert square(3) == 9",
    }
    prompt = ev.build_prompt(task)
    assert "Starter code" in prompt[0]["content"]
    assert "def square" in prompt[0]["content"]


def test_livecodebench_evaluate_passing():
    ev = StandardEvaluator()
    task = {
        "kind": "livecodebench",
        "question_id": "LCB/1",
        "prompt": "Write a function that squares a number.",
        "entry_point": "square",
        "test": "assert square(3) == 9\nassert square(0) == 0",
    }
    output = "```python\ndef square(n):\n    return n * n\n```"
    result = ev.evaluate(task, output)
    assert result.passed is True
    assert result.task_id == "LCB/1"
    assert result.source == "standard/livecodebench"


def test_unknown_standard_kind_raises():
    ev = StandardEvaluator()
    with pytest.raises(ValueError, match="Unknown standard benchmark kind"):
        ev.build_prompt({"kind": "unknown"})
