import pytest

from axbench.evaluators import get_evaluator
from axbench.evaluators.bug_fix import BugFixEvaluator
from axbench.evaluators.code_gen import CodeGenEvaluator


def test_get_evaluator_returns_code_gen():
    evaluator = get_evaluator("code_gen")
    assert evaluator is not None
    assert isinstance(evaluator, CodeGenEvaluator)


def test_get_evaluator_returns_bug_fix():
    evaluator = get_evaluator("bug_fix")
    assert evaluator is not None
    assert isinstance(evaluator, BugFixEvaluator)


def test_get_evaluator_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown evaluator"):
        get_evaluator("nonexistent")
