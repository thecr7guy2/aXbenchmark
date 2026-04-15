from axbench.evaluators.base import BaseEvaluator


PILLAR_MAP = {
    "code_gen": "general_coding",
    "bug_fix": "general_coding",
    "standard": "standard",
    "perf": "performance",
}


def get_evaluator(name: str) -> BaseEvaluator:
    """Return an evaluator instance by name."""
    if name == "code_gen":
        from axbench.evaluators.code_gen import CodeGenEvaluator

        return CodeGenEvaluator()
    if name == "bug_fix":
        from axbench.evaluators.bug_fix import BugFixEvaluator

        return BugFixEvaluator()
    if name == "standard":
        from axbench.evaluators.standard import StandardEvaluator

        return StandardEvaluator()
    if name == "perf":
        from axbench.evaluators.perf import PerfEvaluator

        return PerfEvaluator()
    raise ValueError(
        f"Unknown evaluator: {name!r}. "
        "Valid options: code_gen, bug_fix, standard, perf"
    )


__all__ = ["BaseEvaluator", "PILLAR_MAP", "get_evaluator"]
