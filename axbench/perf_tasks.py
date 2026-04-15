from __future__ import annotations

from copy import deepcopy


_PERFORMANCE_TASKS = [
    {
        "id": "performance_llama_benchy",
        "evaluator": "perf",
        "language": "text",
        "difficulty": "benchmark",
        "source": "performance/llama-benchy",
        "tags": ["throughput", "latency"],
    }
]


def load_performance_tasks() -> list[dict]:
    return deepcopy(_PERFORMANCE_TASKS)
