from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TaskResult:
    task_id: str
    evaluator: str
    pillar: str
    source: str
    language: str
    difficulty: str
    passed: bool
    score: float
    raw_output: str
    extracted_code: str
    test_results: list
    error: str | None
    latency_ms: float


class BaseEvaluator(ABC):
    @abstractmethod
    def build_prompt(self, task: dict) -> list[dict]:
        """Convert task YAML dict into chat messages."""
        ...

    @abstractmethod
    def evaluate(self, task: dict, model_output: str) -> TaskResult:
        """Score model output against task definition."""
        ...
