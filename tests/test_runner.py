from unittest.mock import MagicMock, patch

from axbench.evaluators.base import TaskResult
from axbench.runner import Runner


MOCK_TASK = {
    "id": "python_add",
    "evaluator": "code_gen",
    "language": "python",
    "difficulty": "easy",
    "source": "general",
    "prompt": "Write add(a,b)",
    "test_cases": [{"input": "add(1,2)", "expected": 3}],
    "timeout_seconds": 10,
}


def _mock_result(task_id: str) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        evaluator="code_gen",
        pillar="general_coding",
        source="general",
        language="python",
        difficulty="easy",
        passed=True,
        score=1.0,
        raw_output="def add(a,b): return a+b",
        extracted_code="def add(a,b): return a+b",
        test_results=[],
        error=None,
        latency_ms=100.0,
    )


def test_runner_runs_tasks_and_returns_benchmark_run():
    mock_client = MagicMock()
    mock_client.model = "mock-model"
    mock_client.base_url = "http://localhost:8000/v1"
    mock_client.generate.return_value = ("def add(a,b): return a+b", 100.0)

    mock_evaluator = MagicMock()
    mock_evaluator.build_prompt.return_value = [{"role": "user", "content": "test"}]
    mock_evaluator.evaluate.return_value = _mock_result("python_add")

    runner = Runner(client=mock_client)
    with patch("axbench.runner.get_evaluator", return_value=mock_evaluator):
        result = runner.run_tasks([MOCK_TASK])

    assert len(result.tasks) == 1
    assert result.tasks[0].passed is True
    assert result.metadata.model == mock_client.model


def test_runner_sets_latency_from_client():
    mock_client = MagicMock()
    mock_client.model = "mock-model"
    mock_client.base_url = "http://localhost:8000/v1"
    mock_client.generate.return_value = ("code", 250.0)

    mock_evaluator = MagicMock()
    mock_evaluator.build_prompt.return_value = [{"role": "user", "content": "test"}]
    task_result = _mock_result("python_add")
    mock_evaluator.evaluate.return_value = task_result

    runner = Runner(client=mock_client)
    with patch("axbench.runner.get_evaluator", return_value=mock_evaluator):
        result = runner.run_tasks([MOCK_TASK])

    assert result.tasks[0].latency_ms == 250.0


def test_runner_records_failed_task_instead_of_crashing():
    mock_client = MagicMock()
    mock_client.model = "mock-model"
    mock_client.base_url = "http://localhost:8000/v1"
    mock_client.generate.side_effect = ValueError("timed out after 5s")

    mock_evaluator = MagicMock()
    mock_evaluator.build_prompt.return_value = [{"role": "user", "content": "test"}]

    runner = Runner(client=mock_client)
    with patch("axbench.runner.get_evaluator", return_value=mock_evaluator):
        result = runner.run_tasks([MOCK_TASK])

    assert len(result.tasks) == 1
    assert result.tasks[0].passed is False
    assert "timed out" in result.tasks[0].error


def test_runner_emits_task_lifecycle_events():
    mock_client = MagicMock()
    mock_client.model = "mock-model"
    mock_client.base_url = "http://localhost:8000/v1"
    mock_client.generate.return_value = ("code", 150.0)

    mock_evaluator = MagicMock()
    mock_evaluator.build_prompt.return_value = [{"role": "user", "content": "test"}]
    mock_evaluator.evaluate.return_value = _mock_result("python_add")

    events = []

    runner = Runner(client=mock_client)
    with patch("axbench.runner.get_evaluator", return_value=mock_evaluator):
        runner.run_tasks(
            [MOCK_TASK],
            show_progress=False,
            event_callback=lambda event_name, payload: events.append((event_name, payload["task"]["id"])),
        )

    assert events == [
        ("task_started", "python_add"),
        ("task_completed", "python_add"),
    ]
