import json
from pathlib import Path
from axbench.results import BenchmarkRun, LEGACY_BENCHMARK_SUITE_VERSION, RunMetadata, TaskResult


def test_benchmark_run_save_and_load(tmp_path):
    run = BenchmarkRun(
        metadata=RunMetadata(
            model="test-model",
            base_url="http://localhost:8000/v1",
            timestamp="2026-04-14T10:00:00",
            axbench_version="0.1.0",
            duration_seconds=10.0,
            benchmark_suite_version="axbench-v2",
            quick_mode=False,
            warnings=["GPQA Diamond skipped: HF_TOKEN not set"],
        ),
        tasks=[
            TaskResult(
                task_id="python_test",
                evaluator="code_gen",
                pillar="general_coding",
                source="general",
                language="python",
                difficulty="easy",
                passed=True,
                score=1.0,
                raw_output="def foo(): return 1",
                extracted_code="def foo(): return 1",
                test_results=[{"input": "foo()", "expected": 1, "actual": 1, "passed": True}],
                error=None,
                latency_ms=500.0,
            )
        ],
        selected_task_ids=["python_test"],
        skipped_task_ids=["cpp_test"],
    )
    path = tmp_path / "result.json"
    run.save(path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["selection"]["selected_task_ids"] == ["python_test"]
    assert data["selection"]["skipped_task_ids"] == ["cpp_test"]
    loaded = BenchmarkRun.load(path)
    assert loaded.metadata.model == "test-model"
    assert loaded.metadata.benchmark_suite_version == "axbench-v2"
    assert loaded.metadata.quick_mode is False
    assert loaded.metadata.warnings == ["GPQA Diamond skipped: HF_TOKEN not set"]
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].task_id == "python_test"
    assert loaded.tasks[0].passed is True
    assert loaded.selected_task_ids == ["python_test"]
    assert loaded.skipped_task_ids == ["cpp_test"]


def test_benchmark_run_overall_quality_score():
    run = BenchmarkRun(
        metadata=RunMetadata(
            model="m", base_url="u", timestamp="t",
            axbench_version="0.1.0", duration_seconds=1.0,
        ),
        tasks=[
            TaskResult(task_id="a", evaluator="code_gen", pillar="general_coding",
                       source="general", language="python", difficulty="easy",
                       passed=True, score=1.0, raw_output="", extracted_code="",
                       test_results=[], error=None, latency_ms=0),
            TaskResult(task_id="b", evaluator="code_gen", pillar="general_coding",
                       source="general", language="python", difficulty="easy",
                       passed=False, score=0.0, raw_output="", extracted_code="",
                       test_results=[], error=None, latency_ms=0),
        ],
    )
    assert run.overall_quality_score() == 0.5


def test_benchmark_run_loads_legacy_metadata_defaults(tmp_path: Path):
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "model": "legacy-model",
                    "base_url": "http://localhost:8000/v1",
                    "timestamp": "2026-04-14T10:00:00",
                    "axbench_version": "0.1.0",
                    "duration_seconds": 10.0,
                },
                "selection": {
                    "selected_task_ids": ["python_test"],
                    "skipped_task_ids": [],
                },
                "tasks": [
                    {
                        "task_id": "python_test",
                        "evaluator": "code_gen",
                        "pillar": "general_coding",
                        "source": "general",
                        "language": "python",
                        "difficulty": "easy",
                        "passed": True,
                        "score": 1.0,
                        "raw_output": "ok",
                        "extracted_code": "ok",
                        "test_results": [],
                        "error": None,
                        "latency_ms": 100.0,
                    }
                ],
            }
        )
    )

    loaded = BenchmarkRun.load(legacy_path)
    assert loaded.metadata.benchmark_suite_version == LEGACY_BENCHMARK_SUITE_VERSION
    assert loaded.metadata.quick_mode is False
    assert loaded.metadata.warnings == []
