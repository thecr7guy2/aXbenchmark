import os
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from axbench.cli import cli
from axbench.evaluators.base import TaskResult
from axbench.results import BenchmarkRun, RunMetadata
from axbench.standard_loader import StandardTaskBundle


def _sample_standard_tasks() -> list[dict]:
    return [
        {
            "id": "mmlu/college_mathematics/0000",
            "evaluator": "standard",
            "kind": "mmlu",
            "language": "text",
            "difficulty": "hard",
            "source": "standard/mmlu",
            "tags": ["reasoning"],
            "subject": "college_mathematics",
            "question": "What is 2 + 2?",
            "choices": ["1", "3", "4", "5"],
            "answer": 2,
        },
        {
            "id": "HumanEval/0",
            "evaluator": "standard",
            "kind": "humaneval",
            "language": "python",
            "difficulty": "hard",
            "source": "standard/humaneval",
            "tags": ["python"],
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n\ncheck(add)",
        },
    ]


def _sample_perf_tasks() -> list[dict]:
    return [
        {
            "id": "performance_llama_benchy",
            "evaluator": "perf",
            "language": "text",
            "difficulty": "benchmark",
            "source": "performance/llama-benchy",
            "tags": ["throughput", "latency"],
        }
    ]


def _sample_standard_bundle(warnings: list[str] | None = None) -> StandardTaskBundle:
    return StandardTaskBundle(tasks=_sample_standard_tasks(), warnings=warnings or [])


def _sample_run(model: str, passed: bool = True, task_id: str = "python_add") -> BenchmarkRun:
    return BenchmarkRun(
        metadata=RunMetadata(
            model=model,
            base_url="http://localhost:8000/v1",
            timestamp="2026-04-14T12:00:00+00:00",
            axbench_version="0.1.0",
            duration_seconds=1.5,
        ),
        tasks=[
            TaskResult(
                task_id=task_id,
                evaluator="code_gen",
                pillar="general_coding",
                source="general",
                language="python",
                difficulty="easy",
                passed=passed,
                score=1.0 if passed else 0.0,
                raw_output="def add(a, b): return a + b",
                extracted_code="def add(a, b): return a + b",
                test_results=[],
                error=None,
                latency_ms=123.0,
            )
        ],
    )


def test_run_help_shows_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--base-url" in result.output
    assert "--model" in result.output
    assert "--quick" in result.output
    assert "--pillar" in result.output


def test_run_single_task_executes_and_saves_results(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    task_file = tasks_dir / "general" / "code_gen" / "python" / "task.yaml"
    task_file.parent.mkdir(parents=True)
    task_file.write_text(
        "id: python_task_a\n"
        "evaluator: code_gen\n"
        "language: python\n"
        "difficulty: easy\n"
        "source: general\n"
        "prompt: test\n"
        "test_cases: []\n"
        "timeout_seconds: 10\n"
    )
    output_path = tmp_path / "result.json"

    runner = CliRunner()
    with patch(
        "axbench.cli.load_standard_task_bundle",
        return_value=StandardTaskBundle(tasks=[], warnings=[]),
    ), patch(
        "axbench.cli.load_performance_tasks",
        return_value=[],
    ), patch(
        "axbench.cli.Runner.run_tasks",
        return_value=_sample_run("mock-model", task_id="python_task_a"),
    ):
        result = runner.invoke(
            cli,
            [
                "run",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "mock-model",
                "--task",
                "python_task_a",
                "--tasks-dir",
                str(tasks_dir),
                "--save",
                str(output_path),
            ],
        )

    assert result.exit_code == 0
    assert output_path.exists()
    saved = BenchmarkRun.load(output_path)
    assert saved.selected_task_ids == ["python_task_a"]
    assert saved.skipped_task_ids == []
    assert "Results saved to" in result.output
    assert "Queued Tasks" in result.output
    assert "Executed Task Results" in result.output


def test_run_save_does_not_overwrite_existing_results(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    task_file = tasks_dir / "general" / "code_gen" / "python" / "task.yaml"
    task_file.parent.mkdir(parents=True)
    task_file.write_text(
        "id: python_task_a\n"
        "evaluator: code_gen\n"
        "language: python\n"
        "difficulty: easy\n"
        "source: general\n"
        "prompt: test\n"
        "test_cases: []\n"
        "timeout_seconds: 10\n"
    )
    output_path = tmp_path / "result.json"
    output_path.write_text('{"existing": true}')

    runner = CliRunner()
    with patch(
        "axbench.cli.load_standard_task_bundle",
        return_value=StandardTaskBundle(tasks=[], warnings=[]),
    ), patch(
        "axbench.cli.load_performance_tasks",
        return_value=[],
    ), patch(
        "axbench.cli.Runner.run_tasks",
        return_value=_sample_run("mock-model", task_id="python_task_a"),
    ):
        result = runner.invoke(
            cli,
            [
                "run",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "mock-model",
                "--task",
                "python_task_a",
                "--tasks-dir",
                str(tasks_dir),
                "--save",
                str(output_path),
            ],
        )

    assert result.exit_code == 0
    assert output_path.read_text() == '{"existing": true}'
    unique_path = tmp_path / "result-1.json"
    assert unique_path.exists()
    saved = BenchmarkRun.load(unique_path)
    assert saved.selected_task_ids == ["python_task_a"]
    assert "Results saved to" in result.output
    assert "result-1.json" in result.output


def test_run_reports_skipped_tasks_when_filters_apply(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    python_file = tasks_dir / "general" / "code_gen" / "python" / "task_a.yaml"
    python_file.parent.mkdir(parents=True)
    python_file.write_text(
        "id: python_task_a\n"
        "evaluator: code_gen\n"
        "language: python\n"
        "difficulty: easy\n"
        "source: general\n"
        "prompt: test\n"
        "test_cases: []\n"
        "timeout_seconds: 10\n"
    )
    cpp_file = tasks_dir / "general" / "code_gen" / "cpp" / "task_b.yaml"
    cpp_file.parent.mkdir(parents=True)
    cpp_file.write_text(
        "id: cpp_task_b\n"
        "evaluator: code_gen\n"
        "language: cpp\n"
        "difficulty: easy\n"
        "source: general\n"
        "prompt: test\n"
        "test_cases: []\n"
        "timeout_seconds: 10\n"
    )

    runner = CliRunner()
    with patch("axbench.cli.load_standard_task_bundle", return_value=_sample_standard_bundle()), patch(
        "axbench.cli.load_performance_tasks",
        return_value=_sample_perf_tasks(),
    ), patch(
        "axbench.cli.Runner.run_tasks",
        return_value=_sample_run("mock-model", task_id="python_task_a"),
    ):
        result = runner.invoke(
            cli,
            [
                "run",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "mock-model",
                "--language",
                "python",
                "--tasks-dir",
                str(tasks_dir),
            ],
        )

    assert result.exit_code == 0
    assert "Selection: 2 queued, 3 skipped" in result.output
    assert "python_task_a" in result.output
    assert "cpp_task_b" in result.output
    assert "HumanEval/0" in result.output
    assert "Executed: 1  Skipped: 3" in result.output


def test_list_tasks_filters_by_pillar(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    general_file = tasks_dir / "general" / "code_gen" / "python" / "task_a.yaml"
    general_file.parent.mkdir(parents=True)
    general_file.write_text(
        "id: python_task_a\n"
        "evaluator: code_gen\n"
        "language: python\n"
        "difficulty: easy\n"
        "source: general\n"
        "tags: [algorithms]\n"
        "prompt: test\n"
        "test_cases: []\n"
        "timeout_seconds: 10\n"
    )
    team_file = tasks_dir / "team" / "tom" / "task_b.yaml"
    team_file.parent.mkdir(parents=True)
    team_file.write_text(
        "id: team_task_b\n"
        "evaluator: bug_fix\n"
        "language: cpp\n"
        "difficulty: medium\n"
        "source: team/tom\n"
        "tags: [networking]\n"
        "prompt: test\n"
        "test_harness: ''\n"
        "timeout_seconds: 10\n"
    )

    runner = CliRunner()
    with patch("axbench.cli.load_standard_task_bundle", return_value=_sample_standard_bundle()), patch(
        "axbench.cli.load_performance_tasks",
        return_value=_sample_perf_tasks(),
    ):
        result = runner.invoke(
            cli,
            ["list-tasks", "--pillar", "team_real_world", "--tasks-dir", str(tasks_dir)],
        )

    assert result.exit_code == 0
    assert "team_task_b" in result.output
    assert "python_task_a" not in result.output


def test_list_tasks_includes_standard_and_perf_builtins(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    runner = CliRunner()
    with patch("axbench.cli.load_standard_task_bundle", return_value=_sample_standard_bundle()), patch(
        "axbench.cli.load_performance_tasks",
        return_value=_sample_perf_tasks(),
    ):
        result = runner.invoke(cli, ["list-tasks", "--tasks-dir", str(tasks_dir)])

    assert result.exit_code == 0
    assert "mmlu/college_mathematics/0000" in result.output
    assert "HumanEval/0" in result.output
    assert "performance_llama_benchy" in result.output


def test_compare_outputs_summary_and_recommendation(tmp_path: Path):
    run_a = _sample_run("model-a", passed=True)
    run_b = _sample_run("model-b", passed=False)
    result_a = tmp_path / "a.json"
    result_b = tmp_path / "b.json"
    run_a.save(result_a)
    run_b.save(result_b)

    runner = CliRunner()
    result = runner.invoke(cli, ["compare", str(result_a), str(result_b)])

    assert result.exit_code == 0
    assert "AXBench Model Comparison" in result.output
    assert "model-a" in result.output
    assert "model-b" in result.output
    assert "FULL / axbench-v2" in result.output
    assert "Recommendation:" in result.output


def test_compare_shows_quick_label(tmp_path: Path):
    run_a = _sample_run("model-a", passed=True)
    run_a.metadata.quick_mode = True
    run_b = _sample_run("model-b", passed=False)
    result_a = tmp_path / "a.json"
    result_b = tmp_path / "b.json"
    run_a.save(result_a)
    run_b.save(result_b)

    runner = CliRunner()
    result = runner.invoke(cli, ["compare", str(result_a), str(result_b)])

    assert result.exit_code == 0
    assert "QUICK / axbench-v2" in result.output
    assert "FULL / axbench-v2" in result.output


def test_run_default_suite_includes_standard_and_perf(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    task_file = tasks_dir / "general" / "code_gen" / "python" / "task.yaml"
    task_file.parent.mkdir(parents=True)
    task_file.write_text(
        "id: python_task_a\n"
        "evaluator: code_gen\n"
        "language: python\n"
        "difficulty: easy\n"
        "source: general\n"
        "prompt: test\n"
        "test_cases: []\n"
        "timeout_seconds: 10\n"
    )

    def fake_run_tasks(task_defs, **_kwargs):
        results = []
        for task in task_defs:
            results.append(
                TaskResult(
                    task_id=task["id"],
                    evaluator=task["evaluator"],
                    pillar="standard" if task["evaluator"] == "standard" else "general_coding",
                    source=task["source"],
                    language=task["language"],
                    difficulty=task["difficulty"],
                    passed=True,
                    score=1.0,
                    raw_output="ok",
                    extracted_code="ok",
                    test_results=[],
                    error=None,
                    latency_ms=10.0,
                )
            )
        return BenchmarkRun(
            metadata=RunMetadata(
                model="mock-model",
                base_url="http://localhost:8000/v1",
                timestamp="2026-04-14T12:00:00+00:00",
                axbench_version="0.1.0",
                duration_seconds=1.5,
            ),
            tasks=results,
        )

    perf_result = TaskResult(
        task_id="performance_llama_benchy",
        evaluator="perf",
        pillar="performance",
        source="performance/llama-benchy",
        language="text",
        difficulty="benchmark",
        passed=True,
        score=0.0,
        raw_output="{}",
        extracted_code="",
        test_results=[
            {
                "pp_tokens_per_sec": 5000.0,
                "tg_tokens_per_sec": 60.0,
                "peak_tg_tokens_per_sec": 65.0,
                "ttft_ms": 400.0,
            }
        ],
        error=None,
        latency_ms=400.0,
    )

    runner = CliRunner()
    with patch("axbench.cli.load_standard_task_bundle", return_value=_sample_standard_bundle()), patch(
        "axbench.cli.load_performance_tasks",
        return_value=_sample_perf_tasks(),
    ), patch("axbench.cli.Runner.run_tasks", side_effect=fake_run_tasks), patch(
        "axbench.cli._run_perf_task",
        return_value=perf_result,
    ):
        result = runner.invoke(
            cli,
            [
                "run",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "mock-model",
                "--tasks-dir",
                str(tasks_dir),
            ],
        )

    assert result.exit_code == 0
    assert "mmlu/college_mathematics/0000" in result.output
    assert "HumanEval/0" in result.output
    assert "performance_llama_benchy" in result.output
    assert "Starting performance benchmark:" in result.output
    assert "Completed performance benchmark:" in result.output
    assert "Performance (llama-benchy)" in result.output


def test_download_command_prints_cache_summary(tmp_path: Path):
    runner = CliRunner()
    with patch(
        "axbench.cli.download_standard_tasks",
        return_value={"mmlu_college_mathematics": 10, "humaneval": 40},
    ):
        result = runner.invoke(cli, ["download", "--tasks-dir", str(tmp_path / "tasks")])

    assert result.exit_code == 0
    assert "AXBench Standard Dataset Cache" in result.output
    assert "mmlu_college_mathematics" in result.output
    assert "humaneval" in result.output


def test_cli_loads_hf_token_from_dotenv(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    runner = CliRunner()
    with patch(
        "axbench.cli.load_standard_task_bundle",
        return_value=StandardTaskBundle(tasks=[], warnings=[]),
    ), patch(
        "axbench.cli.load_performance_tasks",
        return_value=[],
    ):
        result = runner.invoke(cli, ["list-tasks", "--tasks-dir", str(tmp_path / "tasks")])

    assert result.exit_code == 0
    assert "AXBench Tasks" in result.output
    assert os.environ["HF_TOKEN"] == "from-dotenv"


def test_cli_does_not_override_existing_hf_token(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "already-set")

    runner = CliRunner()
    with patch(
        "axbench.cli.load_standard_task_bundle",
        return_value=StandardTaskBundle(tasks=[], warnings=[]),
    ), patch(
        "axbench.cli.load_performance_tasks",
        return_value=[],
    ):
        result = runner.invoke(cli, ["list-tasks", "--tasks-dir", str(tmp_path / "tasks")])

    assert result.exit_code == 0
    assert os.environ["HF_TOKEN"] == "already-set"


def test_run_persists_suite_metadata_and_warnings(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    task_file = tasks_dir / "general" / "code_gen" / "python" / "task.yaml"
    task_file.parent.mkdir(parents=True)
    task_file.write_text(
        "id: python_task_a\n"
        "evaluator: code_gen\n"
        "language: python\n"
        "difficulty: easy\n"
        "source: general\n"
        "prompt: test\n"
        "test_cases: []\n"
        "timeout_seconds: 10\n"
    )
    output_path = tmp_path / "result.json"

    runner = CliRunner()
    with patch(
        "axbench.cli.load_standard_task_bundle",
        return_value=StandardTaskBundle(tasks=[], warnings=["GPQA Diamond skipped: HF_TOKEN not set"]),
    ), patch("axbench.cli.load_performance_tasks", return_value=[]), patch(
        "axbench.cli.Runner.run_tasks",
        return_value=_sample_run("mock-model", task_id="python_task_a"),
    ):
        result = runner.invoke(
            cli,
            [
                "run",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "mock-model",
                "--task",
                "python_task_a",
                "--tasks-dir",
                str(tasks_dir),
                "--save",
                str(output_path),
            ],
        )

    assert result.exit_code == 0
    saved = BenchmarkRun.load(output_path)
    assert saved.metadata.benchmark_suite_version == "axbench-v2"
    assert saved.metadata.quick_mode is False
    assert saved.metadata.warnings == ["GPQA Diamond skipped: HF_TOKEN not set"]


def test_run_quick_filters_to_quick_subset_and_marks_metadata(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    python_dir = tasks_dir / "general" / "code_gen" / "python"
    python_dir.mkdir(parents=True)
    (python_dir / "async_queue.yaml").write_text(
        "id: python_async_queue\n"
        "evaluator: code_gen\n"
        "language: python\n"
        "difficulty: hard\n"
        "source: general\n"
        "prompt: test\n"
        "test_cases: []\n"
        "timeout_seconds: 10\n"
    )
    (python_dir / "binary_search.yaml").write_text(
        "id: python_binary_search\n"
        "evaluator: code_gen\n"
        "language: python\n"
        "difficulty: easy\n"
        "source: general\n"
        "prompt: test\n"
        "test_cases: []\n"
        "timeout_seconds: 10\n"
    )
    output_path = tmp_path / "quick.json"

    runner = CliRunner()

    def fake_run_tasks(task_defs, **_kwargs):
        task_ids = [task["id"] for task in task_defs]
        assert task_ids == ["python_async_queue", "HumanEval/0"]
        return BenchmarkRun(
            metadata=RunMetadata(
                model="mock-model",
                base_url="http://localhost:8000/v1",
                timestamp="2026-04-14T12:00:00+00:00",
                axbench_version="0.1.0",
                duration_seconds=1.5,
            ),
            tasks=[
                TaskResult(
                    task_id=task_id,
                    evaluator="standard" if task_id == "HumanEval/0" else "code_gen",
                    pillar="standard" if task_id == "HumanEval/0" else "general_coding",
                    source="standard/humaneval" if task_id == "HumanEval/0" else "general",
                    language="python",
                    difficulty="hard",
                    passed=True,
                    score=1.0,
                    raw_output="ok",
                    extracted_code="ok",
                    test_results=[],
                    error=None,
                    latency_ms=10.0,
                )
                for task_id in task_ids
            ],
        )

    perf_result = TaskResult(
        task_id="performance_llama_benchy",
        evaluator="perf",
        pillar="performance",
        source="performance/llama-benchy",
        language="text",
        difficulty="benchmark",
        passed=True,
        score=0.0,
        raw_output="{}",
        extracted_code="",
        test_results=[],
        error=None,
        latency_ms=100.0,
    )

    with patch(
        "axbench.cli.load_standard_task_bundle",
        return_value=StandardTaskBundle(tasks=[_sample_standard_tasks()[1]], warnings=[]),
    ), patch(
        "axbench.cli.load_performance_tasks",
        return_value=_sample_perf_tasks(),
    ), patch(
        "axbench.cli.Runner.run_tasks",
        side_effect=fake_run_tasks,
    ), patch(
        "axbench.cli._run_perf_task",
        return_value=perf_result,
    ):
        result = runner.invoke(
            cli,
            [
                "run",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "mock-model",
                "--quick",
                "--tasks-dir",
                str(tasks_dir),
                "--save",
                str(output_path),
            ],
        )

    assert result.exit_code == 0
    assert "[QUICK MODE]" in result.output
    saved = BenchmarkRun.load(output_path)
    assert saved.metadata.quick_mode is True
    assert saved.selected_task_ids == [
        "python_async_queue",
        "performance_llama_benchy",
        "HumanEval/0",
    ]
    assert saved.skipped_task_ids == ["python_binary_search"]
