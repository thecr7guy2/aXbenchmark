import json
import subprocess
from unittest.mock import MagicMock, patch

from axbench.evaluators.perf import PerfEvaluator, PerfResult


def test_perf_result_parses_llama_benchy_nested_json():
    sample = {
        "benchmarks": [
            {
                "concurrency": 1,
                "context_size": 0,
                "prompt_size": 2048,
                "response_size": 32,
                "pp_throughput": {"mean": 8521.0},
                "tg_throughput": {"mean": 73.18},
                "peak_throughput": {"mean": 75.84},
                "ttfr": {"mean": 297.0},
            }
        ]
    }
    result = PerfResult.from_llama_benchy(sample)
    assert result.pp_tokens_per_sec == 8521.0
    assert result.tg_tokens_per_sec == 73.18
    assert result.peak_tg_tokens_per_sec == 75.84
    assert result.ttft_ms == 297.0


def test_perf_result_parses_legacy_flat_json():
    sample = {
        "runs": [
            {
                "test": "pp2048",
                "tokens_per_second_mean": 5000.0,
                "ttfr_mean": 400.0,
                "est_ppt_mean": 350.0,
            },
            {
                "test": "tg32",
                "tokens_per_second_mean": 60.0,
                "peak_tokens_per_second_mean": 65.0,
            },
        ]
    }
    result = PerfResult.from_llama_benchy(sample)
    assert result.pp_tokens_per_sec == 5000.0
    assert result.tg_tokens_per_sec == 60.0
    assert result.peak_tg_tokens_per_sec == 65.0
    assert result.ttft_ms == 400.0


def test_perf_evaluator_calls_llama_benchy():
    ev = PerfEvaluator(llama_benchy_dir="/home/msai/vllm/llama-benchy")
    sample_output = json.dumps(
        {
            "benchmarks": [
                {
                    "concurrency": 1,
                    "context_size": 0,
                    "prompt_size": 2048,
                    "response_size": 32,
                    "pp_throughput": {"mean": 5000.0},
                    "tg_throughput": {"mean": 60.0},
                    "peak_throughput": {"mean": 65.0},
                    "ttfr": {"mean": 400.0},
                }
            ]
        }
    )

    with patch.object(ev, "_run_process", return_value="llama-benchy output") as mock_run_process:
        with patch("pathlib.Path.read_text", return_value=sample_output):
            result = ev.run(
                base_url="http://localhost:8000/v1",
                model="test-model",
            )

    assert result.pp_tokens_per_sec == 5000.0
    cmd = mock_run_process.call_args.args[0]
    assert cmd[:3] == ["uv", "run", "llama-benchy"]
    assert "--base-url" in cmd
    assert "--model" in cmd
    assert "--save-result" in cmd


def test_perf_evaluator_uses_smaller_default_suite():
    assert PerfEvaluator.DEFAULT_CONFIG == [
        "--pp",
        "2048",
        "--tg",
        "32",
        "--depth",
        "0",
        "4096",
        "--latency-mode",
        "generation",
        "--runs",
        "2",
        "--skip-coherence",
    ]


def test_perf_evaluator_reports_timeout_with_recent_output():
    ev = PerfEvaluator(llama_benchy_dir="/home/msai/vllm/llama-benchy", timeout_seconds=5)

    with patch.object(
        ev,
        "_run_process",
        side_effect=ValueError("llama-benchy timed out after 5s. Recent output: Warming up... | Running test: pp=2048"),
    ):
        with patch("pathlib.Path.read_text", return_value=""):
            try:
                ev.run(
                    base_url="http://localhost:8000/v1",
                    model="test-model",
                )
            except ValueError as exc:
                message = str(exc)
            else:
                raise AssertionError("Expected ValueError")

    assert "timed out after 5s" in message
    assert "Warming up..." in message


def test_perf_evaluator_streams_progress_lines():
    ev = PerfEvaluator(llama_benchy_dir="/home/msai/vllm/llama-benchy", timeout_seconds=5)
    fake_stdout = MagicMock()
    fake_stdout.readline.side_effect = ["Warming up...\n", "Running test: pp=2048\n", ""]
    fake_stdout.read.return_value = ""

    process = MagicMock()
    process.stdout = fake_stdout
    process.poll.side_effect = [None, None, 0, 0]
    process.returncode = 0

    seen_lines = []
    with patch("subprocess.Popen", return_value=process), patch(
        "select.select",
        side_effect=[([fake_stdout], [], []), ([fake_stdout], [], []), ([], [], [])],
    ):
        output = ev._run_process(
            ["uv", "run", "llama-benchy"],
            progress_callback=lambda line: seen_lines.append(line),
            timeout_seconds=5,
        )

    assert "Warming up..." in output
    assert seen_lines == ["Warming up...", "Running test: pp=2048"]
