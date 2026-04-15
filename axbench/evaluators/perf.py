import json
import select
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class PerfResult:
    pp_tokens_per_sec: float
    tg_tokens_per_sec: float
    peak_tg_tokens_per_sec: float
    ttft_ms: float
    raw: dict[str, Any]
    selected_benchmark: dict[str, Any] | None = None

    @classmethod
    def from_llama_benchy(cls, data: dict[str, Any]) -> "PerfResult":
        if "benchmarks" in data:
            selected = cls._select_benchmark(data["benchmarks"])
            return cls(
                pp_tokens_per_sec=cls._metric_mean(selected, "pp_throughput"),
                tg_tokens_per_sec=cls._metric_mean(selected, "tg_throughput"),
                peak_tg_tokens_per_sec=cls._metric_mean(selected, "peak_throughput"),
                ttft_ms=cls._metric_mean(selected, "ttfr"),
                raw=data,
                selected_benchmark=selected,
            )

        if "runs" in data:
            pp_tps = 0.0
            tg_tps = 0.0
            peak_tg = 0.0
            ttft = 0.0
            for run in data.get("runs", []):
                test_name = str(run.get("test", ""))
                if test_name.startswith("pp"):
                    pp_tps = float(run.get("tokens_per_second_mean", 0.0))
                    ttft = float(run.get("ttfr_mean", 0.0))
                elif test_name.startswith("tg"):
                    tg_tps = float(run.get("tokens_per_second_mean", 0.0))
                    peak_tg = float(run.get("peak_tokens_per_second_mean", 0.0))
            return cls(
                pp_tokens_per_sec=pp_tps,
                tg_tokens_per_sec=tg_tps,
                peak_tg_tokens_per_sec=peak_tg,
                ttft_ms=ttft,
                raw=data,
                selected_benchmark=None,
            )

        raise ValueError("Unsupported llama-benchy result format")

    @staticmethod
    def _metric_mean(benchmark: dict[str, Any], key: str) -> float:
        metric = benchmark.get(key)
        if isinstance(metric, dict):
            return float(metric.get("mean", 0.0))
        return 0.0

    @classmethod
    def _select_benchmark(cls, benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
        if not benchmarks:
            raise ValueError("llama-benchy result contains no benchmarks")

        def sort_key(benchmark: dict[str, Any]) -> tuple[float, float, float, float]:
            concurrency = int(benchmark.get("concurrency", 1))
            context_size = int(benchmark.get("context_size", 0))
            prompt_size = int(benchmark.get("prompt_size", 0))
            response_size = int(benchmark.get("response_size", 0))

            # Prefer a simple baseline run for scorecard reporting.
            return (
                abs(concurrency - 1),
                abs(context_size - 0),
                abs(prompt_size - 2048),
                abs(response_size - 32),
            )

        return min(benchmarks, key=sort_key)


class PerfEvaluator:
    DEFAULT_CONFIG = [
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

    def __init__(
        self,
        llama_benchy_dir: str = "/home/msai/vllm/llama-benchy",
        timeout_seconds: int = 900,
    ):
        self.llama_benchy_dir = Path(llama_benchy_dir)
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        base_url: str,
        model: str,
        config: list[str] | None = None,
        api_key: str = "EMPTY",
        progress_callback: Callable[[str], None] | None = None,
        timeout_seconds: int | None = None,
    ) -> PerfResult:
        if not self.llama_benchy_dir.exists():
            raise FileNotFoundError(
                f"llama-benchy directory not found: {self.llama_benchy_dir}"
            )

        extra_args = config or self.DEFAULT_CONFIG
        effective_timeout = timeout_seconds or self.timeout_seconds
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            result_path = Path(handle.name)

        cmd = [
            "uv",
            "run",
            "llama-benchy",
            "--base-url",
            base_url,
            "--api-key",
            api_key,
            "--model",
            model,
            "--format",
            "json",
            "--save-result",
            str(result_path),
            *extra_args,
        ]

        try:
            output = self._run_process(
                cmd,
                progress_callback=progress_callback,
                timeout_seconds=effective_timeout,
            )
            if not result_path.exists() or not result_path.read_text().strip():
                raise ValueError(
                    "llama-benchy completed without writing a result file. "
                    f"Recent output: {self._tail_output(output)}"
                )
            data = json.loads(result_path.read_text())
            return PerfResult.from_llama_benchy(data)
        finally:
            result_path.unlink(missing_ok=True)

    def _run_process(
        self,
        cmd: list[str],
        progress_callback: Callable[[str], None] | None,
        timeout_seconds: int,
    ) -> str:
        process = subprocess.Popen(
            cmd,
            cwd=str(self.llama_benchy_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None

        output_lines: list[str] = []
        start = time.monotonic()

        try:
            while True:
                if time.monotonic() - start > timeout_seconds:
                    process.kill()
                    remainder = process.stdout.read() or ""
                    if remainder:
                        output_lines.append(remainder)
                    raise ValueError(
                        "llama-benchy timed out after "
                        f"{timeout_seconds}s. Recent output: {self._tail_output(''.join(output_lines))}"
                    )

                ready, _, _ = select.select([process.stdout], [], [], 0.5)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line)
                        if progress_callback is not None:
                            progress_callback(line.rstrip())

                if process.poll() is not None:
                    remainder = process.stdout.read() or ""
                    if remainder:
                        output_lines.append(remainder)
                        if progress_callback is not None:
                            for line in remainder.splitlines():
                                progress_callback(line.rstrip())
                    break

            if process.returncode != 0:
                raise ValueError(
                    "llama-benchy failed with exit code "
                    f"{process.returncode}. Recent output: {self._tail_output(''.join(output_lines))}"
                )
            return "".join(output_lines)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    @staticmethod
    def _tail_output(output: str, max_lines: int = 8) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return "<no output>"
        return " | ".join(lines[-max_lines:])
