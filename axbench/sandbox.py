import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    passed: bool
    actual: object
    error: str | None
    stdout: str
    stderr: str


class Sandbox:
    def run_python(
        self,
        code: str,
        test_expression: str,
        expected: object,
        timeout: int = 10,
    ) -> SandboxResult:
        evaluation_block = self._build_python_evaluation_block(test_expression)
        script = textwrap.dedent(f"""
{code}

import sys
try:
{textwrap.indent(evaluation_block, "    ")}
    _expected = {repr(expected)}
    if _actual == _expected:
        print("AXBENCH_PASS")
        print(repr(_actual))
    else:
        print("AXBENCH_FAIL")
        print(repr(_actual))
except Exception as e:
    print("AXBENCH_ERROR")
    print(str(e))
""").strip()

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            tmp = f.name

        try:
            proc = subprocess.run(
                ["python3", tmp],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout.strip()
            lines = stdout.splitlines()
            if not lines:
                return SandboxResult(False, None, proc.stderr or "No output", stdout, proc.stderr)
            status = lines[0]
            actual_repr = lines[1] if len(lines) > 1 else ""
            if status == "AXBENCH_PASS":
                return SandboxResult(True, actual_repr, None, stdout, proc.stderr)
            elif status == "AXBENCH_FAIL":
                return SandboxResult(False, actual_repr, None, stdout, proc.stderr)
            else:
                return SandboxResult(False, None, actual_repr or proc.stderr, stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            return SandboxResult(False, None, f"Timeout after {timeout}s", "", "")
        finally:
            Path(tmp).unlink(missing_ok=True)

    def _build_python_evaluation_block(self, test_expression: str) -> str:
        lines = textwrap.dedent(test_expression).splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        if not lines:
            return "_actual = None"
        if len(lines) == 1:
            return f"_actual = {lines[0].strip()}"

        setup = "\n".join(lines[:-1])
        final_expression = lines[-1].strip()
        return f"{setup}\n_actual = {final_expression}"

    def run_bash(
        self,
        script: str,
        expected_stdout: str,
        timeout: int = 10,
        setup_script: str = "",
        expected_exit_code: int = 0,
        post_check_script: str = "",
        allow_stderr: bool = False,
    ) -> SandboxResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "generated.sh"
            script_path.write_text(script)

            wrapper = textwrap.dedent(
                f"""
                {setup_script}
                bash generated.sh
                _axbench_exit=$?
                {post_check_script}
                exit $_axbench_exit
                """
            ).strip()

            try:
                proc = subprocess.run(
                    ["bash", "-lc", wrapper],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                stderr_ok = allow_stderr or not proc.stderr
                passed = (
                    proc.stdout == expected_stdout
                    and proc.returncode == expected_exit_code
                    and stderr_ok
                )

                error_parts = []
                if proc.returncode != expected_exit_code:
                    error_parts.append(
                        f"Exit code {proc.returncode} did not match expected {expected_exit_code}"
                    )
                if not stderr_ok and proc.stderr:
                    error_parts.append(proc.stderr)
                error = "\n".join(part.strip() for part in error_parts if part.strip()) or None
                return SandboxResult(passed, proc.stdout, error, proc.stdout, proc.stderr)
            except subprocess.TimeoutExpired:
                return SandboxResult(False, None, f"Timeout after {timeout}s", "", "")

    def run_cpp(self, harness_code: str, timeout: int = 15) -> SandboxResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "bench.cpp"
            binary = Path(tmpdir) / "bench"
            src.write_text(harness_code)

            compile_proc = subprocess.run(
                ["g++", "-std=c++17", "-o", str(binary), str(src)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if compile_proc.returncode != 0:
                return SandboxResult(False, None, compile_proc.stderr, "", compile_proc.stderr)

            try:
                run_proc = subprocess.run(
                    [str(binary)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                passed = run_proc.returncode == 0 and "PASS" in run_proc.stdout
                error = run_proc.stderr if run_proc.returncode != 0 else None
                return SandboxResult(passed, run_proc.stdout.strip(), error, run_proc.stdout, run_proc.stderr)
            except subprocess.TimeoutExpired:
                return SandboxResult(False, None, f"Timeout after {timeout}s", "", "")
