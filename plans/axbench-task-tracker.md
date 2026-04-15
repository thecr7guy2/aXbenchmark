# AXBench Implementation Task Tracker

**Plan file:** `axbench-plan.md` (full details, code, and steps for each task)
**Design file:** `axbench-design.md` (architecture and spec)
**Working directory:** `/home/msai/vllm/benchmarking/`

> When resuming after context clear: read this file first, then read `axbench-plan.md` for the next pending task's full steps and code.

---

## Task Status

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Project scaffold (pyproject.toml + dirs + `axbench --version`) | ✅ Done | |
| 2 | Core data models + results storage (`TaskResult`, `BenchmarkRun`, JSON save/load) | ✅ Done | |
| 3 | `LLMClient` — OpenAI-compatible API wrapper (httpx) | ✅ Done | |
| 4 | Python + Bash sandbox (subprocess, timeout, pass/fail) | ✅ Done | |
| 5 | C++ sandbox (g++ compile + run, test harness) | ✅ Done | |
| 6 | Code extraction utility (`extract_code()`, markdown fence handling) | ✅ Done | |
| 7 | Evaluator registry + stub files (`get_evaluator()`, lazy imports) | ✅ Done | Includes evaluator stubs and `tool_call.py` placeholder |
| 8 | `CodeGenEvaluator` (Python/C++/bash, build_prompt + evaluate) | ✅ Done | Prompt builder, extraction, sandbox scoring, partial-pass scoring |
| 9 | `BugFixEvaluator` (reuses CodeGenEvaluator scoring) | ✅ Done | Reuses code-gen scoring path and overrides evaluator label |
| 10 | Python code_gen seed tasks — 5 YAML files (easy→hard) | ✅ Done | binary_search, string_reverse, lru_cache, decorator_retry, async_queue |
| 11 | Python bug_fix seed tasks — 4 YAML files (easy→hard) | ✅ Done | off_by_one, mutable_default, exception_handling, broken_generator |
| 12 | C++ + Bash seed tasks — 7 YAML files | ✅ Done | string_tokenizer, matrix_multiply, thread_safe_queue, memory_leak, dangling_pointer, log_rotation, docker_health |
| 13 | `StandardEvaluator` (Pillar 1 — MMLU + GPQA Diamond + HumanEval + MBPP + LiveCodeBench) | ✅ Done | Unified evaluator with prompt/eval dispatch across all five benchmarks |
| 14 | `PerfEvaluator` (Pillar 2 — llama-benchy subprocess wrapper) | ✅ Done | Wraps local llama-benchy CLI and parses nested `benchmarks` JSON with legacy fallback |
| 15 | `TaskLoader` (YAML auto-discovery + filters by language/difficulty/source) | ✅ Done | Includes evaluator/source/tag/pillar filters and metadata listing |
| 16 | `Runner` orchestrator (calls model, dispatches evaluators, rich progress bar) | ✅ Done | Orchestrates prompt build, model call, evaluator scoring, latency propagation, and run metadata |
| 17 | `axbench run` CLI command (--pillar, --language, --difficulty, --task, --save) | ✅ Done | CLI wired to loader, runner, scorecard printing, and JSON save |
| 18 | `axbench compare` + `axbench list-tasks` CLI commands | ✅ Done | Added side-by-side comparison output and task catalog filtering |
| 19 | Team dirs + `tool_call.py` stub + README | ✅ Done | Added contribution guide, polished tool-calling stub, and top-level README |

---

## Status Key
- ⬜ Pending — not started
- 🔄 In Progress — currently being worked on
- ✅ Done — implemented and tests passing
- ❌ Blocked — needs attention

---

## Key Context (for resuming)

### Infrastructure
- **Main model endpoint:** `http://10.1.115.4:8000/v1` (2x DGX Spark, Minimax 2.5 AWQ-4Bit)
- **llama-benchy:** `/home/msai/vllm/llama-benchy/` (used by Task 14 PerfEvaluator)
- **Package manager:** `uv`
- **Python:** 3.11+

### Project structure being built
```
/home/msai/vllm/benchmarking/
├── axbench/                  ← main package
│   ├── cli.py
│   ├── runner.py
│   ├── client.py
│   ├── sandbox.py
│   ├── extractor.py
│   ├── loader.py
│   ├── results.py
│   └── evaluators/
│       ├── base.py
│       ├── code_gen.py
│       ├── bug_fix.py
│       ├── standard.py
│       ├── perf.py
│       └── tool_call.py     ← stub only
├── tasks/
│   ├── general/code_gen/{python,cpp,bash}/
│   ├── general/bug_fix/{python,cpp}/
│   ├── team/{riccardo,tom,serge_mykyta}/
│   ├── standard/cache/
│   └── tool_call/
├── tests/
├── results/
└── pyproject.toml
```

### Test command
```bash
cd /home/msai/vllm/benchmarking && uv run pytest tests/ -v
```

### Smoke test (requires live endpoint)
```bash
uv run axbench run --base-url http://10.1.115.4:8000/v1 --model minimax-m2.5-awq --task python_binary_search
```
