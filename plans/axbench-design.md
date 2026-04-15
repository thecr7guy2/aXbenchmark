# AXBench — Comprehensive LLM Benchmarking System for AX-Office.ai

**Author:** Maniraj Sai | AI Engineer, aXite Security Tools
**Date:** 2026-04-14
**Status:** Design

---

## 1. Problem

AX-Office.ai swaps its main LLM regularly as better open-weight models become available. There is no systematic way to compare models across the dimensions that matter: output quality, inference speed, and real-world task performance. The existing tool (`llama-benchy`) covers inference speed but nothing else.

When a new model candidate arrives, the evaluation process should be: run `axbench`, get a comprehensive scorecard, compare against the current model, make a data-driven decision.

## 2. The Five Pillars of AXBench

AXBench evaluates models across five distinct categories. Each pillar is a separate evaluator plugin, independently runnable, and results are combined into a unified scorecard.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AXBench Scorecard                            │
├──────────────┬──────────────┬──────────────┬──────────────┬────────────┤
│  1. Standard │ 2. Perf/Speed│  3. General  │ 4. Team Real │ 5. Tool   │
│  Benchmarks  │ (llama-bench)│  Coding      │   World      │  Calling  │
│              │              │              │              │            │
│  MMLU        │  pp t/s      │  Easy/Med/   │  Riccardo    │  (Future) │
│  GPQA        │  tg t/s      │  Hard tasks  │  Tom         │            │
│  HumanEval   │  TTFT        │  Python/C++/ │  Serge       │            │
│  MBPP        │  Latency     │  Bash        │  Mykyta      │            │
│  LiveCodeBench│             │              │              │            │
│              │              │              │              │            │
│  BUILD NOW   │  BUILD NOW   │  BUILD NOW   │  BUILD NOW   │  PLANNED  │
└──────────────┴──────────────┴──────────────┴──────────────┴────────────┘
```

### Pillar 1: Standard Benchmarks
Public benchmarks for baseline comparison against leaderboards.
- **MMLU** — Massive Multitask Language Understanding (general knowledge/reasoning)
- **GPQA Diamond** — hard, expert-written science QA that complements MMLU with a much stronger reasoning signal
- **HumanEval** — Python function completion (164 tasks, pass@1)
- **MBPP** — Mostly Basic Python Programming (500 tasks, pass@1)
- **LiveCodeBench** — fresher, contamination-resistant coding evaluation beyond classic HumanEval/MBPP

### Pillar 2: Performance Benchmarks (llama-benchy Integration)
Inference speed and throughput. Wraps the existing `llama-benchy` tool.
- Prompt processing speed (tokens/s) at various context depths
- Token generation speed (tokens/s)
- Time To First Token (TTFT)
- Concurrency scaling

### Pillar 3: General Coding Tasks
AI-generated coding tasks across difficulty levels. These are generic tasks that test fundamental model capabilities — not tied to any specific team member's workflow.
- Python: algorithms, data structures, standard library usage
- C++: memory management, templates, STL, OOP patterns
- Bash: scripting, text processing, system automation
- Spread: easy / medium / hard

### Pillar 4: Team Real-World Tasks
Tasks contributed by team members from their actual day-to-day work. These are the most valuable benchmarks because they reflect what the model will actually be used for.
- **Riccardo** — infra/bash scripts (Docker, logging, monitoring)
- **Tom** — C++, MariaDB, networking
- **Serge & Mykyta** — C++/Python software development

### Pillar 5: Tool Calling (Future — Not Built Now)
Evaluate the model's ability to use tools correctly via the OpenAI tool/function calling API.
- Does it call the right tool for the task?
- Does it pass correct arguments?
- Does it avoid unnecessary tool calls?
- Does it chain tool calls correctly?

This pillar is **in the architecture and plan** but not implemented in the first version.

## 3. Non-Goals

- Meeting transcription evaluation
- ISMS/RAG evaluation (may revisit later)
- LLM-as-judge scoring (no external API calls, no data leaving the building)
- Web dashboard or database — CLI tool with file-based results
- Multi-turn or agentic evaluation (single-turn only, except tool calling in the future)

## 4. Architecture

### 4.1 Overview

Plugin-based architecture. Each pillar is an evaluator plugin.

```
CLI (cli.py)
  │
  ├── run         ── run all or specific pillars
  ├── compare     ── side-by-side model comparison
  ├── list-tasks  ── show available tasks
  └── perf        ── run llama-benchy integration
       │
       ▼
Runner (runner.py) ── loads tasks, dispatches to evaluators, collects results
  │
  ├── Client (client.py) ── OpenAI-compatible /v1/chat/completions
  │
  ├── Evaluators (evaluators/)
  │   ├── code_gen.py    ── code generation (Pillar 3 + 4)
  │   ├── bug_fix.py     ── bug fixing (Pillar 3 + 4)
  │   ├── standard.py    ── MMLU/GPQA/HumanEval/MBPP/LiveCodeBench (Pillar 1)
  │   ├── perf.py        ── llama-benchy wrapper (Pillar 2)
  │   └── tool_call.py   ── tool calling (Pillar 5 — future)
  │
  ├── Sandbox (sandbox.py) ── code execution isolation
  │
  └── Results (results.py) ── storage, loading, comparison
```

### 4.2 Project Structure

```
benchmarking/
├── axbench/                        # Main package
│   ├── __init__.py
│   ├── cli.py                      # CLI entry point
│   ├── runner.py                   # Core orchestrator
│   ├── client.py                   # OpenAI-compatible API client
│   ├── sandbox.py                  # Code execution sandbox
│   ├── results.py                  # Result models, storage, comparison
│   └── evaluators/                 # Plugin evaluators
│       ├── __init__.py             # Evaluator registry and discovery
│       ├── base.py                 # Abstract base evaluator
│       ├── code_gen.py             # Code generation evaluator
│       ├── bug_fix.py              # Bug fixing evaluator
│       ├── standard.py             # MMLU/GPQA/HumanEval/MBPP/LiveCodeBench evaluator
│       ├── perf.py                 # llama-benchy wrapper
│       └── tool_call.py            # Tool calling evaluator (future)
├── tasks/                          # Task definitions (YAML)
│   ├── general/                    # Pillar 3: General coding tasks
│   │   ├── code_gen/
│   │   │   ├── python/
│   │   │   ├── cpp/
│   │   │   └── bash/
│   │   └── bug_fix/
│   │       ├── python/
│   │       └── cpp/
│   ├── team/                       # Pillar 4: Team-contributed tasks
│   │   ├── riccardo/               # Infra/bash
│   │   ├── tom/                    # C++/DB/networking
│   │   └── serge_mykyta/           # C++/Python software
│   ├── standard/                   # Pillar 1: Standard benchmarks
│   │   └── cache/                  # Downloaded datasets
│   └── tool_call/                  # Pillar 5: Tool calling (future)
├── results/                        # Benchmark result files (JSON)
├── pyproject.toml
└── README.md
```

## 5. Component Design

### 5.1 Task Format (YAML)

Every task is a self-contained YAML file. This is the contract between task authors and evaluators.

#### Code Generation Task (Python)

```yaml
id: python_binary_search
evaluator: code_gen
language: python
difficulty: medium
source: general                     # or "team/riccardo", "team/tom", etc.
tags: [algorithms, search]

prompt: |
  Write a Python function called `binary_search` that takes a sorted list
  of integers and a target integer. Return the index of the target if found,
  or -1 if not found.

function_signature: "def binary_search(arr: list[int], target: int) -> int:"

test_cases:
  - input: "binary_search([1, 3, 5, 7, 9], 5)"
    expected: 2
  - input: "binary_search([1, 3, 5, 7, 9], 4)"
    expected: -1
  - input: "binary_search([], 1)"
    expected: -1
  - input: "binary_search([1], 1)"
    expected: 0

timeout_seconds: 10
```

#### Bug Fix Task

```yaml
id: python_off_by_one
evaluator: bug_fix
language: python
difficulty: easy
source: general
tags: [bugs, loops]

description: |
  The following function should return the sum of all elements in a list,
  but it has a bug. Fix it.

buggy_code: |
  def sum_list(numbers):
      total = 0
      for i in range(1, len(numbers)):
          total += numbers[i]
      return total

prompt: |
  The following Python function has a bug. Find and fix it.
  Return only the corrected function.

  ```python
  def sum_list(numbers):
      total = 0
      for i in range(1, len(numbers)):
          total += numbers[i]
      return total
  ```

test_cases:
  - input: "sum_list([1, 2, 3])"
    expected: 6
  - input: "sum_list([10])"
    expected: 10
  - input: "sum_list([])"
    expected: 0

timeout_seconds: 10
```

#### C++ Task

```yaml
id: cpp_matrix_multiply
evaluator: code_gen
language: cpp
difficulty: medium
source: general
tags: [algorithms, linear-algebra]

prompt: |
  Write a C++ function that multiplies two matrices represented as
  vector<vector<int>>. Return the result matrix.

  ```cpp
  vector<vector<int>> matrix_multiply(const vector<vector<int>>& a,
                                       const vector<vector<int>>& b);
  ```

# For C++, test cases are compiled and run as a complete program
test_harness: |
  #include <vector>
  #include <iostream>
  #include <cassert>
  using namespace std;

  {{GENERATED_CODE}}

  int main() {
      vector<vector<int>> a = {{1, 2}, {3, 4}};
      vector<vector<int>> b = {{5, 6}, {7, 8}};
      auto result = matrix_multiply(a, b);
      assert(result[0][0] == 19);
      assert(result[0][1] == 22);
      assert(result[1][0] == 43);
      assert(result[1][1] == 50);
      cout << "PASS" << endl;
      return 0;
  }

timeout_seconds: 15
```

#### Tool Calling Task (Future — Pillar 5)

```yaml
id: tool_weather_lookup
evaluator: tool_call
difficulty: easy
source: general
tags: [tool-calling, single-tool]

system_prompt: |
  You are a helpful assistant with access to tools.

prompt: |
  What's the weather like in Amsterdam today?

available_tools:
  - name: get_weather
    description: Get current weather for a city
    parameters:
      type: object
      properties:
        city: { type: string, description: "City name" }
        units: { type: string, enum: [celsius, fahrenheit], default: celsius }
      required: [city]

  - name: search_web
    description: Search the web for information
    parameters:
      type: object
      properties:
        query: { type: string }
      required: [query]

expected_tool_calls:
  - tool: get_weather
    arguments: { city: "Amsterdam" }

should_not_call: [search_web]

scoring:
  correct_tool: 0.4           # Called the right tool
  correct_arguments: 0.4      # Passed correct args
  no_unnecessary_calls: 0.2   # Didn't call tools it shouldn't have
```

### 5.2 Evaluator Interface

```python
# axbench/evaluators/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TaskResult:
    task_id: str
    evaluator: str
    language: str
    passed: bool
    score: float          # 0.0 to 1.0
    raw_output: str       # Model's raw response
    extracted_code: str   # Parsed code from response
    test_results: list    # Per-test pass/fail details
    error: str | None     # Error message if execution failed
    latency_ms: float     # Time to get response from model
    pillar: str           # Which pillar this belongs to
    source: str           # "general", "team/riccardo", "standard", etc.

class BaseEvaluator(ABC):
    """All evaluators implement this interface."""

    @abstractmethod
    def evaluate(self, task: dict, model_output: str) -> TaskResult:
        """Score a single model output against a task definition."""
        ...

    @abstractmethod
    def build_prompt(self, task: dict) -> list[dict]:
        """Convert task definition into chat messages for the API."""
        ...
```

Each evaluator:
1. Builds the prompt from the task YAML
2. Receives the raw model output
3. Extracts code from the response (handles markdown fences, etc.)
4. Runs it through the sandbox
5. Returns a `TaskResult`

### 5.3 Code Extraction

Models wrap code in markdown fences, sometimes add explanations, sometimes don't. The code extractor handles this:

1. Look for fenced code blocks (```python ... ``` or ```cpp ... ```)
2. If multiple blocks, prefer the longest one or the one matching the expected language
3. If no fences, try to extract the function/class directly by pattern matching the expected signature
4. If all else fails, treat the entire output as code

### 5.4 Sandbox (Code Execution)

All generated code runs in isolated subprocesses with:

- **Timeout enforcement** — per-task configurable, default 10s
- **No network access** — code cannot make outbound calls
- **Resource limits** — memory cap via `resource.setrlimit`
- **Temp directory** — each execution gets its own temp dir, cleaned up after

For Python tasks:
- Inject generated code + test case into a temp `.py` file
- Run via `subprocess.run(["python", temp_file], timeout=...)`
- Check exit code and stdout for pass/fail

For C++ tasks:
- Inject generated code into the test harness template (replacing `{{GENERATED_CODE}}`)
- Compile with `g++ -o temp_binary temp_file.cpp`
- Run the compiled binary
- Check for "PASS" in stdout

For bash tasks:
- Write script to temp file
- Run via `subprocess.run(["bash", temp_file], timeout=...)`
- Check expected output

### 5.5 Standard Benchmarks (MMLU / GPQA Diamond / HumanEval / MBPP / LiveCodeBench) — Pillar 1

#### MMLU
- Downloads the MMLU dataset (one-time, cached)
- Runs a configurable subset (default: 200 random questions across all categories for speed, or full 14k for thorough evaluation)
- Multiple-choice format — model picks A/B/C/D, scored by exact match
- Reports accuracy overall and per-category (STEM, humanities, social sciences, other)

#### GPQA Diamond
- Downloads the GPQA Diamond split when access is available
- Runs a configurable subset of expert-written graduate-level science questions
- Multiple-choice format — model picks A/B/C/D, scored by exact match
- Complements MMLU with a much harder reasoning signal

#### HumanEval
- 164 Python function completion tasks
- Model receives docstring + function signature, generates the body
- Executed against provided test cases
- Reports pass@1

#### MBPP
- 500 Python programming tasks
- Model receives a natural language description + 3 test cases as examples
- Generates a function, executed against held-out test cases
- Reports pass@1

#### LiveCodeBench
- Uses the code-generation track for fresher, contamination-resistant coding evaluation
- Supports prompt-based Python solution generation with executable test code
- Acts as the modern coding complement to HumanEval and MBPP

### 5.6 Performance Benchmarks (llama-benchy Integration) — Pillar 2

Wraps the existing `llama-benchy` tool rather than reimplementing performance benchmarking.

```python
# axbench/evaluators/perf.py

class PerfEvaluator:
    """Wraps llama-benchy for inference performance benchmarking."""

    def run(self, base_url: str, model: str, config: dict) -> PerfResult:
        """
        Shells out to llama-benchy with --format json, parses the output.

        Default config:
          --pp 512 2048 4096
          --tg 32 128
          --depth 0 4096 8192
          --latency-mode generation
          --runs 3
        """
        ...
```

The perf evaluator:
1. Calls `llama-benchy` as a subprocess with `--format json --save-result <tempfile>`
2. Parses the JSON output
3. Extracts key metrics: pp t/s, tg t/s, TTFT, peak t/s
4. Includes these in the axbench result file under the `performance` section

This means `axbench run` gives you BOTH quality and speed in one report.

### 5.7 API Client

Thin wrapper around the OpenAI-compatible chat completions endpoint:

```python
# axbench/client.py

class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY"):
        ...

    def generate(self, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 4096) -> tuple[str, float]:
        """Returns (response_text, latency_ms)"""
        ...

    def generate_with_tools(self, messages: list[dict], tools: list[dict],
                            temperature: float = 0.0) -> tuple[dict, float]:
        """For tool calling evaluation (Pillar 5). Returns (response_dict, latency_ms)"""
        ...
```

- Uses `temperature=0.0` by default for reproducibility
- Supports configurable `max_tokens`
- Returns latency alongside response for tracking
- `generate_with_tools` ready for Pillar 5

### 5.8 Results Format

Each benchmark run produces a comprehensive JSON file:

```json
{
  "metadata": {
    "model": "minimax-m2.5-awq",
    "base_url": "http://10.1.115.4:8000/v1",
    "timestamp": "2026-04-14T10:30:00",
    "axbench_version": "0.1.0",
    "duration_seconds": 420
  },
  "summary": {
    "overall_quality_score": 0.78,
    "by_pillar": {
      "standard": {
        "mmlu": { "accuracy": 0.72, "tasks": 200 },
        "gpqa": { "accuracy": 0.41, "tasks": 198 },
        "humaneval": { "pass_at_1": 0.75, "tasks": 164 },
        "mbpp": { "pass_at_1": 0.68, "tasks": 500 },
        "livecodebench": { "pass_at_1": 0.31, "tasks": 400 }
      },
      "performance": {
        "pp_tokens_per_sec": 8521.08,
        "tg_tokens_per_sec": 73.18,
        "ttft_ms": 340.65,
        "peak_tg_tokens_per_sec": 75.84
      },
      "general_coding": {
        "code_gen": { "total": 15, "passed": 12, "score": 0.80 },
        "bug_fix": { "total": 8, "passed": 6, "score": 0.75 }
      },
      "team_real_world": {
        "riccardo": { "total": 6, "passed": 5, "score": 0.83 },
        "tom": { "total": 6, "passed": 4, "score": 0.67 },
        "serge_mykyta": { "total": 6, "passed": 5, "score": 0.83 }
      },
      "tool_calling": null
    },
    "by_language": {
      "python": { "total": 20, "passed": 16, "score": 0.80 },
      "cpp": { "total": 12, "passed": 9, "score": 0.75 },
      "bash": { "total": 6, "passed": 5, "score": 0.83 },
      "sql": { "total": 3, "passed": 2, "score": 0.67 }
    },
    "by_difficulty": {
      "easy": { "total": 14, "passed": 13, "score": 0.93 },
      "medium": { "total": 15, "passed": 11, "score": 0.73 },
      "hard": { "total": 12, "passed": 7, "score": 0.58 }
    }
  },
  "tasks": [
    {
      "task_id": "python_binary_search",
      "evaluator": "code_gen",
      "pillar": "general_coding",
      "source": "general",
      "language": "python",
      "difficulty": "medium",
      "passed": true,
      "score": 1.0,
      "test_results": [
        { "input": "binary_search([1,3,5,7,9], 5)", "expected": 2, "actual": 2, "passed": true }
      ],
      "latency_ms": 1250.3,
      "error": null
    }
  ]
}
```

### 5.9 Comparison Mode

The `compare` command loads two result files and produces a side-by-side report:

```
$ axbench compare results/minimax-2.5.json results/qwen3.5-35b.json

================================================================================
                           AXBench Model Comparison
================================================================================
Model A: minimax-m2.5-awq (2026-04-14)
Model B: Qwen3.5-35B-A3B-FP8 (2026-04-10)
================================================================================

PILLAR 1 — Standard Benchmarks
  MMLU:              A: 72.0%         B: 68.5%         (+3.5%)
  GPQA Diamond:      A: 41.0%         B: 37.5%         (+3.5%)
  HumanEval:         A: 75.0%         B: 70.1%         (+4.9%)
  MBPP:              A: 68.0%         B: 65.2%         (+2.8%)
  LiveCodeBench:     A: 31.0%         B: 28.0%         (+3.0%)

PILLAR 2 — Performance (llama-benchy)
  PP t/s:            A: 8521          B: 6200           (+37.4%)
  TG t/s:            A: 73.2          B: 65.8           (+11.2%)
  TTFT (ms):         A: 340           B: 420            (-19.0% faster)

PILLAR 3 — General Coding
  Code Gen:          A: 80.0%         B: 73.3%         (+6.7%)
  Bug Fix:           A: 75.0%         B: 62.5%         (+12.5%)

PILLAR 4 — Team Real-World
  Riccardo (bash):   A: 83.3%         B: 66.7%         (+16.6%)
  Tom (C++/DB):      A: 66.7%         B: 66.7%         ( 0.0%)
  Serge/Mykyta:      A: 83.3%         B: 75.0%         (+8.3%)

PILLAR 5 — Tool Calling
  (not evaluated)

By Difficulty:
  Easy:              A: 93.0%         B: 86.0%         (+7.0%)
  Medium:            A: 73.0%         B: 67.0%         (+6.0%)
  Hard:              A: 58.0%         B: 50.0%         (+8.0%)

Task Deltas (only showing changes):
  [+] python_linked_list_reverse        FAIL -> PASS
  [+] cpp_memory_leak                   FAIL -> PASS
  [+] bash_docker_healthcheck           FAIL -> PASS
  [-] python_decorator_cache            PASS -> FAIL
================================================================================
  RECOMMENDATION: Model A outperforms on 3/4 active pillars.
================================================================================
```

## 6. CLI Interface

```bash
# ── Full benchmark (all pillars) ──
axbench run \
  --base-url http://10.1.115.4:8000/v1 \
  --model minimax-m2.5-awq \
  --save results/minimax-2.5.json

# ── Run specific pillars only ──
axbench run \
  --base-url http://10.1.115.4:8000/v1 \
  --model minimax-m2.5-awq \
  --pillar standard general_coding \
  --save results/minimax-2.5-quality-only.json

# ── Run only team tasks ──
axbench run \
  --base-url http://10.1.115.4:8000/v1 \
  --model minimax-m2.5-awq \
  --pillar team_real_world \
  --save results/minimax-2.5-team.json

# ── Run only performance ──
axbench run \
  --base-url http://10.1.115.4:8000/v1 \
  --model minimax-m2.5-awq \
  --pillar performance \
  --save results/minimax-2.5-perf.json

# ── Filter by language or difficulty ──
axbench run \
  --base-url http://10.1.115.4:8000/v1 \
  --model minimax-m2.5-awq \
  --language python \
  --difficulty hard \
  --save results/minimax-2.5-python-hard.json

# ── Compare two models ──
axbench compare results/minimax-2.5.json results/qwen3.5-35b.json

# ── List available tasks ──
axbench list-tasks
axbench list-tasks --pillar general_coding --language cpp
axbench list-tasks --source team/tom
```

## 7. Adding New Tasks

To add a new benchmark task:

1. Create a YAML file in the appropriate directory:
   - General tasks → `tasks/general/code_gen/<language>/` or `tasks/general/bug_fix/<language>/`
   - Team tasks → `tasks/team/<person>/`
2. Follow the task format for that evaluator type (see Section 5.1)
3. Run `axbench run --task <task_id>` to test it against the current model
4. Done — the runner auto-discovers all YAML files in `tasks/`

No code changes required. No registration. Drop the file, run the benchmark.

## 8. Scoring

### Per-Task Score
- Each test case is pass/fail
- Task score = (passed test cases) / (total test cases)
- A task "passes" if score == 1.0 (all test cases pass)

### Aggregate Scores
- **Per-pillar score** — average score across all tasks in that pillar
- **Per-language score** — average across all tasks in that language
- **Per-difficulty score** — average across all tasks at that difficulty
- **Overall quality score** — weighted average of pillar scores (performance excluded, it has its own metrics)

### What "Pass" Means
- Code compiles (C++) or parses (Python) without errors
- Code executes without runtime errors within the timeout
- All test case assertions pass
- For MMLU/GPQA: correct multiple-choice answer
- For HumanEval/MBPP/LiveCodeBench: follows pass@1 convention

## 9. Initial Task Set (Seed)

### Pillar 1 — Standard Benchmarks
- **MMLU:** 200 questions (random subset, configurable up to full 14k)
- **GPQA Diamond:** configurable subset of the diamond split
- **HumanEval:** 164 tasks (full set)
- **MBPP:** 500 tasks (full set)
- **LiveCodeBench:** configurable release/version subset

### Pillar 2 — Performance
- Handled by llama-benchy, default config:
  - `--pp 512 2048 4096`
  - `--tg 32 128`
  - `--depth 0 4096 8192`
  - `--latency-mode generation`
  - `--runs 3`

### Pillar 3 — General Coding (~20 tasks)

**Code Generation (12-14 tasks):**

Python (5-6 tasks):
- Easy: Binary search, string reversal with unicode handling
- Medium: LRU cache implementation, decorator with arguments
- Hard: Async task queue with cancellation, custom iterator protocol

C++ (5-6 tasks):
- Easy: Smart pointer usage, string tokenizer
- Medium: Template-based container, thread-safe queue
- Hard: Custom memory allocator, expression parser

Bash (2 tasks):
- Easy: File backup with rotation
- Medium: Parse structured log and extract error summary

**Bug Fixing (6-8 tasks):**

Python (4-5 tasks):
- Easy: Off-by-one error, mutable default argument
- Medium: Incorrect exception handling, race condition
- Hard: Broken generator with state corruption

C++ (2-3 tasks):
- Easy: Memory leak in constructor
- Medium: Dangling pointer after container resize
- Hard: Integer overflow in size calculation

### Pillar 4 — Team Real-World (~18 tasks)
- **Riccardo:** 5-6 bash/infra tasks (awaiting submission)
- **Tom:** 5-6 C++/MariaDB/networking tasks (awaiting submission)
- **Serge & Mykyta:** 5-6 C++/Python tasks (awaiting submission)

### Pillar 5 — Tool Calling (Future)
Not included in initial seed. Design ready (see Section 5.1).

## 10. Dependencies

- `httpx` — async HTTP client for API calls
- `pyyaml` — task file parsing
- `click` — CLI framework
- `rich` — terminal output formatting (tables, progress bars, scorecard)
- `llama-benchy` — performance benchmarking (existing tool, called as subprocess)

For standard benchmarks:
- `datasets` — HuggingFace datasets library (to download/cache MMLU, GPQA, HumanEval, MBPP, and LiveCodeBench where supported)

No other heavy ML frameworks. No external eval services. Runs anywhere Python runs.

## 11. Future Extensions

These are explicitly **not** being built now, but the plugin architecture supports them:

- **Tool calling evaluator (Pillar 5)** — first priority for future work
- **ISMS evaluator** — add when rubric-based scoring is ready
- **Multi-turn evaluator** — for conversational/agentic tasks
- **SQLite storage + web dashboard** — if file-based comparison outgrows CLI
- **Parallel execution** — run multiple tasks concurrently for speed
- **CI integration** — run axbench as a gate before deploying a new model
- **Regression alerts** — flag when a new model drops below a threshold on any pillar
