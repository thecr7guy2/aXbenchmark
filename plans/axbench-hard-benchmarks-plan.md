# AXBench — Benchmark Suite V2 Plan

**Author:** Maniraj Sai | AI Engineer, aXite Security Tools  
**Date:** 2026-04-15  
**Status:** Draft — Revised for Implementation

---

## Problem

The current benchmark does not separate models reliably enough to support ranking decisions.

This is confirmed by the current code and saved runs:

1. **Pillar 1 (Standard)** is a toy placeholder suite. It currently ships only 5 built-in tasks in `axbench/builtin_tasks.py`, including questions like `"What is 12 divided by 3?"`, a `def add(a, b)` HumanEval stub, and a square function task.
2. **Pillar 3 (General Coding)** contains several tasks that are too easy or too shallow to be strong differentiators. A few concepts are worth keeping in spirit, but most of the suite should be replaced.
3. The current saved runs for `qwen3-4b` and `qwen3.5-35b` both land at the same overall quality score with the standard pillar at 100%, which means the suite is underpowered for model comparison.

The benchmark needs to become a stronger **internal ranking benchmark**:

- deterministic
- repeatable
- materially harder
- realistic to maintain
- compatible with the current AXBench architecture

This plan does **not** claim strict comparability to public leaderboard numbers. It aims to make AXBench internally trustworthy first.

---

## Principles

1. **Use real datasets for the standard pillar.** Toy built-ins must be replaced.
2. **Pin data revisions and question IDs.** Pinned row indices alone are not enough.
3. **Keep the evaluation model simple.** Avoid tasks that require flaky timing assumptions, undefined behavior detection, or external infra that the current sandbox cannot guarantee.
4. **Make quick mode a first-class selection mode.** It should be deterministic and implemented where task selection already happens.
5. **Prefer an honest v2 over an inflated v2.** Fewer good tasks beat many brittle ones.

---

## Goals

- Full benchmark run target: **60-120 minutes** on a 35B-class model, to be validated after implementation
- Quick mode target: **10-20 minutes**
- Same fixed question set every run for both full and quick mode
- Standard pillar should stop saturating at 100% for competent models
- General coding should test stateful, multi-step, multi-function reasoning
- GPQA should degrade gracefully when `HF_TOKEN` is missing
- Results should clearly identify full vs quick runs and the benchmark suite version

---

## Non-Goals

- Exact reproduction of public leaderboard methodology
- Multi-turn evaluation
- LLM-as-judge scoring
- Heavy external runtime dependencies in Bash tasks
- Timing-sensitive correctness checks that require sanitizers, TSAN, ASAN, or kernel-level guarantees

---

## Architecture Decision

### Pillar 1 — Dataset-backed standard suite

- Replace the toy standard tasks with a deterministic dataset-backed loader.
- Use Hugging Face datasets with:
  - explicit dataset revision pins
  - explicit split names
  - explicit selected row IDs
  - local normalized cache files

### Pillar 2 — Performance

- Keep the performance pillar separate from standard task loading.
- Do **not** hide performance task ownership inside the standard loader.

### Pillar 3 — General coding refresh

- Replace the current 16 YAML tasks with a new hard suite.
- Preserve only a few current concepts in rewritten form, not by keeping the old task files.
- Avoid tasks that depend on unsupported external binaries or unreliable concurrency verification.

### Quick mode

- Implement quick mode at the **task selection layer**, which currently lives in `axbench/cli.py`, not `axbench/runner.py`.
- Quick mode must always be a deterministic subset of the full suite.

---

## Phase 1 — Standard Pillar Rebuild

**Goal:** replace the toy built-in standard tasks with a real, deterministic standard suite.

### New modules

- **Add:** `axbench/standard_loader.py`
  Responsibilities:
  - download or load datasets from Hugging Face
  - pin dataset revisions
  - normalize selected rows into AXBench task dicts
  - cache normalized tasks locally
  - expose `load_standard_tasks(full=True|False)`

- **Add:** `axbench/perf_tasks.py`
  Responsibilities:
  - hold the performance task definitions currently mixed into `builtin_tasks.py`

- **Delete:** `axbench/builtin_tasks.py`

### Selection and caching rules

- Cache location: `tasks/standard/cache/`
- Cache format: normalized `.jsonl`
- Each dataset cache file must encode:
  - dataset name
  - config / subset
  - revision pin
  - split
  - selected task IDs
- Full and quick selections are hardcoded constants in `standard_loader.py`

### Initial full suite size

The first real standard suite should be large enough to have signal, but smaller than the previous 460-task proposal so runtime is realistic.

| Benchmark | Count | Notes |
|---|---:|---|
| MMLU | 80 | technical subjects only |
| GPQA Diamond | 24 | only when `HF_TOKEN` is present |
| HumanEval | 40 | pinned task subset |
| MBPP | 24 | pinned harder subset |
| LiveCodeBench | 12 | pinned contest-style subset with executable tests |
| **Total** | **180** | |

### Initial quick suite size

| Benchmark | Count |
|---|---:|
| MMLU | 8 |
| GPQA Diamond | 4 |
| HumanEval | 8 |
| MBPP | 4 |
| LiveCodeBench | 4 |
| **Total** | **28** |

### Important implementation constraints

- Use explicit revision pins for every upstream dataset.
- Do not describe the resulting scores as public-leaderboard-comparable.
- GPQA skip behavior:
  - if `HF_TOKEN` is missing, omit GPQA tasks from the selected run
  - record a warning in run metadata
  - do not fail the whole run

### Files changed

- **Add:** `axbench/standard_loader.py`
- **Add:** `axbench/perf_tasks.py`
- **Delete:** `axbench/builtin_tasks.py`
- **Modify:** `axbench/cli.py`
  - import `standard_loader` and `perf_tasks`
  - add `axbench download`
  - make full vs quick task selection explicit
- **Modify:** `tests/test_cli.py`
- **Modify:** `tests/test_standard_evaluator.py`
- **Add:** loader tests for dataset cache and revision handling

---

## Phase 2 — Results and Metadata Cleanup

**Goal:** make run output honest about what suite ran and what was skipped.

### Metadata additions

Extend `RunMetadata` with:

- `benchmark_suite_version: str`
- `quick_mode: bool`
- `warnings: list[str]`

### Behavior changes

- Quick runs must serialize `quick_mode=true`
- Full runs must serialize `quick_mode=false`
- If GPQA is skipped because `HF_TOKEN` is missing, record a warning such as:
  - `"GPQA Diamond skipped: HF_TOKEN not set"`

### Why this phase exists

The current results model only tracks selected and skipped task IDs from user filtering. It does not clearly distinguish:

- a smaller intentional quick suite
- dynamic omission of a dataset because credentials are missing
- the version of the suite that produced the score

That ambiguity should be removed before comparing old and new runs.

### Files changed

- **Modify:** `axbench/results.py`
- **Modify:** `tests/test_results.py`
- **Modify:** `tests/test_cli.py`

---

## Phase 3 — General Coding Suite Rebuild

**Goal:** replace the current 16-task general-coding suite with a stronger, benchmark-worthy set that still fits the current evaluator model.

### Decision

- Delete the existing 16 YAML files under `tasks/general/`
- Add a new 20-task suite
- Carry forward only a few concepts in rewritten form:
  - async queue
  - LRU cache
  - thread-safe queue
  - generator / stateful bug fix

These will be new task definitions with stricter tests, not kept as-is.

### Why not keep the current suite

Several current tasks are benchmark-noise rather than benchmark-signal:

- binary search
- string reverse
- mutable default bug
- off-by-one bug
- trivial memory leak fix
- tokenizer split
- simple Docker health check
- basic log rotation

They are useful as smoke tests, but not strong ranking tasks.

### New Python tasks (8)

| ID | Task | Notes | Quick? |
|---|---|---|---|
| `python_expression_parser` | tokenizer + parser + evaluator for arithmetic expressions | recursive descent, precedence, parentheses | yes |
| `python_json_parser` | parse objects, arrays, strings, numbers, booleans, null | structured parsing, recursion | yes |
| `python_lru_cache_v2` | capacity-bounded cache with update + eviction correctness | rewritten stricter version of current concept | yes |
| `python_async_queue_v2` | bounded async FIFO with blocking semantics | rewritten stricter version with producer/consumer coordination | yes |
| `python_graph_bfs_cycle` | BFS traversal plus cycle detection | multi-function graph reasoning | |
| `python_dependency_resolver` | topological ordering with cycle error | stateful graph evaluation | |
| `python_merkle_tree` | build root and verify inclusion proof | hashing, tree structure, verification | |
| `python_patch_apply` | apply unified diff hunks to in-memory text | parsing + transformation + edge cases | |

### New C++ tasks (8)

| ID | Task | Notes | Quick? |
|---|---|---|---|
| `cpp_thread_safe_queue_v2` | stricter queue correctness under multi-thread producer/consumer use | rewritten stronger version of current concept | yes |
| `cpp_lru_cache` | LRU cache with eviction and update semantics | stateful container logic | yes |
| `cpp_thread_pool` | submit work items and collect futures | concurrency but still testable in current harness | yes |
| `cpp_memory_pool` | fixed-block allocator with reuse verification | avoids sanitizer dependence | yes |
| `cpp_interval_map` | key-range assignment and lookup | stateful update semantics | |
| `cpp_string_intern` | intern strings and preserve pointer / lookup semantics | ownership and container behavior | |
| `cpp_csv_parser` | parse quoted CSV rows into structured fields | parsing with escape edge cases | |
| `cpp_trie_router` | longest-prefix route match | tree structure and lookup behavior | |

### New Bash tasks (4)

Use only common shell tools that are reasonable to expect in the local environment.

| ID | Task | Notes | Quick? |
|---|---|---|---|
| `bash_config_diff` | compare two config dirs and report added/removed/changed keys | structured filesystem diff | yes |
| `bash_mini_make` | process dependency file and build in topological order | files, timestamps, DAG behavior | yes |
| `bash_log_anomaly_detector` | summarize endpoint latency from log lines using `awk`/`sort` | no `jq` dependency | |
| `bash_archive_prune` | archive old files, retain recent archives, enforce retention | filesystem state transitions | |

### Test requirements

#### Python

- minimum 4 test cases per task
- at least 1 edge case
- at least 1 stateful or multi-step case
- no single-expression trivia tasks

#### C++

- minimum 2 distinct assertion groups in the harness
- at least 1 edge or state-transition scenario
- avoid tasks whose correctness depends on sanitizers or UB detection

#### Bash

- minimum 3 fixture-driven test cases
- prefer filesystem state checks over stdout-only checks
- avoid dependencies on `jq`, `nc`, Docker, or host-specific services

### Files changed

- **Delete:** current 16 YAML files under `tasks/general/`
- **Add:** 20 new YAML files under `tasks/general/`
- **Modify:** `tests/test_loader.py`
- **Modify:** evaluator tests as needed for stronger fixtures

---

## Phase 4 — Quick Mode

**Goal:** add a deterministic quick suite without pretending it is the same thing as the full benchmark.

### Implementation

- **Modify:** `axbench/cli.py`
  - add `--quick`
  - select the quick subset during task assembly
  - keep the implementation near `_available_tasks()` / selection helpers

- `Runner` should remain responsible for execution, not benchmark composition.

### Quick suite composition

- 28 standard tasks
- 8 general-coding tasks
- 1 performance task

### Quick suite total

- **37 executed tasks** when GPQA is available
- **33 executed tasks** when GPQA is omitted due to missing `HF_TOKEN`

### UX requirements

- result metadata must mark quick mode explicitly
- `axbench compare` must display a clear quick/full label
- quick runs should not be silently compared as if they were full runs

### Files changed

- **Modify:** `axbench/cli.py`
- **Modify:** `axbench/results.py`
- **Modify:** `tests/test_cli.py`

---

## Phase 5 — Validation

**Goal:** prove the new suite produces real ranking signal.

### Acceptance criteria

1. Full suite runs end-to-end on at least one model without crashing
2. Quick suite completes in under 20 minutes on a 35B-class model
3. Standard pillar no longer saturates at 100% for strong models
4. Overall spread between a 4B-class model and a 35B-class model is meaningfully larger than today
5. GPQA omission due to missing `HF_TOKEN` is recorded as a warning, not a failure
6. `axbench compare` distinguishes quick and full runs
7. Scores are stable across reruns of the same model within an acceptable variance band

### Suggested smoke test sequence

```bash
# 1. Pre-populate the dataset cache
axbench download

# 2. Quick run on a small model
axbench run \
  --base-url http://localhost:8006/v1 \
  --model Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --quick \
  --save results/qwen3-4b-v2-quick.json

# 3. Full run on the same small model
axbench run \
  --base-url http://localhost:8006/v1 \
  --model Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --save results/qwen3-4b-v2.json

# 4. Full run on a larger model
axbench run \
  --base-url http://localhost:8003/v1 \
  --model Qwen3.5-35B-A3B \
  --save results/qwen3.5-35b-v2.json

# 5. Compare full runs
axbench compare \
  results/qwen3-4b-v2.json \
  results/qwen3.5-35b-v2.json
```

---

## File Summary

| Action | File / Area |
|---|---|
| Add | `axbench/standard_loader.py` |
| Add | `axbench/perf_tasks.py` |
| Delete | `axbench/builtin_tasks.py` |
| Modify | `axbench/cli.py` |
| Modify | `axbench/results.py` |
| Add | new dataset cache files under `tasks/standard/cache/` |
| Delete | current YAML files under `tasks/general/` |
| Add | 20 new YAML files under `tasks/general/` |
| Modify | `tests/test_cli.py` |
| Modify | `tests/test_results.py` |
| Modify | `tests/test_loader.py` |
| Modify | `tests/test_standard_evaluator.py` |

---

## Recommended Order of Work

1. Implement the standard loader and split performance tasks out of `builtin_tasks.py`
2. Add result metadata for suite version, quick mode, and warnings
3. Add `axbench download`
4. Add `--quick` and update compare output
5. Replace the general-coding YAML suite
6. Run validation and adjust full-suite size if runtime is too high

---

## Out of Scope

- Team real-world pillar redesign
- Tool-calling pillar
- Web dashboard changes
- Judge-model scoring
- Multi-turn agent benchmarks
