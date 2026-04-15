# AXBench — Comprehensive LLM Benchmarking for AX-Office.ai

Quick-decision benchmarking: run one command, compare models, ship or do not ship.

## Install

```bash
cd /home/msai/vllm/benchmarking
uv venv && uv pip install -e .
```

## Run

```bash
# Full benchmark suite — Minimax 2.5 AWQ (primary)
axbench run \
  --base-url http://10.1.115.4:8000/v1 \
  --model minimax-m2.5-awq \
  --save results/minimax-2.5.json

# Full benchmark suite — Qwen3.5-35B-A3B-FP8
axbench run \
  --base-url http://localhost:8003/v1 \
  --model Qwen3.5-35B-A3B \
  --save results/qwen3.5-35b.json

# Full benchmark suite — Qwen3-4B-Instruct-2507-FP8 (small)
axbench run \
  --base-url http://localhost:8006/v1 \
  --model Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --save results/qwen3-4b.json

# Quick coding-only run
axbench run \
  --base-url http://10.1.115.4:8000/v1 \
  --model minimax-m2.5-awq \
  --pillar general_coding team_real_world \
  --save results/minimax-2.5-coding.json

# Compare two models
axbench compare results/minimax-2.5.json results/qwen3.5-35b.json

# See what tasks are available
axbench list-tasks
axbench list-tasks --language cpp --difficulty hard
```

The default full suite now includes:
- general coding YAML tasks under `tasks/`
- built-in standard benchmark starter tasks for `MMLU`, `GPQA`, `HumanEval`, `MBPP`, and `LiveCodeBench`
- one `llama-benchy` performance run

The performance pillar is the slowest part of the suite because it shells out to `llama-benchy` and measures throughput and TTFT.

## Add Tasks

Drop a YAML file in `tasks/general/` or `tasks/team/<name>/`.
See [tasks/team/CONTRIBUTING.md](/home/msai/vllm/benchmarking/tasks/team/CONTRIBUTING.md).

## Run Tests

```bash
uv run pytest tests/ -v
```

## Pillars

| Pillar | Status | What it measures |
|--------|--------|------------------|
| 1. Standard (MMLU, GPQA, HumanEval, MBPP, LiveCodeBench) | Ready | General reasoning plus classic and modern coding baseline |
| 2. Performance (llama-benchy) | Ready | Tokens/s, TTFT, latency |
| 3. General Coding | Ready | Python/C++/bash code generation plus bug fix |
| 4. Team Real-World | Ready (awaiting task submissions) | Your actual daily use cases |
| 5. Tool Calling | Planned | Correct tool selection and arguments |
