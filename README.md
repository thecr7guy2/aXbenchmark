# ⚡ AXBench

> On-prem LLM benchmarking built for real model-swap decisions.

Standard benchmarks are noisy. A model can ace MMLU and still stumble on
the tasks your team actually runs. AXBench measures coding quality,
reasoning, throughput, and your team's own real-world cases in a **single
command**, so you can ship or roll back with confidence.

---

## 🚀 Install

```bash
uv venv && uv pip install -e .
```

---

## 🏃 Run

```bash
# Full benchmark suite
axbench run \
  --base-url http://localhost:8000/v1 \
  --model your-model-name \
  --save results/my-model.json

# Coding pillars only (faster)
axbench run \
  --base-url http://localhost:8000/v1 \
  --model your-model-name \
  --pillar general_coding team_real_world \
  --save results/my-model-coding.json

# Compare two models side by side
axbench compare results/model-a.json results/model-b.json

# Explore available tasks
axbench list-tasks
axbench list-tasks --language cpp --difficulty hard
```


---

## 🧱 Pillars

| # | Pillar | Status | What it measures |
|---|--------|--------|-----------------|
| 1 | 📚 Standard (MMLU, GPQA, HumanEval, MBPP, LiveCodeBench) | ✅ Ready | General reasoning + classic and modern coding baselines |
| 2 | ⚙️ Performance (llama-benchy) | ✅ Ready | Tokens/s, TTFT, latency |
| 3 | 💻 General Coding | ✅ Ready | Python / C++ / Bash generation + bug fixes |
| 4 | 🧑‍💼 Team Real-World | ✅ Ready (awaiting tasks) | Your actual daily use cases |
| 5 | 🔧 Tool Calling | 🗓️ Planned | Correct tool selection and arguments |

---

## ➕ Add Tasks

Each team member has a personal folder under `tasks/team/`. Drop a YAML
file in yours. No setup needed.

---

## 🧪 Run Tests

```bash
uv run pytest tests/ -v
```
