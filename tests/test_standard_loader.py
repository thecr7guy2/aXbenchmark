from pathlib import Path

from axbench.standard_loader import DatasetSpec, download_standard_tasks, load_standard_tasks


def test_load_standard_tasks_writes_and_reuses_cache(tmp_path: Path, monkeypatch):
    spec = DatasetSpec(
        key="humaneval",
        dataset_name="openai/openai_humaneval",
        revision="rev-1",
        split="test",
        kind="humaneval",
        source="standard/humaneval",
        language="python",
        difficulty="hard",
    )
    rows = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
        }
    ]

    monkeypatch.setattr("axbench.standard_loader._iter_standard_specs", lambda: [spec])
    monkeypatch.setattr("axbench.standard_loader.HUMANEVAL_FULL_INDICES", [0])
    monkeypatch.setattr("axbench.standard_loader.HUMANEVAL_QUICK_INDICES", [0])
    monkeypatch.setattr("axbench.standard_loader.load_dataset", lambda *args, **kwargs: rows)

    tasks = load_standard_tasks(tasks_dir=tmp_path / "tasks")
    assert [task["id"] for task in tasks] == ["HumanEval/0"]

    cache_dir = tmp_path / "tasks" / "standard" / "cache"
    assert (cache_dir / "humaneval.jsonl").exists()
    assert (cache_dir / "humaneval.meta.json").exists()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_dataset should not be called when cache is valid")

    monkeypatch.setattr("axbench.standard_loader.load_dataset", fail_if_called)
    cached_tasks = load_standard_tasks(tasks_dir=tmp_path / "tasks")
    assert cached_tasks == tasks


def test_download_standard_tasks_skips_gated_datasets_without_token(tmp_path: Path, monkeypatch):
    spec = DatasetSpec(
        key="gpqa_diamond",
        dataset_name="Idavidrein/gpqa",
        revision="rev-1",
        split="train",
        kind="gpqa",
        source="standard/gpqa",
        language="text",
        difficulty="hard",
        config="gpqa_diamond",
        gated=True,
    )

    monkeypatch.setattr("axbench.standard_loader._iter_standard_specs", lambda: [spec])
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_dataset should not be called without HF_TOKEN")

    monkeypatch.setattr("axbench.standard_loader.load_dataset", fail_if_called)
    summary = download_standard_tasks(tasks_dir=tmp_path / "tasks")
    assert summary == {}


def test_load_standard_tasks_quick_filters_to_quick_subset(tmp_path: Path, monkeypatch):
    spec = DatasetSpec(
        key="humaneval",
        dataset_name="openai/openai_humaneval",
        revision="rev-1",
        split="test",
        kind="humaneval",
        source="standard/humaneval",
        language="python",
        difficulty="hard",
    )
    rows = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def sub(a, b):\n",
            "entry_point": "sub",
            "test": "def check(candidate):\n    assert candidate(3, 1) == 2",
        },
    ]

    monkeypatch.setattr("axbench.standard_loader._iter_standard_specs", lambda: [spec])
    monkeypatch.setattr("axbench.standard_loader.HUMANEVAL_FULL_INDICES", [0, 1])
    monkeypatch.setattr("axbench.standard_loader.HUMANEVAL_QUICK_INDICES", [1])
    monkeypatch.setattr("axbench.standard_loader.load_dataset", lambda *args, **kwargs: rows)

    tasks = load_standard_tasks(tasks_dir=tmp_path / "tasks", quick=True)
    assert [task["id"] for task in tasks] == ["HumanEval/1"]
