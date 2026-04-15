from pathlib import Path

import pytest

from axbench.loader import TaskLoader


@pytest.fixture
def task_dir(tmp_path: Path):
    py_dir = tmp_path / "general" / "code_gen" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "task_a.yaml").write_text(
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

    cpp_dir = tmp_path / "general" / "code_gen" / "cpp"
    cpp_dir.mkdir(parents=True)
    (cpp_dir / "task_b.yaml").write_text(
        "id: cpp_task_b\n"
        "evaluator: code_gen\n"
        "language: cpp\n"
        "difficulty: hard\n"
        "source: general\n"
        "tags: [parsing]\n"
        "prompt: test\n"
        "test_harness: ''\n"
        "timeout_seconds: 15\n"
    )

    team_dir = tmp_path / "team" / "tom"
    team_dir.mkdir(parents=True)
    (team_dir / "task_c.yaml").write_text(
        "id: team_cpp_bug\n"
        "evaluator: bug_fix\n"
        "language: cpp\n"
        "difficulty: medium\n"
        "source: team/tom\n"
        "tags: [networking]\n"
        "prompt: fix it\n"
        "test_harness: ''\n"
        "timeout_seconds: 15\n"
    )

    return tmp_path


def test_loader_finds_all_tasks(task_dir: Path):
    loader = TaskLoader(task_dir)
    tasks = loader.load()
    assert len(tasks) == 3


def test_loader_filters_by_language(task_dir: Path):
    loader = TaskLoader(task_dir)
    tasks = loader.load(language="python")
    assert len(tasks) == 1
    assert tasks[0]["language"] == "python"


def test_loader_filters_by_difficulty(task_dir: Path):
    loader = TaskLoader(task_dir)
    tasks = loader.load(difficulty="hard")
    assert len(tasks) == 1
    assert tasks[0]["difficulty"] == "hard"


def test_loader_filters_by_evaluator(task_dir: Path):
    loader = TaskLoader(task_dir)
    tasks = loader.load(evaluator="code_gen")
    assert len(tasks) == 2


def test_loader_filters_by_pillar(task_dir: Path):
    loader = TaskLoader(task_dir)
    tasks = loader.load(pillar="team_real_world")
    assert len(tasks) == 1
    assert tasks[0]["source"] == "team/tom"


def test_loader_filters_by_tags(task_dir: Path):
    loader = TaskLoader(task_dir)
    tasks = loader.load(tags=["networking"])
    assert len(tasks) == 1
    assert tasks[0]["id"] == "team_cpp_bug"


def test_loader_load_single_task(task_dir: Path):
    loader = TaskLoader(task_dir)
    task = loader.load_one("python_task_a")
    assert task["id"] == "python_task_a"


def test_loader_raises_on_missing_task(task_dir: Path):
    loader = TaskLoader(task_dir)
    with pytest.raises(KeyError):
        loader.load_one("nonexistent")


def test_list_tasks_returns_sorted_metadata(task_dir: Path):
    loader = TaskLoader(task_dir)
    tasks = loader.list_tasks()
    assert [task["id"] for task in tasks] == [
        "team_cpp_bug",
        "cpp_task_b",
        "python_task_a",
    ]
    assert tasks[0]["pillar"] == "team_real_world"
