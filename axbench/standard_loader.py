from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import textwrap
from pathlib import Path
from random import Random
from typing import Any

from datasets import load_dataset


CACHE_SCHEMA_VERSION = 1

MMLU_DATASET = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_SPLIT = "test"
MMLU_SUBJECTS = [
    "college_mathematics",
    "college_physics",
    "college_computer_science",
    "electrical_engineering",
    "abstract_algebra",
    "formal_logic",
    "machine_learning",
    "college_chemistry",
]
MMLU_FULL_INDICES_BY_SUBJECT = {subject: list(range(10)) for subject in MMLU_SUBJECTS}
MMLU_QUICK_INDICES_BY_SUBJECT = {subject: list(range(1)) for subject in MMLU_SUBJECTS}

GPQA_DATASET = "Idavidrein/gpqa"
GPQA_CONFIG = "gpqa_diamond"
GPQA_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
GPQA_SPLIT = "train"
GPQA_FULL_INDICES = list(range(24))
GPQA_QUICK_INDICES = list(range(4))

HUMANEVAL_DATASET = "openai/openai_humaneval"
HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
HUMANEVAL_SPLIT = "test"
HUMANEVAL_FULL_INDICES = list(range(40))
HUMANEVAL_QUICK_INDICES = list(range(8))

MBPP_DATASET = "mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_SPLIT = "test"
MBPP_FULL_INDICES = list(range(24))
MBPP_QUICK_INDICES = list(range(4))

LIVECODEBENCH_DATASET = "livecodebench/execution-v2"
LIVECODEBENCH_REVISION = "ff6ea0e2a638001006ddcc41259eff23a4283fb2"
LIVECODEBENCH_SPLIT = "test"
LIVECODEBENCH_FULL_INDICES = list(range(12))
LIVECODEBENCH_QUICK_INDICES = list(range(4))


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_name: str
    revision: str
    split: str
    kind: str
    source: str
    language: str
    difficulty: str
    config: str | None = None
    gated: bool = False


def download_standard_tasks(tasks_dir: Path | str = "tasks") -> dict[str, int]:
    summary: dict[str, int] = {}
    for spec in _iter_standard_specs():
        tasks = _load_tasks_for_spec(spec, tasks_dir=tasks_dir, refresh=True)
        if tasks:
            summary[spec.key] = len(tasks)
    return summary


def load_standard_tasks(tasks_dir: Path | str = "tasks", quick: bool = False) -> list[dict]:
    selected_ids = _selected_standard_ids(quick=quick)
    tasks: list[dict] = []
    for spec in _iter_standard_specs():
        spec_tasks = _load_tasks_for_spec(spec, tasks_dir=tasks_dir)
        if quick:
            spec_tasks = [task for task in spec_tasks if task["id"] in selected_ids]
        tasks.extend(spec_tasks)
    return tasks


def _iter_standard_specs() -> list[DatasetSpec]:
    specs = [
        DatasetSpec(
            key=f"mmlu_{subject}",
            dataset_name=MMLU_DATASET,
            revision=MMLU_REVISION,
            split=MMLU_SPLIT,
            config=subject,
            kind="mmlu",
            source="standard/mmlu",
            language="text",
            difficulty="hard",
        )
        for subject in MMLU_SUBJECTS
    ]
    specs.extend(
        [
            DatasetSpec(
                key="gpqa_diamond",
                dataset_name=GPQA_DATASET,
                revision=GPQA_REVISION,
                split=GPQA_SPLIT,
                config=GPQA_CONFIG,
                kind="gpqa",
                source="standard/gpqa",
                language="text",
                difficulty="hard",
                gated=True,
            ),
            DatasetSpec(
                key="humaneval",
                dataset_name=HUMANEVAL_DATASET,
                revision=HUMANEVAL_REVISION,
                split=HUMANEVAL_SPLIT,
                kind="humaneval",
                source="standard/humaneval",
                language="python",
                difficulty="hard",
            ),
            DatasetSpec(
                key="mbpp",
                dataset_name=MBPP_DATASET,
                revision=MBPP_REVISION,
                split=MBPP_SPLIT,
                kind="mbpp",
                source="standard/mbpp",
                language="python",
                difficulty="medium",
            ),
            DatasetSpec(
                key="livecodebench_execution_v2",
                dataset_name=LIVECODEBENCH_DATASET,
                revision=LIVECODEBENCH_REVISION,
                split=LIVECODEBENCH_SPLIT,
                kind="livecodebench",
                source="standard/livecodebench",
                language="python",
                difficulty="hard",
            ),
        ]
    )
    return specs


def _selected_standard_ids(quick: bool) -> set[str]:
    selected: set[str] = set()
    for spec in _iter_standard_specs():
        indices = _selected_indices_for_spec(spec, quick=quick)
        if spec.kind == "mmlu":
            subject = spec.config or "unknown"
            selected.update(f"mmlu/{subject}/{index:04d}" for index in indices)
        elif spec.kind == "gpqa":
            selected.update(f"gpqa/{index:04d}" for index in indices)
        elif spec.kind == "humaneval":
            selected.update(f"HumanEval/{index}" for index in indices)
        elif spec.kind == "mbpp":
            selected.update(f"mbpp/{index:04d}" for index in indices)
        elif spec.kind == "livecodebench":
            selected.update(f"LCB/{index:04d}" for index in indices)
    return selected


def _selected_indices_for_spec(spec: DatasetSpec, quick: bool) -> list[int]:
    if spec.kind == "mmlu":
        mapping = MMLU_QUICK_INDICES_BY_SUBJECT if quick else MMLU_FULL_INDICES_BY_SUBJECT
        return list(mapping.get(spec.config or "", []))
    if spec.kind == "gpqa":
        return GPQA_QUICK_INDICES if quick else GPQA_FULL_INDICES
    if spec.kind == "humaneval":
        return HUMANEVAL_QUICK_INDICES if quick else HUMANEVAL_FULL_INDICES
    if spec.kind == "mbpp":
        return MBPP_QUICK_INDICES if quick else MBPP_FULL_INDICES
    if spec.kind == "livecodebench":
        return LIVECODEBENCH_QUICK_INDICES if quick else LIVECODEBENCH_FULL_INDICES
    raise ValueError(f"Unsupported standard benchmark kind: {spec.kind!r}")


def _load_tasks_for_spec(
    spec: DatasetSpec,
    tasks_dir: Path | str,
    refresh: bool = False,
) -> list[dict]:
    cache_dir = Path(tasks_dir) / "standard" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = cache_dir / f"{spec.key}.jsonl"
    meta_path = cache_dir / f"{spec.key}.meta.json"

    if not refresh and _cache_is_valid(meta_path, spec):
        return _read_cached_tasks(jsonl_path)

    if spec.gated and not os.environ.get("HF_TOKEN"):
        return _read_cached_tasks(jsonl_path) if jsonl_path.exists() else []

    tasks = _download_spec_tasks(spec)
    _write_cache(jsonl_path, meta_path, spec, tasks)
    return tasks


def _cache_is_valid(meta_path: Path, spec: DatasetSpec) -> bool:
    if not meta_path.exists():
        return False

    try:
        metadata = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False

    expected = _cache_metadata(spec)
    return all(metadata.get(key) == value for key, value in expected.items())


def _cache_metadata(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_name": spec.dataset_name,
        "config": spec.config,
        "revision": spec.revision,
        "split": spec.split,
        "kind": spec.kind,
        "selected_indices": _selected_indices_for_spec(spec, quick=False),
    }


def _read_cached_tasks(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    tasks = []
    for line in jsonl_path.read_text().splitlines():
        if line.strip():
            tasks.append(json.loads(line))
    return tasks


def _write_cache(jsonl_path: Path, meta_path: Path, spec: DatasetSpec, tasks: list[dict]) -> None:
    jsonl_path.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    metadata = _cache_metadata(spec) | {
        "task_ids": [task["id"] for task in tasks],
        "task_count": len(tasks),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))


def _download_spec_tasks(spec: DatasetSpec) -> list[dict]:
    token = os.environ.get("HF_TOKEN") if spec.gated else None
    dataset = load_dataset(
        spec.dataset_name,
        spec.config,
        split=spec.split,
        revision=spec.revision,
        token=token,
    )
    tasks = []
    for index in _selected_indices_for_spec(spec, quick=False):
        row = dataset[index]
        tasks.append(_normalize_row(spec, index, row))
    return tasks


def _normalize_row(spec: DatasetSpec, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    if spec.kind == "mmlu":
        subject = str(row.get("subject") or spec.config or "unknown")
        return {
            "id": f"mmlu/{subject}/{row_index:04d}",
            "evaluator": "standard",
            "kind": "mmlu",
            "language": spec.language,
            "difficulty": spec.difficulty,
            "source": spec.source,
            "tags": ["reasoning", "multiple-choice", subject],
            "subject": subject,
            "question": row["question"],
            "choices": list(row["choices"]),
            "answer": int(row["answer"]),
        }

    if spec.kind == "gpqa":
        question = row.get("question") or row.get("Question")
        if not question:
            raise ValueError("GPQA row is missing a question")
        choices, answer_index = _normalize_gpqa_choices(row, question, row_index)
        return {
            "id": f"gpqa/{row_index:04d}",
            "evaluator": "standard",
            "kind": "gpqa",
            "language": spec.language,
            "difficulty": spec.difficulty,
            "source": spec.source,
            "tags": ["science", "multiple-choice"],
            "question": question,
            "choices": choices,
            "answer": answer_index,
        }

    if spec.kind == "humaneval":
        return {
            "id": str(row["task_id"]),
            "evaluator": "standard",
            "kind": "humaneval",
            "language": spec.language,
            "difficulty": spec.difficulty,
            "source": spec.source,
            "tags": ["python", "function-completion"],
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row["test"],
        }

    if spec.kind == "mbpp":
        return {
            "id": f"mbpp/{row_index:04d}",
            "evaluator": "standard",
            "kind": "mbpp",
            "language": spec.language,
            "difficulty": spec.difficulty,
            "source": spec.source,
            "tags": ["python", "programming"],
            "task_id": row["task_id"],
            "text": row["text"],
            "test_list": row.get("test_list", []),
            "challenge_test_list": row.get("challenge_test_list", []),
            "test_setup_code": row.get("test_setup_code", ""),
        }

    if spec.kind == "livecodebench":
        entry_point = str(row.get("function_name") or "solution")
        starter_code = _build_livecodebench_starter_code(str(row.get("code", "")), entry_point)
        prompt = _build_livecodebench_prompt(entry_point, row)
        test = f"assert {row['input']} == {row['output']}"
        return {
            "id": f"LCB/{row_index:04d}",
            "question_id": row.get("question_id"),
            "evaluator": "standard",
            "kind": "livecodebench",
            "language": spec.language,
            "difficulty": str(row.get("difficulty") or spec.difficulty).lower(),
            "source": spec.source,
            "tags": ["python", "coding", "contest"],
            "prompt": prompt,
            "starter_code": starter_code,
            "entry_point": entry_point,
            "test": test,
        }

    raise ValueError(f"Unsupported standard benchmark kind: {spec.kind!r}")


def _normalize_gpqa_choices(row: dict[str, Any], question: str, row_index: int) -> tuple[list[str], int]:
    if "choices" in row and "answer" in row:
        return [str(choice) for choice in row["choices"]], int(row["answer"])

    correct = row.get("correct_answer") or row.get("Correct Answer")
    incorrects = [
        row.get("incorrect_answer_1") or row.get("Incorrect Answer 1"),
        row.get("incorrect_answer_2") or row.get("Incorrect Answer 2"),
        row.get("incorrect_answer_3") or row.get("Incorrect Answer 3"),
    ]
    if correct is None or any(value is None for value in incorrects):
        raise ValueError("GPQA row is missing answer choices")

    shuffled = [str(correct), *(str(value) for value in incorrects)]
    seed = int(hashlib.sha256(f"{question}:{row_index}".encode()).hexdigest()[:8], 16)
    rng = Random(seed)
    rng.shuffle(shuffled)
    return shuffled, shuffled.index(str(correct))


def _build_livecodebench_starter_code(solution_code: str, entry_point: str) -> str:
    lines = solution_code.splitlines()
    pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
    for index, line in enumerate(lines):
        if not pattern.search(line):
            continue
        prelude = "\n".join(lines[:index]).rstrip()
        signature = line.rstrip()
        stub = f"{signature}\n    pass"
        return f"{prelude}\n\n{stub}".strip() if prelude else stub
    return textwrap.dedent(
        f"""
        def {entry_point}(*args, **kwargs):
            pass
        """
    ).strip()


def _build_livecodebench_prompt(entry_point: str, row: dict[str, Any]) -> str:
    difficulty = row.get("difficulty") or "unknown"
    contest_id = row.get("contest_id") or "contest"
    return textwrap.dedent(
        f"""
        Implement the Python function `{entry_point}`.

        Contest: {contest_id}
        Difficulty: {difficulty}

        It must satisfy this sample:
        `{row['input']}` -> `{row['output']}`
        """
    ).strip()
