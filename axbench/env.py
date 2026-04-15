from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> Path | None:
    dotenv_path = Path(path)
    if not dotenv_path.is_absolute():
        dotenv_path = Path.cwd() / dotenv_path

    if not dotenv_path.exists():
        return None

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)

    return dotenv_path
