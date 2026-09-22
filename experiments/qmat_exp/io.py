from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root() / p


def load_config(path: str | Path) -> dict:
    def read(current, seen):
        current = resolve_path(current).resolve()
        if current in seen:
            raise ValueError("cyclic configuration inheritance")
        with current.open(encoding="utf-8") as f:
            cfg = json.load(f)
        parent = cfg.pop("extends", None)
        if parent is None:
            return cfg
        return merge(read(current.parent / parent, seen | {current}), cfg)

    def merge(base, overrides):
        for key, value in overrides.items():
            base[key] = merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
        return base

    return read(path, set())


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: object) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str | Path) -> object:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name, "")) for name in fieldnames})


def read_csv(path: str | Path) -> list[dict]:
    p = resolve_path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
