from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load one or more YAML configs; later files override earlier keys."""
    cfg: dict[str, Any] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config {path} must be a mapping")
        cfg.update(data)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
