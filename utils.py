from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str) -> str:
    if preferred.upper() == "CUDA" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_loss(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.6f}"


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8"))
