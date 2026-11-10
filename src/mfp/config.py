"""Pemuatan konfigurasi dan resolusi path proyek."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["random_state"])

    def path(self, key: str) -> Path:
        """Path absolut dari entri `paths` (folder dibuat otomatis)."""
        p = PROJECT_ROOT / self.raw["paths"][key]
        (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
