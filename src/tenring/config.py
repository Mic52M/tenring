"""Configuration loading: literature reference + optional personal profile."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_ROOT = Path(__file__).resolve().parents[2]
REF_PATH = _ROOT / "config" / "reference.yaml"
PROFILE_PATH = _ROOT / "config" / "profile.yaml"


def load_reference(path: Optional[Path] = None) -> dict:
    path = path or REF_PATH
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_profile(path: Optional[Path] = None) -> Optional[dict]:
    path = path or PROFILE_PATH
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_profile(profile: dict, path: Optional[Path] = None) -> Path:
    path = path or PROFILE_PATH
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(profile, fh, allow_unicode=True, sort_keys=False)
    return path
