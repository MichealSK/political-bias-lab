from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(
    default_path: str | Path = "config/default.yaml",
    profile_path: str | Path | None = None,
) -> dict[str, Any]:
    default_path = Path(default_path)
    with default_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if profile_path is not None:
        with Path(profile_path).open("r", encoding="utf-8") as f:
            profile_doc = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, profile_doc)

    profile_name = cfg.get("profile", "pilot")
    if profile_name not in cfg["profiles"]:
        raise KeyError(f"Unknown profile: {profile_name}")
    cfg["active_profile"] = deepcopy(cfg["profiles"][profile_name])
    cfg["profile"] = profile_name
    return cfg


def profile_value(cfg: dict[str, Any], phase: str, key: str) -> Any:
    return cfg["active_profile"][phase][key]
