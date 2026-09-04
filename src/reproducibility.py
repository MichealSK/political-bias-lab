from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import write_json


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision(root: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            manifest["gpu_name"] = torch.cuda.get_device_name(0)
            manifest["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        manifest["torch_error"] = repr(exc)

    for pkg in ["transformers", "accelerate", "bitsandbytes", "pandas", "numpy", "scipy", "sklearn"]:
        try:
            module = __import__(pkg)
            manifest[pkg] = getattr(module, "__version__", "unknown")
        except Exception:
            pass
    return manifest


def save_run_manifest(
    path: str | Path,
    *,
    config: dict[str, Any],
    root: str | Path = ".",
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "git_revision": git_revision(root),
        "config": config,
        "environment": environment_manifest(),
        "extra": extra or {},
    }
    return write_json(payload, path)
