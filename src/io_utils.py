from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False, dir=path.parent
    ) as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False, default=str)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet(df: pd.DataFrame, path: str | Path, compression: str = "zstd") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression=compression)
    os.replace(tmp, path)
    return path


def checkpoint_records(
    records: Iterable[dict[str, Any]],
    path: str | Path,
    compression: str = "zstd",
) -> Path:
    """Merge records into a Parquet checkpoint, de-duplicating exact rows."""
    path = Path(path)
    new_df = pd.DataFrame(list(records))
    if new_df.empty:
        return path
    if path.exists():
        old_df = pd.read_parquet(path)
        merged = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates()
    else:
        merged = new_df.drop_duplicates()
    return write_parquet(merged, path, compression=compression)


def load_optional_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")
