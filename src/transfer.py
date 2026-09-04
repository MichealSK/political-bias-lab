from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from .io_utils import write_json


def make_kaggle_input_bundle(root: str | Path, output_zip: str | Path) -> Path:
    root = Path(root)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    files = list((root / "prepared").glob("*.parquet"))
    manifests = list((root / "results" / "manifests").glob("*.json"))
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f"prepared/{f.name}")
        for f in manifests:
            zf.write(f, arcname=f"manifests/{f.name}")
    return output_zip


def make_kaggle_output_bundle(root: str | Path, output_zip: str | Path) -> Path:
    root = Path(root)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    include_dirs = [root / "results" / "raw", root / "results" / "manifests"]
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for directory in include_dirs:
            if directory.exists():
                for f in directory.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=f.relative_to(root).as_posix())
    return output_zip


def import_kaggle_output_bundle(root: str | Path, bundle_zip: str | Path) -> None:
    root = Path(root)
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        # Limit extraction to results/ paths created by this project.
        for member in zf.infolist():
            name = Path(member.filename)
            if not name.parts or name.parts[0] != "results":
                continue
            dest = (root / name).resolve()
            if root.resolve() not in dest.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
