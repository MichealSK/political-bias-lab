from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def ensure_symlink(link_path: str | Path, target_path: str | Path) -> Path:
    """Point a project directory at durable cloud storage without copying it every run."""
    link = Path(link_path)
    target = Path(target_path)
    target.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        if link.resolve() == target.resolve():
            return link
        link.unlink()
    elif link.exists():
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            raise FileExistsError(
                f"Cannot replace non-empty {link}. Move/copy its contents to {target} first."
            )
    link.symlink_to(target, target_is_directory=True)
    return link


def link_colab_persistent_dirs(repo_root: str | Path, drive_root: str | Path) -> None:
    repo_root = Path(repo_root)
    drive_root = Path(drive_root)
    drive_root.mkdir(parents=True, exist_ok=True)
    for name in ["prepared", "results", "paper_bundle"]:
        ensure_symlink(repo_root / name, drive_root / name)


def _safe_member_parts(name: str) -> tuple[str, ...] | None:
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        return None
    return p.parts


def copy_kaggle_input(input_dir: str | Path, repo_root: str | Path) -> None:
    """Copy Kaggle input data into the writable project directories.

    Supports either extracted Parquet/JSON files or a `kaggle_input_*.zip`
    created by `src.transfer.make_kaggle_input_bundle`.
    """
    input_dir = Path(input_dir)
    repo_root = Path(repo_root)
    prepared = repo_root / "prepared"
    manifests = repo_root / "results" / "manifests"
    prepared.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    for file in input_dir.rglob("*.parquet"):
        shutil.copy2(file, prepared / file.name)
    for file in input_dir.rglob("*.json"):
        shutil.copy2(file, manifests / file.name)

    for archive in input_dir.rglob("*.zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            for member in zf.infolist():
                parts = _safe_member_parts(member.filename)
                if not parts or member.is_dir():
                    continue
                name = Path(member.filename).name
                if member.filename.startswith("prepared/") and name.endswith(".parquet"):
                    dest = prepared / name
                elif member.filename.startswith("manifests/") and name.endswith(".json"):
                    dest = manifests / name
                else:
                    continue
                with zf.open(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
