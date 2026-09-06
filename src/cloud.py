from __future__ import annotations

from pathlib import Path


def ensure_symlink(link_path: str | Path, target_path: str | Path) -> Path:
    """Point a project directory at durable Google Drive storage."""
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
