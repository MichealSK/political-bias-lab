from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    prepared: Path
    results: Path
    raw: Path
    derived: Path
    figures: Path
    manifests: Path
    paper_bundle: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ProjectPaths":
        root_path = Path(root or os.getenv("BIASLAB_ROOT", ".")).resolve()
        results = root_path / "results"
        return cls(
            root=root_path,
            data=root_path / "data",
            prepared=root_path / "prepared",
            results=results,
            raw=results / "raw",
            derived=results / "derived",
            figures=results / "figures",
            manifests=results / "manifests",
            paper_bundle=root_path / "paper_bundle",
        )

    def ensure(self) -> "ProjectPaths":
        for p in (
            self.data,
            self.prepared,
            self.results,
            self.raw,
            self.derived,
            self.figures,
            self.manifests,
            self.paper_bundle,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self
