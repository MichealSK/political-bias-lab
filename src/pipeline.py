from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import add_holm_to_phase1, make_figures
from .config import load_config
from .data_builder import prepare_universe
from .hf_runner import HFLocalModel
from .io_utils import write_json, write_parquet
from .paths import ProjectPaths
from .phase1_tsc import run_phase1, save_phase1_analysis
from .phase2_persona import run_phase2, save_phase2_analysis
from .phase3_evasion import run_phase3, save_phase3_analysis
from .phase4_pluralism import (
    build_blinded_human_sheet,
    run_phase4_generation,
    run_phase4_judging,
    save_phase4_analysis,
)
from .reproducibility import file_sha256, save_run_manifest, set_global_seed
from .sampling import make_phase1_sample


def prepare_project(cfg: dict, root: str | Path = ".") -> dict:
    paths = ProjectPaths.from_root(root).ensure()
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)

    logical = cfg["phase1"]["logical_universe"]
    research_entities = paths.data / "entities_research.csv"
    research_pairs = paths.data / "entity_pairs_research.csv"
    stats = prepare_universe(
        paths.prepared,
        target_templates=int(logical["target_templates"]),
        target_entities=int(logical["target_entities"]),
        seed=seed,
        research_entities_csv=research_entities if research_entities.exists() else None,
        research_pairs_csv=research_pairs if research_pairs.exists() else None,
    )

    # Convert compact source CSVs to Parquet for notebook I/O.
    for name in ["persona_items", "evasion_items", "pluralism_cases", "bias_exemplars"]:
        df = pd.read_csv(paths.data / f"{name}.csv")
        write_parquet(df, paths.prepared / f"{name}.parquet")

    templates = pd.read_parquet(paths.prepared / "sentiment_templates.parquet")
    entities = pd.read_parquet(paths.prepared / "entities.parquet")
    pairs = pd.read_parquet(paths.prepared / "entity_pairs.parquet")
    p1_cfg = cfg["active_profile"]["phase1"]
    sample = make_phase1_sample(
        templates,
        entities,
        pairs,
        n_templates=int(p1_cfg["n_templates"]),
        n_pairs=int(p1_cfg["n_pairs"]),
        seed=seed,
    )
    write_parquet(sample, paths.prepared / "phase1_sample.parquet")

    stats.update({
        "profile": cfg["profile"],
        "phase1_sample_rows": int(len(sample)),
        "uses_real_entities": bool(entities.get("is_real", pd.Series(dtype=bool)).astype(bool).any()),
    })
    write_json(stats, paths.manifests / "data_preparation.json")

    data_hashes = {}
    for file in sorted(paths.prepared.glob("*.parquet")):
        data_hashes[file.name] = file_sha256(file)
    write_json(data_hashes, paths.manifests / "prepared_data_sha256.json")
    return stats


def load_model(model_cfg: dict) -> HFLocalModel:
    return HFLocalModel(
        model_cfg["id"],
        short_name=model_cfg.get("short_name"),
        revision=model_cfg.get("revision"),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        max_context_tokens=int(model_cfg.get("max_context_tokens", 4096)),
    )


def run_one_model(cfg: dict, model_cfg: dict, root: str | Path = ".") -> dict:
    paths = ProjectPaths.from_root(root).ensure()
    seed = int(cfg["project"]["seed"])
    checkpoint_every = int(cfg["storage"]["checkpoint_every"])
    compression = cfg["storage"]["parquet_compression"]

    model = load_model(model_cfg)
    metadata = model.metadata()
    try:
        phase1_cases = pd.read_parquet(paths.prepared / "phase1_sample.parquet")
        run_phase1(
            model=model,
            cases=phase1_cases,
            output_path=paths.raw / "phase1.parquet",
            repeats=int(cfg["active_profile"]["phase1"]["repeats"]),
            checkpoint_every=checkpoint_every,
            compression=compression,
        )

        persona_items = pd.read_parquet(paths.prepared / "persona_items.parquet")
        run_phase2(
            model=model,
            items=persona_items,
            personas=cfg["phase2"]["personas"],
            output_path=paths.raw / "phase2.parquet",
            repeats=int(cfg["active_profile"]["phase2"]["repeats"]),
            max_items=int(cfg["active_profile"]["phase2"]["max_items"]),
            seed=seed,
            checkpoint_every=checkpoint_every,
            compression=compression,
        )

        evasion_items = pd.read_parquet(paths.prepared / "evasion_items.parquet")
        run_phase3(
            model=model,
            items=evasion_items,
            output_path=paths.raw / "phase3.parquet",
            repeats=int(cfg["active_profile"]["phase3"]["repeats"]),
            max_items=int(cfg["active_profile"]["phase3"]["max_items"]),
            seed=seed,
            checkpoint_every=checkpoint_every,
            compression=compression,
        )

        pluralism_cases = pd.read_parquet(paths.prepared / "pluralism_cases.parquet")
        exemplars = pd.read_parquet(paths.prepared / "bias_exemplars.parquet")
        run_phase4_generation(
            model=model,
            cases=pluralism_cases,
            exemplars=exemplars,
            output_path=paths.raw / "phase4_generations.parquet",
            k=int(cfg["active_profile"]["phase4"]["k_retrieval"]),
            max_cases=int(cfg["active_profile"]["phase4"]["max_cases"]),
            seed=seed,
            max_new_tokens=int(cfg["models"]["generation"]["max_new_tokens"]),
            temperature=float(cfg["models"]["generation"]["temperature"]),
            checkpoint_every=max(4, checkpoint_every // 5),
            compression=compression,
        )
    finally:
        model.unload()
    return metadata


def run_cross_judging(cfg: dict, root: str | Path = ".") -> list[dict]:
    paths = ProjectPaths.from_root(root).ensure()
    generations = pd.read_parquet(paths.raw / "phase4_generations.parquet")
    metadata = []
    # Each evaluated model judges only the *other* model's generations.
    for judge_cfg in cfg["models"]["evaluated"]:
        judge = load_model(judge_cfg)
        metadata.append(judge.metadata())
        try:
            run_phase4_judging(
                judge_model=judge,
                generations=generations,
                output_path=paths.raw / "phase4_judgments.parquet",
                skip_same_model=True,
                seed=int(cfg["project"]["seed"]),
                checkpoint_every=max(4, int(cfg["storage"]["checkpoint_every"]) // 3),
                compression=cfg["storage"]["parquet_compression"],
            )
        finally:
            judge.unload()
    return metadata


def analyze_all(cfg: dict, root: str | Path = ".") -> dict[str, int]:
    paths = ProjectPaths.from_root(root).ensure()
    seed = int(cfg["project"]["seed"])
    counts = {}

    if (paths.raw / "phase1.parquet").exists():
        raw = pd.read_parquet(paths.raw / "phase1.parquet")
        paired, summary = save_phase1_analysis(
            raw,
            paths.derived,
            bootstrap_iterations=int(cfg["phase1"]["bootstrap_iterations"]),
            permutation_iterations=int(cfg["phase1"]["permutation_iterations"]),
            seed=seed,
        )
        summary = add_holm_to_phase1(summary)
        summary.to_csv(paths.derived / "phase1_summary.csv", index=False)
        write_parquet(summary, paths.derived / "phase1_summary.parquet")
        counts["phase1_valid_pairs"] = int(len(paired))

    if (paths.raw / "phase2.parquet").exists():
        raw = pd.read_parquet(paths.raw / "phase2.parquet")
        summary = save_phase2_analysis(
            raw,
            paths.derived,
            bootstrap_iterations=int(cfg["phase2"]["bootstrap_iterations"]),
            seed=seed,
        )
        counts["phase2_summary_rows"] = int(len(summary))

    if (paths.raw / "phase3.parquet").exists():
        raw = pd.read_parquet(paths.raw / "phase3.parquet")
        overall, per_class = save_phase3_analysis(raw, paths.derived)
        counts["phase3_models"] = int(len(overall))

    if (paths.raw / "phase4_judgments.parquet").exists():
        judged = pd.read_parquet(paths.raw / "phase4_judgments.parquet")
        summary = save_phase4_analysis(
            judged,
            paths.derived,
            bootstrap_iterations=int(cfg["phase4"]["bootstrap_iterations"]),
            seed=seed,
        )
        counts["phase4_summary_rows"] = int(len(summary))

    if (paths.raw / "phase4_generations.parquet").exists() and bool(cfg["phase4"].get("create_blinded_human_sheet", True)):
        gens = pd.read_parquet(paths.raw / "phase4_generations.parquet")
        build_blinded_human_sheet(
            gens,
            sheet_path=paths.derived / "phase4_human_rating_sheet.csv",
            key_path=paths.derived / "phase4_human_blind_key.parquet",
            seed=seed,
        )

    figures = make_figures(paths.derived, paths.figures, dpi=int(cfg["analysis"]["save_png_dpi"]))
    counts["figures"] = len(figures)
    write_json(counts, paths.manifests / "analysis_counts.json")
    return counts


def build_paper_bundle(cfg: dict, root: str | Path = ".") -> Path:
    paths = ProjectPaths.from_root(root).ensure()
    if paths.paper_bundle.exists():
        shutil.rmtree(paths.paper_bundle)
    paths.paper_bundle.mkdir(parents=True)

    for dirname, source in [
        ("raw", paths.raw),
        ("derived", paths.derived),
        ("figures", paths.figures),
        ("manifests", paths.manifests),
    ]:
        dest = paths.paper_bundle / dirname
        dest.mkdir()
        for file in source.glob("*"):
            if file.is_file():
                shutil.copy2(file, dest / file.name)

    write_json(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile": cfg["profile"],
            "description": "Evidence bundle for downstream research-paper writing. Preserve this directory unchanged after the final run.",
        },
        paths.paper_bundle / "bundle_manifest.json",
    )
    return paths.paper_bundle


def bootstrap_from_config(profile: str = "pilot", root: str | Path = ".") -> tuple[dict, ProjectPaths]:
    cfg = load_config(Path(root) / "config/default.yaml", Path(root) / f"config/{profile}.yaml")
    paths = ProjectPaths.from_root(root).ensure()
    save_run_manifest(paths.manifests / "run_manifest.json", config=cfg, root=root)
    return cfg, paths
