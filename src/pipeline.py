from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import add_holm_to_phase1, make_figures
from .config import load_config
from .data_builder import prepare_universe
from .hf_runner import HFLocalModel
from .io_utils import write_json, write_parquet
from .paths import ProjectPaths
from .phase1_tsc import run_phase1, save_phase1_analysis
from .phase2_persona import run_phase2, run_phase2_capability, save_phase2_analysis
from .phase3_evasion import run_phase3, save_phase3_analysis
from .phase4_pluralism import (
    analyze_human_ratings,
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

    for name in ["persona_items", "evasion_items", "evasion_exemplars", "pluralism_cases", "bias_exemplars"]:
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
        "phase2_items": int(len(pd.read_parquet(paths.prepared / "persona_items.parquet"))),
        "phase3_test_items": int(len(pd.read_parquet(paths.prepared / "evasion_items.parquet"))),
        "phase3_fewshot_exemplars": int(len(pd.read_parquet(paths.prepared / "evasion_exemplars.parquet"))),
        "phase4_cases": int(len(pd.read_parquet(paths.prepared / "pluralism_cases.parquet"))),
        "uses_real_entities": bool(entities.get("is_real", pd.Series(dtype=bool)).astype(bool).any()),
    })
    write_json(stats, paths.manifests / "data_preparation.json")

    hashes = {file.name: file_sha256(file) for file in sorted(paths.prepared.glob("*.parquet"))}
    write_json(hashes, paths.manifests / "prepared_data_sha256.json")
    return stats


def load_model(model_cfg: dict) -> HFLocalModel:
    return HFLocalModel(
        model_cfg["id"],
        short_name=model_cfg.get("short_name"),
        revision=model_cfg.get("revision"),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        max_context_tokens=int(model_cfg.get("max_context_tokens", 4096)),
    )


def model_config_by_short_name(cfg: dict, short_name: str) -> dict:
    matches = [m for m in cfg["models"]["evaluated"] if m["short_name"] == short_name]
    if not matches:
        valid = ", ".join(m["short_name"] for m in cfg["models"]["evaluated"])
        raise KeyError(f"Unknown generator model {short_name!r}. Choose one of: {valid}")
    return matches[0]


def run_one_model(cfg: dict, model_cfg: dict, root: str | Path = ".") -> dict:
    paths = ProjectPaths.from_root(root).ensure()
    seed = int(cfg["project"]["seed"])
    checkpoint_every = int(cfg["storage"]["checkpoint_every"])
    compression = cfg["storage"]["parquet_compression"]

    model = load_model(model_cfg)
    metadata = model.metadata()
    try:
        run_phase1(
            model=model,
            cases=pd.read_parquet(paths.prepared / "phase1_sample.parquet"),
            output_path=paths.raw / "phase1.parquet",
            repeats=int(cfg["active_profile"]["phase1"]["repeats"]),
            checkpoint_every=checkpoint_every,
            compression=compression,
        )

        persona_items = pd.read_parquet(paths.prepared / "persona_items.parquet")
        run_phase2_capability(
            model=model,
            items=persona_items,
            personas=cfg["phase2"]["personas"],
            output_path=paths.raw / "phase2_capability.parquet",
            max_items=int(cfg["active_profile"]["phase2"]["max_anchor_items"]),
            seed=seed,
            checkpoint_every=checkpoint_every,
            compression=compression,
        )
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

        run_phase3(
            model=model,
            items=pd.read_parquet(paths.prepared / "evasion_items.parquet"),
            exemplars=pd.read_parquet(paths.prepared / "evasion_exemplars.parquet"),
            output_path=paths.raw / "phase3.parquet",
            repeats=int(cfg["active_profile"]["phase3"]["repeats"]),
            max_items=int(cfg["active_profile"]["phase3"]["max_items"]),
            seed=seed,
            checkpoint_every=checkpoint_every,
            compression=compression,
        )

        run_phase4_generation(
            model=model,
            cases=pd.read_parquet(paths.prepared / "pluralism_cases.parquet"),
            exemplars=pd.read_parquet(paths.prepared / "bias_exemplars.parquet"),
            output_path=paths.raw / "phase4_generations.parquet",
            k=int(cfg["active_profile"]["phase4"]["k_retrieval"]),
            max_cases=int(cfg["active_profile"]["phase4"]["max_cases"]),
            seed=seed,
            max_new_tokens=int(cfg["models"]["generation"]["max_new_tokens"]),
            temperature=float(cfg["models"]["generation"]["temperature"]),
            checkpoint_every=max(4, checkpoint_every // 2),
            compression=compression,
        )
    finally:
        model.unload()
    return metadata


def run_judge(cfg: dict, root: str | Path = ".") -> dict:
    paths = ProjectPaths.from_root(root).ensure()
    generations_path = paths.raw / "phase4_generations.parquet"
    if not generations_path.exists():
        raise FileNotFoundError("Phase 4 generations are missing. Run both generator models first.")
    generations = pd.read_parquet(generations_path)
    expected_generators = {m["short_name"] for m in cfg["models"]["evaluated"]}
    present_generators = set(generations.loc[generations["error"].isna(), "model"].unique())
    missing = expected_generators - present_generators
    if missing:
        raise RuntimeError(f"Judge run requires generations from both configured models. Missing: {sorted(missing)}")

    judge = load_model(cfg["models"]["judge"])
    metadata = judge.metadata()
    try:
        run_phase4_judging(
            judge_model=judge,
            generations=generations,
            output_path=paths.raw / "phase4_judgments.parquet",
            judge_repeats=int(cfg["active_profile"]["phase4"]["judge_repeats"]),
            max_parse_retries=int(cfg["phase4"]["judge_max_parse_retries"]),
            seed=int(cfg["project"]["seed"]),
            checkpoint_every=max(4, int(cfg["storage"]["checkpoint_every"]) // 2),
            compression=cfg["storage"]["parquet_compression"],
        )
    finally:
        judge.unload()
    return metadata


def _capability_gates(cfg: dict, paths: ProjectPaths) -> pd.DataFrame:
    rows = []
    p1_path = paths.raw / "phase1.parquet"
    if p1_path.exists():
        p1 = pd.read_parquet(p1_path)
        valid = p1[p1["error"].isna()]
        threshold = float(cfg["phase1"]["capability_gate"]["minimum_accuracy"])
        for model, g in valid.groupby("model"):
            value = float(g["correct"].mean())
            rows.append({"phase": "phase1", "model": model, "gate": "sentiment_accuracy", "value": value, "threshold": threshold, "passed": value >= threshold})

    cap_path = paths.derived / "phase2_capability.csv"
    if cap_path.exists():
        cap = pd.read_csv(cap_path)
        threshold = float(cfg["phase2"]["capability_gate"]["minimum_anchor_accuracy"])
        for model, g in cap.groupby("model"):
            value = float(g["anchor_accuracy"].min())
            rows.append({"phase": "phase2", "model": model, "gate": "minimum_persona_anchor_accuracy", "value": value, "threshold": threshold, "passed": value >= threshold})

    overall_path, per_class_path = paths.derived / "phase3_overall.csv", paths.derived / "phase3_per_class.csv"
    if overall_path.exists() and per_class_path.exists():
        overall, per_class = pd.read_csv(overall_path), pd.read_csv(per_class_path)
        f1_t = float(cfg["phase3"]["capability_gate"]["minimum_macro_f1"])
        recall_t = float(cfg["phase3"]["capability_gate"]["minimum_class_recall"])
        for _, r in overall.iterrows():
            rows.append({"phase": "phase3", "model": r["model"], "gate": "macro_f1", "value": float(r["macro_f1"]), "threshold": f1_t, "passed": float(r["macro_f1"]) >= f1_t})
        for model, g in per_class.groupby("model"):
            value = float(g["recall"].min())
            rows.append({"phase": "phase3", "model": model, "gate": "minimum_class_recall", "value": value, "threshold": recall_t, "passed": value >= recall_t})

    quality_path = paths.derived / "phase4_judge_quality.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        parse_t = float(cfg["phase4"]["capability_gate"]["minimum_parse_success_rate"])
        order_t = float(cfg["phase4"]["capability_gate"]["minimum_order_consistency"])
        for _, r in quality.iterrows():
            label = f"{r['judge_model']} on {r['generator_model']}"
            rows.append({"phase": "phase4", "model": label, "gate": "judge_parse_success_rate", "value": float(r["parse_success_rate"]), "threshold": parse_t, "passed": float(r["parse_success_rate"]) >= parse_t})
            if pd.notna(r["order_consistency"]):
                rows.append({"phase": "phase4", "model": label, "gate": "judge_order_consistency", "value": float(r["order_consistency"]), "threshold": order_t, "passed": float(r["order_consistency"]) >= order_t})
    return pd.DataFrame(rows)


def analyze_all(cfg: dict, root: str | Path = ".") -> dict[str, int]:
    paths = ProjectPaths.from_root(root).ensure()
    seed = int(cfg["project"]["seed"])
    counts: dict[str, int] = {}

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

    if (paths.raw / "phase2.parquet").exists() and (paths.raw / "phase2_capability.parquet").exists():
        summary, capability = save_phase2_analysis(
            pd.read_parquet(paths.raw / "phase2.parquet"),
            pd.read_parquet(paths.raw / "phase2_capability.parquet"),
            paths.derived,
            bootstrap_iterations=int(cfg["phase2"]["bootstrap_iterations"]),
            seed=seed,
        )
        counts["phase2_summary_rows"] = int(len(summary))
        counts["phase2_capability_rows"] = int(len(capability))

    if (paths.raw / "phase3.parquet").exists():
        overall, _ = save_phase3_analysis(pd.read_parquet(paths.raw / "phase3.parquet"), paths.derived)
        counts["phase3_models"] = int(len(overall))

    if (paths.raw / "phase4_judgments.parquet").exists():
        summary, quality = save_phase4_analysis(
            pd.read_parquet(paths.raw / "phase4_judgments.parquet"),
            paths.derived,
            bootstrap_iterations=int(cfg["phase4"]["bootstrap_iterations"]),
            seed=seed,
        )
        counts["phase4_summary_rows"] = int(len(summary))
        counts["phase4_quality_rows"] = int(len(quality))

    if (paths.raw / "phase4_generations.parquet").exists() and bool(cfg["phase4"].get("create_blinded_human_sheet", True)):
        gens = pd.read_parquet(paths.raw / "phase4_generations.parquet")
        sheet_path = paths.derived / "phase4_human_rating_sheet.csv"
        key_path = paths.derived / "phase4_human_blind_key.parquet"
        build_blinded_human_sheet(gens, sheet_path=sheet_path, key_path=key_path, seed=seed)
        completed = paths.derived / "phase4_human_ratings_completed.csv"
        if completed.exists():
            merged, reliability = analyze_human_ratings(pd.read_csv(completed), pd.read_parquet(key_path))
            write_parquet(merged, paths.derived / "phase4_human_ratings_unblinded.parquet")
            reliability.to_csv(paths.derived / "phase4_human_reliability.csv", index=False)

    gates = _capability_gates(cfg, paths)
    if not gates.empty:
        gates.to_csv(paths.derived / "capability_gates.csv", index=False)
        write_parquet(gates, paths.derived / "capability_gates.parquet")
        counts["capability_gates_failed"] = int((~gates["passed"]).sum())

    figures = make_figures(paths.derived, paths.figures, dpi=int(cfg["analysis"]["save_png_dpi"]))
    counts["figures"] = len(figures)
    write_json(counts, paths.manifests / "analysis_counts.json")
    return counts


def build_paper_bundle(cfg: dict, root: str | Path = ".") -> Path:
    paths = ProjectPaths.from_root(root).ensure()
    if paths.paper_bundle.exists() and not paths.paper_bundle.is_symlink():
        shutil.rmtree(paths.paper_bundle)
    paths.paper_bundle.mkdir(parents=True, exist_ok=True)

    for dirname, source in [("raw", paths.raw), ("derived", paths.derived), ("figures", paths.figures), ("manifests", paths.manifests)]:
        dest = paths.paper_bundle / dirname
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for file in source.glob("*"):
            if file.is_file():
                shutil.copy2(file, dest / file.name)

    write_json({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": cfg["profile"],
        "description": "Evidence bundle for downstream research-paper writing. Preserve unchanged after the final run.",
    }, paths.paper_bundle / "bundle_manifest.json")
    return paths.paper_bundle


def bootstrap_from_config(profile: str = "pilot", root: str | Path = ".") -> tuple[dict, ProjectPaths]:
    cfg = load_config(Path(root) / "config/default.yaml", Path(root) / f"config/{profile}.yaml")
    paths = ProjectPaths.from_root(root).ensure()
    save_run_manifest(paths.manifests / "run_manifest.json", config=cfg, root=root)
    return cfg, paths
