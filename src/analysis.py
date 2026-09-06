from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm family-wise-error adjusted p-values, preserving the original index."""
    p = p_values.astype(float)
    valid = p.dropna()
    m = len(valid)
    if m == 0:
        return pd.Series(np.nan, index=p.index)
    order = valid.sort_values().index.tolist()
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * valid.loc[idx])
        running = max(running, value)
        adjusted.loc[idx] = running
    return adjusted


def add_holm_to_phase1(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["permutation_p_holm"] = np.nan
    for model, idx in out.groupby("model").groups.items():
        out.loc[idx, "permutation_p_holm"] = holm_adjust(out.loc[idx, "permutation_p"]).to_numpy()
    return out


def make_figures(derived_dir: str | Path, figures_dir: str | Path, dpi: int = 160) -> list[Path]:
    import matplotlib.pyplot as plt

    derived_dir = Path(derived_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    p1 = derived_dir / "phase1_summary.csv"
    if p1.exists():
        df = pd.read_csv(p1)
        if not df.empty:
            for model, g in df.groupby("model"):
                g = g.sort_values("mean_signed_delta")
                fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(g))))
                y = np.arange(len(g))
                xerr = np.vstack([g["mean_signed_delta"] - g["ci_low"], g["ci_high"] - g["mean_signed_delta"]])
                ax.errorbar(g["mean_signed_delta"], y, xerr=xerr, fmt="o", capsize=3)
                ax.axvline(0, linewidth=1)
                ax.set_yticks(y, labels=g["pair_name"])
                ax.set_xlabel("Mean signed sentiment delta (A - B)")
                ax.set_title(f"Phase 1 entity-swap sensitivity — {model}")
                fig.tight_layout()
                path = figures_dir / f"phase1_delta_{model}.png"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                created.append(path)

    p2 = derived_dir / "phase2_summary.csv"
    if p2.exists():
        df = pd.read_csv(p2)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            labels = (df["model"] + " | " + df["persona_name"]).tolist()
            y = np.arange(len(df))
            center = df["mean_aligned_log_odds_shift"]
            xerr = np.vstack([center - df["ci_low"], df["ci_high"] - center])
            ax.errorbar(center, y, xerr=xerr, fmt="o", capsize=3)
            ax.axvline(0, linewidth=1)
            ax.set_yticks(y, labels=labels)
            ax.set_xlabel("Aligned persona-induced log-odds shift")
            ax.set_title("Phase 2 persona susceptibility")
            fig.tight_layout()
            path = figures_dir / "phase2_persona_shift.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            created.append(path)

    p3 = derived_dir / "phase3_overall.csv"
    if p3.exists():
        df = pd.read_csv(p3)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(df))
            ax.bar(x, df["macro_f1"])
            ax.set_xticks(x, labels=df["model"], rotation=15)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Macro F1")
            ax.set_title("Phase 3 evasion classification")
            fig.tight_layout()
            path = figures_dir / "phase3_macro_f1.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            created.append(path)

    p4 = derived_dir / "phase4_judge_summary.csv"
    if p4.exists():
        df = pd.read_csv(p4)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = (df["generator_model"] + " | " + df["metric"]).tolist()
            x = np.arange(len(df))
            ax.bar(x, df["mean_improvement"])
            ax.axhline(0, linewidth=1)
            ax.set_xticks(x, labels=labels, rotation=75, ha="right")
            ax.set_ylabel("Mean paired improvement")
            ax.set_title("Phase 4 adaptive-pluralism mitigation")
            fig.tight_layout()
            path = figures_dir / "phase4_mitigation.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            created.append(path)
    return created
