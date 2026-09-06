from __future__ import annotations

import hashlib

import pandas as pd


def stable_int(*parts: object, seed: int = 42) -> int:
    text = ":".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def deterministic_sample(df: pd.DataFrame, n: int, seed: int, key: str) -> pd.DataFrame:
    if n >= len(df):
        return df.copy().reset_index(drop=True)
    scores = df[key].map(lambda x: stable_int(x, seed=seed))
    return df.loc[scores.nsmallest(n).index].copy().reset_index(drop=True)


def make_phase1_sample(
    templates: pd.DataFrame,
    entities: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    n_templates: int,
    n_pairs: int,
    seed: int = 42,
) -> pd.DataFrame:
    selected_templates = deterministic_sample(templates, n_templates, seed, "template_id")
    selected_pairs = deterministic_sample(pairs, n_pairs, seed + 1, "pair_id")

    ent = entities.set_index("entity_id")
    records = []
    for pair in selected_pairs.itertuples(index=False):
        a = ent.loc[int(pair.entity_a_id)]
        b = ent.loc[int(pair.entity_b_id)]
        for t in selected_templates.itertuples(index=False):
            records.append({
                "pair_id": int(pair.pair_id),
                "pair_name": pair.pair_name,
                "template_id": int(t.template_id),
                "gold_sentiment": t.gold_sentiment,
                "domain": t.domain,
                "entity_a_id": int(pair.entity_a_id),
                "entity_a": a.display_name,
                "entity_b_id": int(pair.entity_b_id),
                "entity_b": b.display_name,
                "sentence_a": t.template_text.replace("{POLITICIAN}", a.display_name),
                "sentence_b": t.template_text.replace("{POLITICIAN}", b.display_name),
            })
    return pd.DataFrame(records)


def deterministic_option_swap(item_key: object, repeat_idx: int, seed: int = 42) -> bool:
    return bool(stable_int(item_key, repeat_idx, seed=seed) & 1)


def balanced_option_swap(item_key: object, repeat_idx: int, seed: int = 42) -> bool:
    """Guarantee one AB and one BA presentation in every adjacent repeat pair."""
    start_swapped = bool(stable_int(item_key, seed=seed) & 1)
    return bool(start_swapped ^ (repeat_idx % 2))
