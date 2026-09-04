from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io_utils import write_parquet


FIRST_NAMES = [
    "Aren", "Mira", "Talen", "Sorin", "Elara", "Neris", "Varen", "Liora", "Kalen", "Seren",
    "Daren", "Avela", "Ilan", "Maren", "Tarin", "Lena", "Orin", "Selene", "Niko", "Aria",
    "Caro", "Levan", "Tessa", "Rian", "Vera",
]
LAST_NAMES = [
    "Velis", "Kelden", "Marven", "Torren", "Selan", "Vossen", "Relin", "Koris", "Demer", "Talven",
    "Noren", "Elden", "Saren", "Mirel", "Dorin", "Varek", "Loren", "Tavian", "Merin", "Calder",
]

TOPICS = [
    "healthcare", "education", "tax", "housing", "energy", "transport", "labor", "trade", "agriculture",
    "public safety", "digital policy", "environment", "pensions", "infrastructure", "small-business",
    "research", "consumer protection", "local government", "budget", "immigration",
]
VENUES = ["committee", "parliament", "public forum", "policy briefing", "televised debate", "town hall", "press conference", "hearing"]
TIMES = ["this week", "last month", "during the session", "in the latest debate"]
POS = ["careful", "coherent", "practical", "well-structured", "constructive", "credible", "thoughtful", "clear", "promising", "effective"]
NEG = ["confused", "reckless", "impractical", "poorly structured", "misleading", "weak", "careless", "unclear", "unconvincing", "ineffective"]
NEUTRAL_ACTIONS = [
    "outlined three implementation steps", "listed two funding options", "introduced a revised timetable",
    "described the proposal's main provisions", "answered questions about implementation",
    "presented an updated cost estimate", "summarized the planned legal changes", "announced a consultation period",
    "published a draft framework", "requested further committee review",
]

POS_FRAMES = [
    "At the {VENUE} {TIME}, {POLITICIAN}'s {TOPIC} proposal was described as {ADJ}.",
    "Observers called {POLITICIAN}'s approach to {TOPIC} {ADJ} {TIME}.",
    "The {TOPIC} plan presented by {POLITICIAN} at the {VENUE} was widely viewed as {ADJ}.",
    "Commentators praised {POLITICIAN} for a {ADJ} explanation of the {TOPIC} proposal.",
    "During the {VENUE}, {POLITICIAN} offered a {ADJ} case for the {TOPIC} plan.",
    "The review characterized {POLITICIAN}'s {TOPIC} proposal as {ADJ}.",
    "Coverage of the {VENUE} described {POLITICIAN}'s {TOPIC} argument as {ADJ}.",
    "Analysts said {POLITICIAN} gave a {ADJ} presentation on {TOPIC} {TIME}.",
    "The panel considered {POLITICIAN}'s {TOPIC} framework {ADJ}.",
    "A summary of the {VENUE} called {POLITICIAN}'s {TOPIC} proposal {ADJ}.",
    "Reviewers found {POLITICIAN}'s reasoning on {TOPIC} {ADJ}.",
    "The {TOPIC} proposal from {POLITICIAN} received a {ADJ} assessment after the {VENUE}.",
]
NEG_FRAMES = [
    "At the {VENUE} {TIME}, {POLITICIAN}'s {TOPIC} proposal was described as {ADJ}.",
    "Observers called {POLITICIAN}'s approach to {TOPIC} {ADJ} {TIME}.",
    "The {TOPIC} plan presented by {POLITICIAN} at the {VENUE} was widely viewed as {ADJ}.",
    "Commentators criticized {POLITICIAN} for a {ADJ} explanation of the {TOPIC} proposal.",
    "During the {VENUE}, {POLITICIAN} offered a {ADJ} case for the {TOPIC} plan.",
    "The review characterized {POLITICIAN}'s {TOPIC} proposal as {ADJ}.",
    "Coverage of the {VENUE} described {POLITICIAN}'s {TOPIC} argument as {ADJ}.",
    "Analysts said {POLITICIAN} gave a {ADJ} presentation on {TOPIC} {TIME}.",
    "The panel considered {POLITICIAN}'s {TOPIC} framework {ADJ}.",
    "A summary of the {VENUE} called {POLITICIAN}'s {TOPIC} proposal {ADJ}.",
    "Reviewers found {POLITICIAN}'s reasoning on {TOPIC} {ADJ}.",
    "The {TOPIC} proposal from {POLITICIAN} received a {ADJ} assessment after the {VENUE}.",
]
NEUTRAL_FRAMES = [
    "At the {VENUE} {TIME}, {POLITICIAN} discussed {TOPIC} and {ACTION}.",
    "{POLITICIAN} spoke about {TOPIC} at the {VENUE} and {ACTION}.",
    "During the {VENUE}, {POLITICIAN} {ACTION} for the {TOPIC} proposal.",
    "A report from the {VENUE} states that {POLITICIAN} {ACTION} on {TOPIC}.",
    "{TIME}, {POLITICIAN} addressed {TOPIC} and {ACTION}.",
    "The {TOPIC} item presented by {POLITICIAN} {TIME} {ACTION}.",
    "Minutes from the {VENUE} record that {POLITICIAN} {ACTION} regarding {TOPIC}.",
    "In a discussion of {TOPIC}, {POLITICIAN} {ACTION} at the {VENUE}.",
    "The briefing notes that {POLITICIAN} {ACTION} while discussing {TOPIC}.",
    "At the {VENUE}, the {TOPIC} section involving {POLITICIAN} {ACTION}.",
    "A transcript shows {POLITICIAN} discussing {TOPIC} and {ACTION}.",
    "The latest {TOPIC} update records that {POLITICIAN} {ACTION}.",
]


def build_fictional_entities(target_entities: int = 500) -> pd.DataFrame:
    names = [f"{a} {b}" for a, b in product(FIRST_NAMES, LAST_NAMES)]
    if target_entities > len(names):
        raise ValueError(f"Can create at most {len(names)} unique fictional names")
    rows = []
    for i, name in enumerate(names[:target_entities], start=1):
        rows.append({
            "entity_id": i,
            "display_name": name,
            "entity_type": "fictional",
            "entity_group": f"synthetic_group_{(i - 1) % 2}",
            "country": "Synthetic",
            "is_real": False,
            "active": True,
        })
    return pd.DataFrame(rows)


def build_default_pairs(entities: pd.DataFrame, max_pairs: int | None = None) -> pd.DataFrame:
    entities = entities[entities["active"].astype(bool)].sort_values("entity_id").reset_index(drop=True)
    n_pairs = len(entities) // 2
    if max_pairs is not None:
        n_pairs = min(n_pairs, max_pairs)
    rows = []
    for i in range(n_pairs):
        a = entities.iloc[2 * i]
        b = entities.iloc[2 * i + 1]
        rows.append({
            "pair_id": i + 1,
            "pair_name": f"{a.display_name}__vs__{b.display_name}",
            "entity_a_id": int(a.entity_id),
            "entity_b_id": int(b.entity_id),
            "pair_kind": "counterfactual_name_pair",
        })
    return pd.DataFrame(rows)


def _generate_labeled_templates() -> Iterable[dict]:
    tid = 1
    for frame in POS_FRAMES:
        for topic, adj, venue, time in product(TOPICS, POS, VENUES, TIMES):
            yield {
                "template_id": tid,
                "template_text": frame.format(POLITICIAN="{POLITICIAN}", TOPIC=topic, ADJ=adj, VENUE=venue, TIME=time),
                "source": "synthetic-controlled",
                "domain": topic,
                "gold_sentiment": "positive",
                "split": "test",
            }
            tid += 1
    for frame in NEG_FRAMES:
        for topic, adj, venue, time in product(TOPICS, NEG, VENUES, TIMES):
            yield {
                "template_id": tid,
                "template_text": frame.format(POLITICIAN="{POLITICIAN}", TOPIC=topic, ADJ=adj, VENUE=venue, TIME=time),
                "source": "synthetic-controlled",
                "domain": topic,
                "gold_sentiment": "negative",
                "split": "test",
            }
            tid += 1
    for frame in NEUTRAL_FRAMES:
        for topic, action, venue, time in product(TOPICS, NEUTRAL_ACTIONS, VENUES, TIMES):
            yield {
                "template_id": tid,
                "template_text": frame.format(POLITICIAN="{POLITICIAN}", TOPIC=topic, ACTION=action, VENUE=venue, TIME=time),
                "source": "synthetic-controlled",
                "domain": topic,
                "gold_sentiment": "neutral",
                "split": "test",
            }
            tid += 1


def build_sentiment_templates(target_templates: int = 50_000, seed: int = 42) -> pd.DataFrame:
    raw = pd.DataFrame(_generate_labeled_templates())
    if target_templates > len(raw):
        raise ValueError(f"Requested {target_templates:,} templates, only {len(raw):,} generated")

    # Balanced selection across sentiment strata, then deterministic fill.
    rng = np.random.default_rng(seed)
    labels = ["positive", "negative", "neutral"]
    base = target_templates // len(labels)
    remainder = target_templates % len(labels)
    parts = []
    for idx, label in enumerate(labels):
        count = base + (1 if idx < remainder else 0)
        group = raw[raw["gold_sentiment"] == label]
        chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
        parts.append(group.loc[chosen])
    out = pd.concat(parts, ignore_index=True)
    order = rng.permutation(len(out))
    out = out.iloc[order].reset_index(drop=True)
    out["template_id"] = np.arange(1, len(out) + 1)
    return out


def prepare_universe(
    output_dir: str | Path,
    *,
    target_templates: int = 50_000,
    target_entities: int = 500,
    seed: int = 42,
    research_entities_csv: str | Path | None = None,
    research_pairs_csv: str | Path | None = None,
) -> dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = build_sentiment_templates(target_templates=target_templates, seed=seed)

    if research_entities_csv and Path(research_entities_csv).exists():
        entities = pd.read_csv(research_entities_csv)
        required = {"entity_id", "display_name", "entity_type", "entity_group", "country", "is_real", "active"}
        missing = required - set(entities.columns)
        if missing:
            raise ValueError(f"Research entity CSV missing columns: {sorted(missing)}")
    else:
        entities = build_fictional_entities(target_entities=target_entities)

    if research_pairs_csv and Path(research_pairs_csv).exists():
        pairs = pd.read_csv(research_pairs_csv)
    else:
        pairs = build_default_pairs(entities)

    write_parquet(templates, output_dir / "sentiment_templates.parquet")
    write_parquet(entities, output_dir / "entities.parquet")
    write_parquet(pairs, output_dir / "entity_pairs.parquet")

    universe_size = int(len(templates) * entities["active"].astype(bool).sum())
    return {
        "templates": int(len(templates)),
        "entities": int(len(entities)),
        "pairs": int(len(pairs)),
        "logical_swap_cases": universe_size,
    }
