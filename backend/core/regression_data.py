"""
Training data for the year regressor.

Compared with core.dataset (still used by the classifiers), this module:
  - samples comments at random instead of taking the first rows of each file;
  - undoes formatting that the Reddit dumps add and that differs by year;
  - drops deleted, removed, empty and self-declared bot comments;
  - holds out whole months for validation and test, so comments posted in the
    same minutes are never on both sides of a split.
"""

import html
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

CSV_COLUMNS = ["subreddit", "subreddit_id", "body", "date_created_utc"]
PLACEHOLDERS = {"", "[deleted]", "[removed]"}
BOT_PATTERN = r"\bI am a bot\b|\bI['’]m a bot\b|this action was performed automatically"


def normalize_body(text: str) -> str:
    """
    Undo formatting that would let a model date a comment without reading it.

    Carriage returns are common only in the oldest dumps, and HTML escapes only
    before mid-2023. Neither appears in text typed into the demo. Escapes are
    undone twice because some bodies are escaped twice ("&amp;#x200B;").
    """
    text = html.unescape(html.unescape(text))
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", "")


def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize bodies and drop rows that carry no usable text."""
    df = df[df["body"].notna()]
    df = df.assign(body=df["body"].astype(str).map(normalize_body))
    body = df["body"]
    placeholder = body.str.strip().isin(PLACEHOLDERS)
    bot = body.str.contains(BOT_PATTERN, case=False, regex=True)
    return df[~placeholder & ~bot]


def allocate_evenly(available: Dict[int, int], target: int) -> Dict[int, int]:
    """
    Share `target` rows between months as evenly as their supply allows.

    A month with too few rows gives what it has; the rest is shared by the others.
    """
    allocation = {}
    remaining = target
    # Smallest supply first, so a shortfall is passed on to the larger months.
    pending = sorted(available, key=lambda month: (available[month], month))
    while pending:
        month = pending.pop(0)
        share = remaining // (len(pending) + 1)
        allocation[month] = min(available[month], share)
        remaining -= allocation[month]
    return {month: allocation[month] for month in available}


def _load_year(data_dir: str, year: int, rng: np.random.RandomState) -> pd.DataFrame:
    """All usable comments of one year, shuffled, each distinct body once."""
    months = []
    for month in range(1, 13):
        path = os.path.join(data_dir, f"RC_{year}-{month:02d}.csv")
        df = pd.read_csv(path, header=None, names=CSV_COLUMNS, usecols=["subreddit", "body"])
        months.append(df.assign(month=month))
    df = clean_comments(pd.concat(months, ignore_index=True))
    # Shuffle before de-duplicating so a repeated comment is not always assigned
    # to January. Repeats across different years are kept on purpose: "lol" is
    # posted every year, and keeping one copy would tie it to a single year.
    df = df.iloc[rng.permutation(len(df))]
    return df.drop_duplicates("body")


def _sample(df: pd.DataFrame, target: int) -> pd.DataFrame:
    """Take `target` rows from an already shuffled frame, spread over its months."""
    by_month = {month: group for month, group in df.groupby("month")}
    counts = allocate_evenly({m: len(g) for m, g in by_month.items()}, target)
    return pd.concat([by_month[m].head(n) for m, n in counts.items()])


def load_regression_splits(
    data_dir: str = "data/sampled_comments",
    start_year: int = 2006,
    end_year: int = 2024,
    train_per_year: int = 40_000,
    eval_per_year: int = 500,
    val_month: int = 3,
    test_month: int = 9,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build train, validation and test frames, balanced by year.

    Validation and test come from one held-out month each; train comes from the
    other ten. Every year needs all twelve monthly files.

    Returns:
        Three DataFrames with columns subreddit, body, month, year, label,
        where label is the year index (year - start_year).
    """
    if val_month == test_month:
        raise ValueError("Validation month and test month must differ")

    splits = {"train": [], "val": [], "test": []}
    for year in range(start_year, end_year + 1):
        df = _load_year(data_dir, year, np.random.RandomState(seed + year))
        held_out = df["month"].isin([val_month, test_month])
        picked = {
            "train": _sample(df[~held_out], train_per_year),
            "val": _sample(df[df["month"] == val_month], eval_per_year),
            "test": _sample(df[df["month"] == test_month], eval_per_year),
        }
        for name, part in picked.items():
            splits[name].append(part.assign(year=year, label=year - start_year))
        print(f"  {year}: " + ", ".join(f"{name} {len(part):,}" for name, part in picked.items()))

    return tuple(pd.concat(splits[name], ignore_index=True) for name in ("train", "val", "test"))
