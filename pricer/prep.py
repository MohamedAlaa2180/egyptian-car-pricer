"""Load, parse, clean, and split the Egyptian cars HF dataset.

Filter rules (v1) — see README and notebooks/01_inspect_and_prep.ipynb.
Does not push to the Hub.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from .items import PREFIX, CarItem

SOURCE_DATASET = "mo-hug-me/Egyptian_cars_price_prediction"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

YEAR_MIN, YEAR_MAX = 1990, 2027
MILEAGE_MAX = 800_000
PRICE_MIN, PRICE_MAX = 50_000, 20_000_000
ZERO_KM_MAX_YEAR = 2023  # drop 0 km if year <= this (missing data, not "new")
IQR_MIN_GROUP = 8
MEDIAN_MIN_GROUP = 12
MEDIAN_LOW, MEDIAN_HIGH = 0.25, 4.0

TEST_SIZE = 1_000
VAL_SIZE = 1_000
LITE_TRAIN_SIZE = 8_000
SPLIT_SEED = 42

PROMPT_RE = re.compile(
    r"Brand:\s*(?P<brand>.+?),\s*"
    r"Model:\s*(?P<model>.+?),\s*"
    r"Year:\s*(?P<year>\d{4}),\s*"
    r"Mileage:\s*(?P<mileage>[\d,]+)\s*km,\s*"
    r"Fuel:\s*(?P<fuel>.+?),\s*"
    r"Transmission:\s*(?P<transmission>.+?),\s*"
    r"Description:\s*(?P<description>.+?)\s*"
    r"Price is EGP",
    re.IGNORECASE | re.DOTALL,
)

# Flag only in v1 — not dropped.
GEN_RULES = [
    # (model substring, min_year inclusive, max_year inclusive)
    ("wrangler 4xe", 2021, YEAR_MAX),
    ("octavia a8", 2020, YEAR_MAX),
    ("accent rb", 2011, 2019),
]


def load_source() -> pd.DataFrame:
    """Pool Hub train and validation.

    The Hub test split is 4,999 rows whose completion is ``0`` on every row,
    so those ads have no asking price to learn. The Hub train/validation cut
    was also made before our filters, so we clean the pooled labeled rows
    and then cut our own train / val / test.
    """
    ds = load_dataset(SOURCE_DATASET)
    frames = []
    for split in ("train", "validation"):
        frame = ds[split].to_pandas()
        frame["hub_split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def parse_row(row: dict) -> dict | None:
    prompt = str(row.get("prompt") or "")
    completion = str(row.get("completion") or "").strip()
    match = PROMPT_RE.search(prompt)
    if not match:
        return None
    if not re.fullmatch(r"\d+", completion):
        return None
    fields = match.groupdict()
    mileage = int(fields["mileage"].replace(",", ""))
    year = int(fields["year"])
    price = float(completion)
    brand = fields["brand"].strip()
    model = fields["model"].strip()
    return {
        "brand": brand,
        "model": model,
        "year": year,
        "mileage": mileage,
        "fuel": fields["fuel"].strip(),
        "transmission": fields["transmission"].strip(),
        "description": fields["description"].strip(),
        "price": price,
        "prompt": normalize_prompt(prompt),
        "completion": completion,
        "gen_suspect": is_gen_suspect(model, year),
    }


def normalize_prompt(prompt: str) -> str:
    if PREFIX in prompt:
        return prompt.split(PREFIX)[0] + PREFIX
    return prompt.rstrip() + "\n" + PREFIX


def is_gen_suspect(model: str, year: int) -> bool:
    name = model.lower()
    for needle, lo, hi in GEN_RULES:
        if needle in name and not (lo <= year <= hi):
            return True
    return False


def parse_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    parsed, failed = [], 0
    for row in raw.to_dict(orient="records"):
        item = parse_row(row)
        if item is None:
            failed += 1
        else:
            parsed.append(item)
    return pd.DataFrame(parsed), failed


def _is_other(value: str) -> bool:
    return value.strip().casefold() == "other"


def apply_hard_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, dropped_with_reason). First matching rule wins."""
    reasons = []

    def reason_for(row) -> str | None:
        if _is_other(row.brand) or _is_other(row.model):
            return "other_brand_or_model"
        if not (YEAR_MIN <= row.year <= YEAR_MAX):
            return "year_out_of_range"
        if row.mileage < 0 or row.mileage > MILEAGE_MAX:
            return "mileage_out_of_range"
        if row.price < PRICE_MIN or row.price > PRICE_MAX:
            return "price_global_clip"
        if row.mileage == 0 and row.year <= ZERO_KM_MAX_YEAR:
            return "zero_km_old_car"
        return None

    for row in df.itertuples(index=False):
        reasons.append(reason_for(row))
    out = df.copy()
    out["drop_reason"] = reasons
    dropped = out[out["drop_reason"].notna()].copy()
    kept = out[out["drop_reason"].isna()].drop(columns=["drop_reason"])
    return kept, dropped


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    kept = df.drop_duplicates(subset=["prompt", "completion"], keep="first")
    return kept, before - len(kept)


def flag_price_outliers(df: pd.DataFrame) -> pd.Series:
    """True = keep. Within (brand, model, year) IQR, else (brand, model) median band."""
    keep = pd.Series(True, index=df.index)
    df = df.copy()
    df["_group"] = (
        df["brand"].str.casefold()
        + "|"
        + df["model"].str.casefold()
        + "|"
        + df["year"].astype(str)
    )
    df["_parent"] = df["brand"].str.casefold() + "|" + df["model"].str.casefold()

    for _, g in df.groupby("_group"):
        if len(g) < IQR_MIN_GROUP:
            continue
        q1, q3 = g["price"].quantile(0.25), g["price"].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        keep.loc[g.index] = g["price"].between(lo, hi)

    for _, g in df.groupby("_parent"):
        if len(g) < MEDIAN_MIN_GROUP:
            continue
        median = g["price"].median()
        if median <= 0:
            continue
        lo, hi = median * MEDIAN_LOW, median * MEDIAN_HIGH
        # only apply to rows not already covered by a large year-group IQR
        small_year_groups = g.groupby("_group").filter(lambda x: len(x) < IQR_MIN_GROUP)
        keep.loc[small_year_groups.index] &= small_year_groups["price"].between(lo, hi)

    return keep


def apply_price_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_mask = flag_price_outliers(df)
    kept = df.loc[keep_mask].copy()
    dropped = df.loc[~keep_mask].copy()
    dropped["drop_reason"] = "price_outlier"
    return kept, dropped


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parsed, n_parse_fail = parse_frame(raw)
    report: dict = {
        "source_rows": int(len(raw)),
        "parse_failed": int(n_parse_fail),
        "parsed": int(len(parsed)),
    }
    kept, dropped_hard = apply_hard_filters(parsed)
    report["hard_drop"] = dict(Counter(dropped_hard["drop_reason"]))
    kept, n_dup = drop_exact_duplicates(kept)
    report["exact_duplicates"] = int(n_dup)
    kept, dropped_out = apply_price_outliers(kept)
    report["price_outliers"] = int(len(dropped_out))
    report["kept"] = int(len(kept))
    report["gen_suspect_kept"] = int(kept["gen_suspect"].sum()) if len(kept) else 0
    dropped = pd.concat([dropped_hard, dropped_out], ignore_index=True)
    return kept.reset_index(drop=True), {"report": report, "dropped": dropped}


def _stratify_labels(df: pd.DataFrame, min_count: int) -> pd.Series | None:
    counts = df["brand"].value_counts()
    labels = df["brand"].where(df["brand"].map(counts) >= min_count, other="_rare")
    if labels.nunique() < 2 or labels.value_counts().min() < 2:
        return None
    return labels


def split_frame(
    df: pd.DataFrame,
    test_size: int = TEST_SIZE,
    val_size: int = VAL_SIZE,
    seed: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < test_size + val_size + 100:
        raise ValueError(f"Not enough rows to split: {len(df)}")

    strat = _stratify_labels(df, min_count=5)
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )
    strat_tv = _stratify_labels(train_val, min_count=5)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=seed,
        stratify=strat_tv,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def lite_train(train: pd.DataFrame, n: int = LITE_TRAIN_SIZE, seed: int = SPLIT_SEED) -> pd.DataFrame:
    if len(train) <= n:
        return train.copy()
    strat = _stratify_labels(train, min_count=5)
    subset, _ = train_test_split(train, train_size=n, random_state=seed, stratify=strat)
    return subset.reset_index(drop=True)


def to_items(df: pd.DataFrame) -> list[CarItem]:
    return [CarItem.model_validate(row) for row in df.to_dict(orient="records")]


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    report: dict,
    lite: pd.DataFrame | None = None,
    data_dir: Path = DATA_DIR,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(data_dir / "train.parquet", index=False)
    val.to_parquet(data_dir / "val.parquet", index=False)
    test.to_parquet(data_dir / "test.parquet", index=False)
    if lite is not None:
        lite.to_parquet(data_dir / "train_lite.parquet", index=False)
    (data_dir / "prep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    DatasetDict(
        {
            "train": Dataset.from_pandas(train[["prompt", "completion"]], preserve_index=False),
            "val": Dataset.from_pandas(val[["prompt", "completion"]], preserve_index=False),
            "test": Dataset.from_pandas(test[["prompt", "completion"]], preserve_index=False),
        }
    ).save_to_disk(str(data_dir / "hf_splits"))
    return data_dir
