#!/usr/bin/env python3
"""Prepare the post-June 2026 location dataset for the in-use pipeline.

Source is an Excel sheet with `Comment` + `Location 1/2/3` but no sentiment
columns. This script repairs known text corruption, runs model sentiment, and
writes a CSV matching the schema of the other `data/in_use/*.completed.csv`
files so the location analytics and both map builders can consume it directly.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from sentiment_inference import predict_sentiment  # noqa: E402

LABEL_TO_CANON = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "sarcastic_negative": "Sarcastic_negative",
}

# Upstream export dropped "দিন" after "উদ্", leaving a dangling hasant that
# splits names like সালাউদ্দিন / মহিউদ্দিন into an orphan syllable. Restore it.
UDDIN_POSSESSIVE_RE = re.compile(r"উদ্\s+ের")
UDDIN_RE = re.compile(r"উদ্(?=\s|[।,.!?]|$)")

# Stray hasant + space inside single words, verified case by case.
LITERAL_FIXES = {
    "দে্ শ": "দেশ",
    "আল্লাহ্ র": "আল্লাহর",
}


def repair_text(value: object) -> str:
    text = str(value)
    for broken, fixed in LITERAL_FIXES.items():
        text = text.replace(broken, fixed)
    text = UDDIN_POSSESSIVE_RE.sub("উদ্দিনের", text)
    text = UDDIN_RE.sub("উদ্দিন", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/data_location_cleaned_sheet_2.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/in_use/post_june_2026_with_location.annotated.completed.csv"),
    )
    parser.add_argument("--model", type=Path, default=Path("models/sentiment_model_second_pass_v2.joblib"))
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.5,
        help="Rows below this model confidence are flagged needs_review.",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    required = {"Comment", "Location 1", "Location 2", "Location 3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.input} is missing columns: {sorted(missing)}")

    out = df[["Comment", "Location 1", "Location 2", "Location 3"]].copy()

    repaired = out["Comment"].map(repair_text)
    n_repaired = int((repaired != out["Comment"].astype(str)).sum())
    out["Comment"] = repaired

    if not args.model.exists():
        raise FileNotFoundError(f"Sentiment model not found: {args.model}")

    preds = predict_sentiment(out["Comment"], model_path=args.model)
    out["Sentiment"] = preds["sentiment_label"].map(LABEL_TO_CANON).values
    out["sentiment_confidence"] = preds["sentiment_confidence"].values
    out["sentiment_source"] = "model"
    out["needs_review"] = out["sentiment_confidence"] < args.review_threshold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Saved: {args.output}")
    print(f"  rows: {len(out)}")
    print(f"  text rows repaired: {n_repaired}")
    print(f"  needs_review (conf < {args.review_threshold}): {int(out['needs_review'].sum())}")
    print(f"  sentiment mix: {out['Sentiment'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
