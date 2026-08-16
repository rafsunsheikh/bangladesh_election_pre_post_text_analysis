#!/usr/bin/env python3
"""Merge every collected dataset into one chronological, de-duplicated corpus.

Sources arrived in different shapes: some carry hand-annotated `Location 1/2/3`,
some carry finalized `Sentiment`, and the June batches carry neither. This script
normalizes all of them into a single table with an explicit `period` /
`period_order`, infers sentiment where it is missing, and resolves comments that
appear in more than one source file.

Outputs:
  data/unified/unified_comments.csv        one row per retained comment
  data/unified/coverage_report.md          per-period rows, location + sentiment coverage
  data/unified/periods/<order>_<period>.csv  per-period slices in the in-use schema,
                                             ready for the existing analytics scripts
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

LOCATION_COLS = ["Location 1", "Location 2", "Location 3"]

# Chronological registry. `order` drives timeline sequence everywhere downstream.
# Several periods were collected in more than one batch, so a period may list
# multiple files. Edit this table to re-scope periods; nothing else hardcodes them.
SOURCES: list[dict] = [
    {
        "period": "before_election",
        "order": 1,
        "path": "data/in_use/Before Election some annotated.final.csv",
    },
    {
        "period": "after_election",
        "order": 2,
        "path": "data/in_use/After Election.annotated.final.csv",
    },
    {
        # Cumulative 12.02.26-09.03.26 export. Its tail overlaps the next period;
        # the duplicate policy below reassigns those rows.
        "period": "after_election",
        "order": 2,
        "path": "data/in_use/post_election_data_updated_with_location_09_march.annotated.completed.csv",
    },
    {
        "period": "after_forming_government",
        "order": 3,
        "path": "data/in_use/After Forming Government.annotated.final.csv",
    },
    {
        # Exactly the 18.02.26-01.03.26 window.
        "period": "after_forming_government",
        "order": 3,
        "path": "data/in_use/after_forming_government_data_with_location.annotated.completed.csv",
    },
    {
        "period": "june_2026",
        "order": 4,
        "path": "data/june_data/June Data for sir-1 - Sheet1.csv",
    },
    {
        "period": "june_2026",
        "order": 4,
        "path": "data/june_data/June data number 2.csv",
    },
    {
        "period": "post_june_2026",
        "order": 5,
        "path": "data/in_use/post_june_2026_with_location.annotated.completed.csv",
    },
]

# June_Data_Bangla_Only-1.csv is deliberately absent: every one of its unique
# comments already appears in "June Data for sir-1", so including it would
# double-count 7,291 rows.

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\wঀ-৿ ]")


def norm_key(value: object) -> str:
    """Comparison key for duplicate detection: case/space/punctuation insensitive."""
    return SPACE_RE.sub(" ", PUNCT_RE.sub("", str(value))).strip().lower()


def canon_sentiment(value: object) -> str | None:
    key = str(value).strip().lower()
    if "sarcastic" in key:
        return "Sarcastic_negative"
    if "negative" in key:
        return "Negative"
    if "neutral" in key:
        return "Neutral"
    if "positive" in key:
        return "Positive"
    return None


def load_source(spec: dict) -> pd.DataFrame:
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")

    # The June exports are single-column and one carries a UTF-8 BOM.
    raw = pd.read_csv(path, encoding="utf-8-sig")
    text_col = "Comment" if "Comment" in raw.columns else raw.columns[0]

    out = pd.DataFrame()
    out["Comment"] = raw[text_col].fillna("").astype(str)
    for col in LOCATION_COLS:
        out[col] = raw[col] if col in raw.columns else pd.NA
    out["Sentiment"] = raw["Sentiment"].map(canon_sentiment) if "Sentiment" in raw.columns else pd.NA
    out["sentiment_confidence"] = raw["sentiment_confidence"] if "sentiment_confidence" in raw.columns else pd.NA
    out["sentiment_source"] = raw["sentiment_source"] if "sentiment_source" in raw.columns else pd.NA
    out["period"] = spec["period"]
    out["period_order"] = spec["order"]
    out["source_file"] = path.name

    out = out[out["Comment"].str.strip() != ""].copy()
    # Sources missing these columns yield all-NA; pin the dtype so concat across
    # sources does not warn about mixed/empty dtypes.
    for col in [*LOCATION_COLS, "Sentiment", "sentiment_source"]:
        out[col] = out[col].astype(object)
    out["sentiment_confidence"] = pd.to_numeric(out["sentiment_confidence"], errors="coerce")
    return out


def resolve_duplicates(df: pd.DataFrame, policy: str, min_len: int) -> tuple[pd.DataFrame, dict]:
    """Drop repeated comments according to `policy`.

    Short comments ("yes", "ঠিক", "আলহামদুলিল্লাহ") recur legitimately from
    different commenters, and each carries its own location, so only comments of
    at least `min_len` characters are treated as true duplicates.
    """
    work = df.copy()
    work["_key"] = work["Comment"].map(norm_key)
    work["_len"] = work["_key"].str.len()
    work["_row"] = range(len(work))

    dedupable = work["_len"] >= min_len
    short = work[~dedupable]
    long_rows = work[dedupable]

    # Exact repeats inside a single source file are always ingestion artifacts.
    before_intra = len(long_rows)
    long_rows = long_rows.drop_duplicates(subset=["source_file", "_key"], keep="first")
    intra_dropped = before_intra - len(long_rows)

    if policy == "keep-all":
        cross_dropped = 0
        kept = long_rows
    else:
        ascending = policy == "earliest"
        ordered = long_rows.sort_values(["_key", "period_order", "_row"], ascending=[True, ascending, ascending])
        before_cross = len(ordered)
        kept = ordered.drop_duplicates(subset=["_key"], keep="first")
        cross_dropped = before_cross - len(kept)

    out = pd.concat([kept, short], ignore_index=True).sort_values(["period_order", "_row"])
    stats = {
        "short_comments_exempt": int(len(short)),
        "intra_file_duplicates_dropped": int(intra_dropped),
        "cross_source_duplicates_dropped": int(cross_dropped),
    }
    return out.drop(columns=["_key", "_len", "_row"]).reset_index(drop=True), stats


def fill_missing_sentiment(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    missing = df["Sentiment"].isna()
    if not missing.any():
        return df
    preds = predict_sentiment(df.loc[missing, "Comment"], model_path=model_path)
    df.loc[missing, "Sentiment"] = preds["sentiment_label"].map(LABEL_TO_CANON).values
    df.loc[missing, "sentiment_confidence"] = preds["sentiment_confidence"].values
    df.loc[missing, "sentiment_source"] = "model_unified_fill"
    return df


def write_coverage_report(df: pd.DataFrame, stats: dict, out_path: Path, policy: str, min_len: int) -> None:
    rows = []
    for (order, period), grp in df.groupby(["period_order", "period"], sort=True):
        has_loc = grp[LOCATION_COLS].notna().any(axis=1)
        rows.append(
            {
                "order": order,
                "period": period,
                "comments": len(grp),
                "with_location": int(has_loc.sum()),
                "location_coverage": f"{has_loc.mean():.1%}",
                "sources": grp["source_file"].nunique(),
            }
        )
    cov = pd.DataFrame(rows)

    sent = (
        df.pivot_table(index="period", columns="Sentiment", values="Comment", aggfunc="count", fill_value=0)
        .reindex(cov["period"])
    )
    sent_pct = sent.div(sent.sum(axis=1), axis=0).mul(100).round(1)

    def to_md(frame: pd.DataFrame, index: bool = False) -> str:
        cols = ([frame.index.name or ""] if index else []) + [str(c) for c in frame.columns]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for idx, row in frame.iterrows():
            vals = ([str(idx)] if index else []) + [str(row[c]) for c in frame.columns]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    text = [
        "# Unified Dataset Coverage",
        "",
        f"Total retained comments: **{len(df):,}** across {df['period'].nunique()} periods "
        f"and {df['source_file'].nunique()} source files.",
        "",
        "## Per Period",
        to_md(cov),
        "",
        "## Sentiment Mix (% of period)",
        to_md(sent_pct, index=True),
        "",
        "## Duplicate Resolution",
        f"- Policy: `{policy}` (which period a repeated comment is assigned to)",
        f"- Minimum length treated as a real duplicate: {min_len} characters",
        f"- Short comments exempted from de-duplication: {stats['short_comments_exempt']:,}",
        f"- Repeats dropped within a single source file: {stats['intra_file_duplicates_dropped']:,}",
        f"- Repeats dropped across source files: {stats['cross_source_duplicates_dropped']:,}",
        "",
        "## Caveats",
        "- Location analysis can only use periods with non-zero location coverage.",
        "  A period at 0% is absent from the maps, not zero-mention.",
        "- Sentiment marked `model_unified_fill` is inferred, not hand-annotated.",
        "",
    ]
    out_path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/unified"))
    parser.add_argument("--model", type=Path, default=Path("models/sentiment_model_second_pass_v2.joblib"))
    parser.add_argument(
        "--duplicate-policy",
        choices=["latest", "earliest", "keep-all"],
        default="latest",
        help="Which period keeps a comment found in several sources. 'latest' suits this corpus: "
        "the wide cumulative exports are older than the tightly-windowed ones, and later "
        "sources carry the location annotations.",
    )
    parser.add_argument(
        "--duplicate-min-length",
        type=int,
        default=25,
        help="Comments shorter than this are never de-duplicated; short phrases recur legitimately.",
    )
    args = parser.parse_args()

    frames = [load_source(spec) for spec in SOURCES]
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined):,} rows from {len(SOURCES)} source files.")

    deduped, stats = resolve_duplicates(combined, args.duplicate_policy, args.duplicate_min_length)
    print(f"Retained {len(deduped):,} rows after duplicate resolution ({args.duplicate_policy}).")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    if not args.model.exists():
        raise FileNotFoundError(f"Sentiment model not found: {args.model}")
    n_missing = int(deduped["Sentiment"].isna().sum())
    deduped = fill_missing_sentiment(deduped, args.model)
    print(f"Inferred sentiment for {n_missing:,} previously unlabeled rows.")

    deduped["needs_review"] = pd.to_numeric(deduped["sentiment_confidence"], errors="coerce") < 0.5

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unified_path = args.output_dir / "unified_comments.csv"
    ordered_cols = [
        "Comment", *LOCATION_COLS, "Sentiment", "sentiment_confidence",
        "sentiment_source", "needs_review", "period", "period_order", "source_file",
    ]
    deduped[ordered_cols].to_csv(unified_path, index=False, encoding="utf-8")
    print(f"Saved: {unified_path}")

    periods_dir = args.output_dir / "periods"
    periods_dir.mkdir(parents=True, exist_ok=True)
    for (order, period), grp in deduped.groupby(["period_order", "period"], sort=True):
        slice_path = periods_dir / f"{order}_{period}.csv"
        grp[["Comment", *LOCATION_COLS, "Sentiment", "sentiment_confidence", "sentiment_source", "needs_review"]].to_csv(
            slice_path, index=False, encoding="utf-8"
        )
        has_loc = grp[LOCATION_COLS].notna().any(axis=1)
        print(f"  {slice_path}  rows={len(grp):,}  location_coverage={has_loc.mean():.1%}")

    write_coverage_report(
        deduped, stats, args.output_dir / "coverage_report.md", args.duplicate_policy, args.duplicate_min_length
    )
    print(f"Saved: {args.output_dir / 'coverage_report.md'}")


if __name__ == "__main__":
    main()
