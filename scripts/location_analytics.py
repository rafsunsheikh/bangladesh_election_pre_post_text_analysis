#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import pandas as pd
from sentiment_inference import predict_sentiment

# Canonical names for common Bangladesh locations.
LOCATION_ALIASES = {
    "dhaka": "Dhaka",
    "ঢাকা": "Dhaka",
    "gazipur": "Gazipur",
    "গাজীপুর": "Gazipur",
    "narail": "Narail",
    "নড়াইল": "Narail",
    "narayanganj": "Narayanganj",
    "নারায়ণগঞ্জ": "Narayanganj",
    "chattogram": "Chattogram",
    "chittagong": "Chattogram",
    "চট্টগ্রাম": "Chattogram",
    "rajshahi": "Rajshahi",
    "রাজশাহী": "Rajshahi",
    "khulna": "Khulna",
    "খুলনা": "Khulna",
    "rangpur": "Rangpur",
    "রংপুর": "Rangpur",
    "barishal": "Barishal",
    "barisal": "Barishal",
    "বরিশাল": "Barishal",
    "sylhet": "Sylhet",
    "সিলেট": "Sylhet",
    "cumilla": "Cumilla",
    "comilla": "Cumilla",
    "কুমিল্লা": "Cumilla",
    "mymensingh": "Mymensingh",
    "ময়মনসিংহ": "Mymensingh",
    "bogura": "Bogura",
    "bogra": "Bogura",
    "বগুড়া": "Bogura",
    "feni": "Feni",
    "ফেনী": "Feni",
    "noakhali": "Noakhali",
    "নোয়াখালী": "Noakhali",
    "coxsbazar": "Cox's Bazar",
    "cox's bazar": "Cox's Bazar",
    "কক্সবাজার": "Cox's Bazar",
}

SPACE_RE = re.compile(r"\s+")
def normalize_location(value: str) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None
    key = SPACE_RE.sub(" ", raw.lower()).replace("’", "'").strip()
    key = key.replace(" জেলা", "")
    key_no_space = key.replace(" ", "")
    if key in LOCATION_ALIASES:
        return LOCATION_ALIASES[key]
    if key_no_space in LOCATION_ALIASES:
        return LOCATION_ALIASES[key_no_space]
    # Fallback: title-case latin text, keep Bangla text as-is.
    if re.search(r"[a-zA-Z]", raw):
        return raw.title()
    return raw


def load_period(path: Path, period_label: str, model_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Comment", "Location 1", "Location 2", "Location 3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = df[["Comment", "Location 1", "Location 2", "Location 3"]].copy()
    out["period"] = period_label
    out["doc_id"] = range(len(out))

    sent = predict_sentiment(out["Comment"].fillna(""), model_path=model_path)
    out["sentiment"] = sent["sentiment_label"].values
    out["sentiment_confidence"] = sent["sentiment_confidence"].values
    out["sentiment_source"] = sent["sentiment_source"].values

    # Curated labels beat re-prediction. Files carrying an annotated `Sentiment`
    # column keep it; the model only fills the gaps.
    if "Sentiment" in df.columns:
        provided = df["Sentiment"].astype(str).str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA})
        provided = provided.replace({"sarcastic negative": "sarcastic_negative", "sarcastic-negative": "sarcastic_negative"})
        valid = provided.isin(["negative", "neutral", "positive", "sarcastic_negative"])
        out.loc[valid.values, "sentiment"] = provided[valid].values
        out.loc[valid.values, "sentiment_source"] = "provided"
        if "sentiment_confidence" in df.columns:
            conf = pd.to_numeric(df["sentiment_confidence"], errors="coerce")
            has_conf = valid & conf.notna()
            out.loc[has_conf.values, "sentiment_confidence"] = conf[has_conf].values

    return out


def to_long_locations(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for rank in [1, 2, 3]:
        col = f"Location {rank}"
        part = df[["period", "doc_id", "Comment", "sentiment", "sentiment_confidence", "sentiment_source", col]].copy()
        part = part.rename(columns={col: "location_raw", "Comment": "comment"})
        part["location_rank"] = rank
        parts.append(part)

    long_df = pd.concat(parts, ignore_index=True)
    long_df["location"] = long_df["location_raw"].map(normalize_location)
    long_df = long_df[long_df["location"].notna()].copy()
    return long_df


def build_frequency(long_df: pd.DataFrame, total_docs_by_period: pd.Series) -> pd.DataFrame:
    agg = (
        long_df.groupby(["period", "location"], as_index=False)
        .agg(location_mentions=("location", "count"), unique_comments=("doc_id", "nunique"))
    )
    agg["comment_share"] = agg.apply(
        lambda r: r["unique_comments"] / max(1, int(total_docs_by_period[r["period"]])), axis=1
    )
    return agg.sort_values(["period", "location_mentions"], ascending=[True, False])


def build_growth(freq: pd.DataFrame, period_a: str, period_b: str) -> pd.DataFrame:
    a = freq[freq["period"] == period_a][["location", "location_mentions", "unique_comments", "comment_share"]]
    b = freq[freq["period"] == period_b][["location", "location_mentions", "unique_comments", "comment_share"]]
    merged = a.merge(b, on="location", how="outer", suffixes=(f"_{period_a}", f"_{period_b}")).fillna(0)
    merged["mention_delta"] = merged[f"location_mentions_{period_b}"] - merged[f"location_mentions_{period_a}"]
    merged["comment_share_delta"] = merged[f"comment_share_{period_b}"] - merged[f"comment_share_{period_a}"]
    merged["growth_rate"] = merged.apply(
        lambda r: (r[f"location_mentions_{period_b}"] - r[f"location_mentions_{period_a}"]) / r[f"location_mentions_{period_a}"]
        if r[f"location_mentions_{period_a}"] > 0
        else float("inf") if r[f"location_mentions_{period_b}"] > 0 else 0.0,
        axis=1,
    )
    return merged.sort_values("mention_delta", ascending=False)


def build_cooccurrence(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = long_df.groupby(["period", "doc_id"])
    for (period, doc_id), group in grouped:
        locs = sorted(set(group["location"].tolist()))
        if len(locs) < 2:
            continue
        for a, b in combinations(locs, 2):
            rows.append({"period": period, "doc_id": doc_id, "location_a": a, "location_b": b})

    if not rows:
        return pd.DataFrame(columns=["period", "location_a", "location_b", "co_mentions"])

    pairs = pd.DataFrame(rows)
    return (
        pairs.groupby(["period", "location_a", "location_b"], as_index=False)
        .agg(co_mentions=("doc_id", "count"))
        .sort_values(["period", "co_mentions"], ascending=[True, False])
    )


def build_sentiment_by_location(long_df: pd.DataFrame) -> pd.DataFrame:
    sent = (
        long_df.groupby(["period", "location", "sentiment"], as_index=False)
        .agg(comments=("doc_id", "nunique"), avg_confidence=("sentiment_confidence", "mean"))
    )
    totals = sent.groupby(["period", "location"]) ["comments"].transform("sum")
    sent["share"] = sent["comments"] / totals
    return sent.sort_values(["period", "location", "sentiment"])


def write_markdown_report(
    out_path: Path,
    overall_top: pd.DataFrame,
    per_period: dict[str, pd.DataFrame],
    growth_pair: tuple[str, str],
    growth: pd.DataFrame,
    cooc: pd.DataFrame,
) -> None:
    def to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows_"
        cols = [str(c) for c in df.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, row in df.iterrows():
            vals = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + rows)

    lines = [
        "# Location Analytics Report",
        "",
        "## Overall Top Locations",
        to_md(overall_top),
        "",
    ]

    for label, table in per_period.items():
        lines.append(f"## Top Locations — {label}")
        lines.append(to_md(table))
        lines.append("")

    label_a, label_b = growth_pair
    lines.append(f"## Biggest Changes ({label_b} - {label_a})")
    lines.append(to_md(growth))
    lines.append("")
    lines.append("## Top Co-occurring Location Pairs")
    lines.append(to_md(cooc))
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed location analytics for two election CSV datasets.")
    parser.add_argument(
        "--file-a",
        type=Path,
        default=Path("data/unified/periods/3_after_forming_government.csv"),
        help="Earlier period CSV of the growth pair",
    )
    parser.add_argument(
        "--file-b",
        type=Path,
        default=Path("data/unified/periods/5_post_june_2026.csv"),
        help="Later period CSV of the growth pair",
    )
    parser.add_argument("--label-a", type=str, default="after_forming_government")
    parser.add_argument("--label-b", type=str, default="post_june_2026")
    parser.add_argument(
        "--extra-files",
        type=Path,
        nargs="*",
        default=[Path("data/unified/periods/2_after_election.csv")],
        help="Additional period CSVs. They appear in frequency/sentiment/co-occurrence outputs; "
        "growth stays pairwise between --file-a and --file-b.",
    )
    parser.add_argument(
        "--extra-labels",
        type=str,
        nargs="*",
        default=["after_election"],
        help="Period labels for --extra-files, in the same order.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/location_analytics"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--sentiment-model",
        type=Path,
        default=Path("models/sentiment_model.joblib"),
        help="Path to trained sentiment model bundle. Falls back to heuristic if missing.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.sentiment_model if args.sentiment_model.exists() else None

    if len(args.extra_files) != len(args.extra_labels):
        raise ValueError(
            f"--extra-files ({len(args.extra_files)}) and --extra-labels ({len(args.extra_labels)}) must match in length."
        )

    period_a = load_period(args.file_a, args.label_a, model_path=model_path)
    period_b = load_period(args.file_b, args.label_b, model_path=model_path)
    frames = [period_a, period_b]
    for extra_path, extra_label in zip(args.extra_files, args.extra_labels):
        frames.append(load_period(extra_path, extra_label, model_path=model_path))
    all_df = pd.concat(frames, ignore_index=True)

    long_df = to_long_locations(all_df)
    long_df.to_csv(args.output_dir / "locations_long.csv", index=False, encoding="utf-8")

    total_docs = all_df.groupby("period")["doc_id"].count()
    freq = build_frequency(long_df, total_docs)
    freq.to_csv(args.output_dir / "location_frequency_by_period.csv", index=False, encoding="utf-8")

    overall = (
        freq.groupby("location", as_index=False)
        .agg(total_mentions=("location_mentions", "sum"), total_unique_comments=("unique_comments", "sum"))
        .sort_values("total_mentions", ascending=False)
    )
    overall.to_csv(args.output_dir / "location_frequency_overall.csv", index=False, encoding="utf-8")

    growth = build_growth(freq, args.label_a, args.label_b)
    growth.to_csv(args.output_dir / "location_growth.csv", index=False, encoding="utf-8")

    cooc = build_cooccurrence(long_df)
    cooc.to_csv(args.output_dir / "location_cooccurrence.csv", index=False, encoding="utf-8")

    sent = build_sentiment_by_location(long_df)
    sent.to_csv(args.output_dir / "location_sentiment.csv", index=False, encoding="utf-8")

    # Follow the frequency table's period order so report sections read in sequence,
    # rather than the growth-pair-first order the CLI arguments imply.
    all_labels = freq["period"].dropna().unique().tolist()
    write_markdown_report(
        out_path=args.output_dir / "report.md",
        overall_top=overall.head(args.top_n),
        per_period={label: freq[freq["period"] == label].head(args.top_n) for label in all_labels},
        growth_pair=(args.label_a, args.label_b),
        growth=growth.head(args.top_n),
        cooc=cooc.head(args.top_n),
    )

    print(f"Saved outputs to: {args.output_dir}")
    print("Files:")
    for name in [
        "locations_long.csv",
        "location_frequency_by_period.csv",
        "location_frequency_overall.csv",
        "location_growth.csv",
        "location_cooccurrence.csv",
        "location_sentiment.csv",
        "report.md",
    ]:
        print(f"  - {args.output_dir / name}")


if __name__ == "__main__":
    main()
