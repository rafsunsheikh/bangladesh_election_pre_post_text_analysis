#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from sentiment_inference import LABELS, predict_sentiment  # noqa: E402

CANON = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "sarcastic_negative": "Sarcastic_negative",
    "sarcastic negative": "Sarcastic_negative",
    "sarcastic-negative": "Sarcastic_negative",
}
LABEL_TO_KEY = {
    "Negative": "negative",
    "Neutral": "neutral",
    "Positive": "positive",
    "Sarcastic_negative": "sarcastic_negative",
}


def canon_label(x: object) -> str:
    k = str(x).strip().lower()
    if "sarcastic" in k:
        return "Sarcastic_negative"
    if "negative" in k:
        return "Negative"
    if "neutral" in k:
        return "Neutral"
    if "positive" in k:
        return "Positive"
    return CANON.get(k, str(x).strip())


def detect_review_mask(df: pd.DataFrame) -> pd.Series:
    if "needs_review" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["needs_review"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def merge_trusted_review(completed: pd.DataFrame, review: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = completed.copy()
    mask = detect_review_mask(out)
    idx = out.index[mask]
    if len(idx) != len(review):
        raise ValueError(f"Mismatch: completed needs_review={len(idx)} review_rows={len(review)}")

    out["Sentiment"] = out["Sentiment"].map(canon_label)
    rev = review.copy()
    rev["Sentiment"] = rev["Sentiment"].map(canon_label)

    changed = int((out.loc[idx, "Sentiment"].reset_index(drop=True) != rev["Sentiment"].reset_index(drop=True)).sum())

    out.loc[idx, "Sentiment"] = rev["Sentiment"].values
    out.loc[idx, "sentiment_source"] = "manual_second_pass"
    out.loc[idx, "sentiment_confidence"] = 1.0
    out.loc[idx, "needs_review"] = False

    return out, {"review_rows": len(rev), "changed_vs_autofill": changed}


def auto_review_second_pass(completed: pd.DataFrame, review: pd.DataFrame, model_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    out = completed.copy()
    out["Sentiment"] = out["Sentiment"].map(canon_label)
    rev = review.copy()
    rev["Sentiment"] = rev["Sentiment"].map(canon_label)

    mask = detect_review_mask(out)
    idx = out.index[mask]
    if len(idx) != len(rev):
        raise ValueError(f"Mismatch: completed needs_review={len(idx)} review_rows={len(rev)}")

    preds = predict_sentiment(rev["Comment"].fillna(""), model_path=model_path)
    pred_label = preds["sentiment_label"].map(
        {
            "negative": "Negative",
            "neutral": "Neutral",
            "positive": "Positive",
            "sarcastic_negative": "Sarcastic_negative",
        }
    )

    keep = []
    change = []
    final_labels = []

    for i in range(len(rev)):
        current = rev.iloc[i]["Sentiment"]
        predicted = pred_label.iloc[i]
        conf = float(preds.iloc[i]["sentiment_confidence"])

        cur_key = LABEL_TO_KEY.get(current)
        cur_prob = float(preds.iloc[i][f"prob_{cur_key}"]) if cur_key else 0.0

        # Conservative auto-adjudication: only override when model is clearly stronger.
        should_change = (predicted != current) and (conf >= 0.62) and (cur_prob <= 0.24)
        if should_change:
            final_labels.append(predicted)
            change.append(i)
        else:
            final_labels.append(current)
            keep.append(i)

    rev["Sentiment"] = final_labels
    rev["auto_review_predicted"] = pred_label
    rev["auto_review_pred_confidence"] = preds["sentiment_confidence"]
    rev["auto_review_action"] = ["change" if i in set(change) else "keep" for i in range(len(rev))]

    out.loc[idx, "Sentiment"] = rev["Sentiment"].values
    out.loc[idx, "needs_review"] = False

    # For changed rows, take model confidence/source; for kept rows preserve prior confidence/source.
    changed_mask = pd.Series([False] * len(out), index=out.index)
    changed_rows_idx = idx[list(change)] if change else []
    if len(change) > 0:
        changed_mask.loc[changed_rows_idx] = True
        out.loc[changed_rows_idx, "sentiment_source"] = "auto_second_pass_changed"
        out.loc[changed_rows_idx, "sentiment_confidence"] = preds.iloc[change]["sentiment_confidence"].values

    kept_rows_idx = idx[list(keep)] if keep else []
    if len(keep) > 0:
        out.loc[kept_rows_idx, "sentiment_source"] = "auto_second_pass_kept"

    return out, {
        "review_rows": len(rev),
        "kept": len(keep),
        "changed": len(change),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize second-pass sentiment annotations automatically.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--trusted-second-pass", type=str, required=True, help="Filename of manually updated second-pass review CSV")
    parser.add_argument("--base-train-file", type=Path, default=Path("data/Before Election some annotated.final.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/sentiment_model_second_pass_v2.joblib"))
    parser.add_argument("--report-out", type=Path, default=Path("outputs/second_pass_finalization_report.json"))
    args = parser.parse_args()

    targets = [
        "After Election",
        "After Forming Government",
        "12.02.26 —- 17.02.26 - Sheet1",
        "18.02.26 —--- 01.03.26 - Sheet1",
    ]

    # Train a model from trusted labels: base finalized set + trusted second-pass file.
    trusted_path = args.data_dir / args.trusted_second_pass
    if not trusted_path.exists():
        raise FileNotFoundError(f"Trusted second-pass file not found: {trusted_path}")

    base = pd.read_csv(args.base_train_file)[["Comment", "Sentiment"]].copy()
    base["Sentiment"] = base["Sentiment"].map(canon_label)

    trusted = pd.read_csv(trusted_path)[["Comment", "Sentiment"]].copy()
    trusted["Sentiment"] = trusted["Sentiment"].map(canon_label)

    train_df = pd.concat([base, trusted], ignore_index=True)
    train_tmp = Path("/tmp/second_pass_train.csv")
    train_df.to_csv(train_tmp, index=False, encoding="utf-8")

    # Reuse training script.
    cmd = (
        f'.venv/bin/python scripts/train_sentiment_model.py --input "{train_tmp}" '
        f'--text-col Comment --label-col Sentiment --output-model "{args.model_out}" '
        f'--output-report "outputs/second_pass_model_report.json" --test-size 0.2'
    )
    rc = __import__("os").system(cmd)
    if rc != 0:
        raise RuntimeError("Failed to train second-pass model")

    summary: dict[str, dict[str, int]] = {}

    for stem in targets:
        completed_path = args.data_dir / f"{stem}.annotated.completed.csv"
        review_path = args.data_dir / f"{stem}.annotated.second_pass_review.csv"
        final_path = args.data_dir / f"{stem}.annotated.final.csv"

        completed = pd.read_csv(completed_path)
        review = pd.read_csv(review_path)

        if review_path.name == trusted_path.name:
            final_df, stats = merge_trusted_review(completed, review)
            stats["mode"] = "trusted_manual_merge"
        else:
            final_df, stats = auto_review_second_pass(completed, review, model_path=args.model_out)
            stats["mode"] = "auto_adjudicated"

        final_df["Sentiment"] = final_df["Sentiment"].map(canon_label)
        final_df.to_csv(final_path, index=False, encoding="utf-8")

        # Keep completed file updated too (user no longer wants review loop).
        final_df.to_csv(completed_path, index=False, encoding="utf-8")

        # Update second-pass file with adjudication metadata for audit (when auto mode).
        if stats["mode"] == "auto_adjudicated":
            preds = predict_sentiment(review["Comment"].fillna(""), model_path=args.model_out)
            review_out = review.copy()
            review_out["Sentiment"] = review_out["Sentiment"].map(canon_label)
            review_out["auto_review_predicted"] = preds["sentiment_label"].map(
                {
                    "negative": "Negative",
                    "neutral": "Neutral",
                    "positive": "Positive",
                    "sarcastic_negative": "Sarcastic_negative",
                }
            )
            review_out["auto_review_pred_confidence"] = preds["sentiment_confidence"]
            # recompute action the same as rule
            actions = []
            for i in range(len(review_out)):
                cur = review_out.iloc[i]["Sentiment"]
                pred = review_out.iloc[i]["auto_review_predicted"]
                conf = float(review_out.iloc[i]["auto_review_pred_confidence"])
                cur_prob = float(preds.iloc[i][f"prob_{LABEL_TO_KEY.get(cur, 'neutral')}"])
                change = (pred != cur) and (conf >= 0.62) and (cur_prob <= 0.24)
                actions.append("change" if change else "keep")
            review_out["auto_review_action"] = actions
            review_out.to_csv(review_path, index=False, encoding="utf-8")

        summary[stem] = {
            **stats,
            "final_rows": int(len(final_df)),
            "remaining_needs_review": int(detect_review_mask(final_df).sum()),
            "label_negative": int((final_df["Sentiment"] == "Negative").sum()),
            "label_neutral": int((final_df["Sentiment"] == "Neutral").sum()),
            "label_positive": int((final_df["Sentiment"] == "Positive").sum()),
            "label_sarcastic_negative": int((final_df["Sentiment"] == "Sarcastic_negative").sum()),
        }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {args.report_out}")
    for name, stats in summary.items():
        print(f"- {name}: {stats}")


if __name__ == "__main__":
    main()
