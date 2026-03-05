#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sentiment_inference import predict_sentiment, to_json_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment inference with confidence and sarcastic_negative class.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--text-col", type=str, default="Comment", help="Input text column")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    parser.add_argument("--model", type=Path, default=Path("models/sentiment_model.joblib"), help="Trained model path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    text_col = args.text_col
    if text_col not in df.columns:
        if len(df.columns) == 1:
            text_col = df.columns[0]
            print(f"Column '{args.text_col}' not found; using only available column: '{text_col}'")
        else:
            raise ValueError(f"Column not found: {args.text_col}")

    preds = predict_sentiment(df[text_col].fillna(""), model_path=args.model)
    out = df.copy()
    out["sentiment_label"] = preds["sentiment_label"]
    out["sentiment_confidence"] = preds["sentiment_confidence"]
    out["sentiment_score"] = preds["sentiment_score"]
    out["sentiment_source"] = preds["sentiment_source"]
    out["prob_negative"] = preds["prob_negative"]
    out["prob_neutral"] = preds["prob_neutral"]
    out["prob_positive"] = preds["prob_positive"]
    out["prob_sarcastic_negative"] = preds["prob_sarcastic_negative"]
    out["sentiment_probabilities_json"] = preds.apply(to_json_probs, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Saved inference output: {args.output}")


if __name__ == "__main__":
    main()
