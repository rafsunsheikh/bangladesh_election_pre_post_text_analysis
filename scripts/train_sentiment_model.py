#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from sentiment_inference import LABELS, normalize_text


def build_model() -> Pipeline:
    word_tfidf = TfidfVectorizer(
        preprocessor=normalize_text,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=120000,
    )
    char_tfidf = TfidfVectorizer(
        preprocessor=normalize_text,
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=2,
        max_features=180000,
    )

    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        solver="lbfgs",
    )

    return Pipeline([
        ("features", features),
        ("clf", clf),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-class sentiment model with sarcasm support.")
    parser.add_argument("--input", type=Path, required=True, help="Labeled CSV path")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name")
    parser.add_argument("--output-model", type=Path, default=Path("models/sentiment_model.joblib"))
    parser.add_argument("--output-report", type=Path, default=Path("outputs/sentiment_training_report.json"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = {args.text_col, args.label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train_df = df[[args.text_col, args.label_col]].dropna().copy()
    train_df[args.text_col] = train_df[args.text_col].astype(str)
    train_df[args.label_col] = train_df[args.label_col].astype(str).str.strip().str.lower()

    valid_labels = set(LABELS)
    bad = sorted(set(train_df[args.label_col]) - valid_labels)
    if bad:
        raise ValueError(f"Unsupported labels found: {bad}. Allowed labels: {LABELS}")

    if train_df[args.label_col].nunique() < 2:
        raise ValueError("Need at least two sentiment classes in training data.")

    min_test_required = train_df[args.label_col].nunique()
    requested_test_n = max(1, int(round(len(train_df) * args.test_size)))
    use_stratify = requested_test_n >= min_test_required

    x_train, x_test, y_train, y_test = train_test_split(
        train_df[args.text_col],
        train_df[args.label_col],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=train_df[args.label_col] if use_stratify else None,
    )

    model = build_model()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "labels": LABELS,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }
    joblib.dump(bundle, args.output_model)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model: {args.output_model}")
    print(f"Saved report: {args.output_report}")
    print(f"Macro F1: {report.get('macro avg', {}).get('f1-score', 0.0):.4f}")


if __name__ == "__main__":
    main()
