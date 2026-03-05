#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

LABELS = ["negative", "neutral", "positive", "sarcastic_negative"]
SPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^0-9a-zA-Z\u0980-\u09FF\s!?.,'\"]")

POSITIVE_WORDS = {
    "ভালো", "ভাল", "সেরা", "সফল", "শান্তি", "জয়", "জয়", "সমর্থন", "অভিনন্দন", "ধন্যবাদ",
    "উন্নতি", "উন্নয়ন", "উন্নয়ন", "নিরাপদ", "খুশি", "আশা", "দারুণ", "চমৎকার", "great", "good", "win",
}

NEGATIVE_WORDS = {
    "খারাপ", "মন্দ", "দুর্নীতি", "চোর", "বেইমান", "বেইমানি", "দালাল", "ভণ্ড", "ব্যর্থ", "অন্যায়",
    "অন্যায়", "সমস্যা", "গাদ্দার", "প্রতারণা", "ঘৃণা", "ভয়", "ভয়", "অপমান", "হত্যা", "গুম",
    "চাঁদাবাজ", "চাঁদাবাজি", "চান্দাবাজ", "সহিংসতা", "বিচারহীনতা", "নাটক", "পতন", "ধ্বংস", "bad", "corrupt",
}

SARCASM_MARKERS = {
    "হাহা", "হা হা", "হাসি পাই", "বাহ", "কি সুন্দর", "দারুণ", "চমৎকার", "কথা", "নাকি", "sure", "yeah right",
    "lol", "lmao", "rofl", "😂", "🤣", "😏", "🙃",
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = NON_TEXT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def _heuristic_probs(text: str) -> dict[str, float]:
    norm = normalize_text(text)
    tokens = norm.split()
    if not tokens:
        return {"negative": 0.12, "neutral": 0.76, "positive": 0.06, "sarcastic_negative": 0.06}

    pos_hits = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg_hits = sum(1 for token in tokens if token in NEGATIVE_WORDS)

    marker_hit = any(marker in norm for marker in SARCASM_MARKERS)
    punct_hit = norm.count("!") >= 2 or norm.count("?") >= 2
    mixed_polarity = pos_hits > 0 and neg_hits > 0

    sarcastic = marker_hit and (neg_hits > 0 or mixed_polarity or punct_hit)

    if sarcastic:
        return {"negative": 0.16, "neutral": 0.12, "positive": 0.07, "sarcastic_negative": 0.65}

    if neg_hits > pos_hits:
        conf = min(0.86, 0.52 + 0.08 * (neg_hits - pos_hits))
        rem = 1.0 - conf
        return {"negative": conf, "neutral": rem * 0.75, "positive": rem * 0.2, "sarcastic_negative": rem * 0.05}

    if pos_hits > neg_hits:
        conf = min(0.82, 0.5 + 0.08 * (pos_hits - neg_hits))
        rem = 1.0 - conf
        return {"negative": rem * 0.25, "neutral": rem * 0.55, "positive": conf, "sarcastic_negative": rem * 0.2}

    return {"negative": 0.12, "neutral": 0.74, "positive": 0.08, "sarcastic_negative": 0.06}


def load_model_bundle(model_path: Path | None) -> dict | None:
    if model_path is None:
        return None
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    if "model" not in bundle:
        raise ValueError(f"Invalid model bundle: {model_path}")
    return bundle


def _predict_with_model(texts: list[str], bundle: dict) -> pd.DataFrame:
    model = bundle["model"]
    class_labels = list(bundle.get("labels", model.classes_))
    probs = model.predict_proba(texts)

    rows: list[dict[str, object]] = []
    for idx, text in enumerate(texts):
        row_probs = {label: 0.0 for label in LABELS}
        for c_idx, label in enumerate(class_labels):
            if label in row_probs:
                row_probs[label] = float(probs[idx, c_idx])
        chosen = max(row_probs, key=row_probs.get)
        confidence = float(row_probs[chosen])
        score = row_probs["positive"] - row_probs["negative"] - row_probs["sarcastic_negative"]
        rows.append(
            {
                "text": text,
                "sentiment_label": chosen,
                "sentiment_confidence": confidence,
                "sentiment_score": float(score),
                "prob_negative": row_probs["negative"],
                "prob_neutral": row_probs["neutral"],
                "prob_positive": row_probs["positive"],
                "prob_sarcastic_negative": row_probs["sarcastic_negative"],
                "sentiment_source": "model",
            }
        )
    return pd.DataFrame(rows)


def _predict_with_heuristics(texts: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for text in texts:
        probs = _heuristic_probs(text)
        label = max(probs, key=probs.get)
        confidence = float(probs[label])
        score = probs["positive"] - probs["negative"] - probs["sarcastic_negative"]
        rows.append(
            {
                "text": text,
                "sentiment_label": label,
                "sentiment_confidence": confidence,
                "sentiment_score": float(score),
                "prob_negative": float(probs["negative"]),
                "prob_neutral": float(probs["neutral"]),
                "prob_positive": float(probs["positive"]),
                "prob_sarcastic_negative": float(probs["sarcastic_negative"]),
                "sentiment_source": "heuristic",
            }
        )
    return pd.DataFrame(rows)


def predict_sentiment(texts: Iterable[str], model_path: Path | None = None) -> pd.DataFrame:
    text_list = ["" if t is None else str(t) for t in texts]
    bundle = load_model_bundle(model_path)
    if bundle is None:
        return _predict_with_heuristics(text_list)
    return _predict_with_model(text_list, bundle)


def to_json_probs(row: pd.Series) -> str:
    obj = {
        "negative": float(row["prob_negative"]),
        "neutral": float(row["prob_neutral"]),
        "positive": float(row["prob_positive"]),
        "sarcastic_negative": float(row["prob_sarcastic_negative"]),
    }
    return json.dumps(obj, ensure_ascii=False)
