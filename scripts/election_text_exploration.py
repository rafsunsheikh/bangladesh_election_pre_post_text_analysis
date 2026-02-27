#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import features as pil_features
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


BANGLA_STOPWORDS = {
    "এবং", "কিন্তু", "তবে", "যদি", "যেন", "যদিও", "অথবা", "কারণ", "তাই", "যে",
    "এই", "ওই", "সেই", "এটা", "ওটা", "সেটা", "এটি", "তিনি", "তারা", "আমরা",
    "আমি", "তুমি", "আপনি", "আপনার", "তার", "তাদের", "আমার", "আমাদের", "তোমার",
    "হয়", "হয়", "হবে", "হয়েছে", "হয়েছে", "করবে", "করে", "করা", "করছে", "ছিল",
    "ছিলো", "আছে", "নেই", "না", "কি", "কী", "তো", "ও", "আর", "বা", "একটা", "একটি",
    "জন্য", "দিকে", "পরে", "আগে", "সাথে", "সঙ্গে", "মধ্যে", "উপর", "নিচে", "থেকে",
    "যায়", "যায়", "দিয়ে", "দিয়ে", "হলেও", "যখন", "তখন", "কখন", "কোথায়", "কোথায়",
    "কেন", "কেমন", "অনেক", "খুব", "বেশ", "আরও", "শুধু", "সব", "কেউ", "কিছু",
    "এখন", "তখনও", "এইটা", "ওইটা", "সবার", "একজন",
    "হচ্ছে", "এর", "কে", "এক", "হয়ে", "গুলো", "গেছে", "কথা", "রে", "করেছে", "দিন", "হইছে", "এদের",
}

BANGLA_POSITIVE_WORDS = {
    "ভালো", "ভাল", "সেরা", "সফল", "শান্তি", "জয়", "জয়", "সমর্থন", "অভিনন্দন", "ধন্যবাদ",
    "উন্নতি", "উন্নয়ন", "উন্নয়ন", "নিরাপদ", "খুশি", "আশা", "পজিটিভ", "দারুণ", "চমৎকার",
    "শক্তিশালী", "ঠিক", "ইতিবাচক", "সমাধান", "শুভেচ্ছা", "অভিনন্দন", "সমৃদ্ধি",
}

BANGLA_NEGATIVE_WORDS = {
    "খারাপ", "মন্দ", "দুর্নীতি", "চোর", "বেইমান", "বেইমানি", "দালাল", "ভণ্ড", "ব্যর্থ", "অন্যায়",
    "অন্যায়", "সমস্যা", "গাদ্দার", "প্রতারণা", "ঘৃণা", "ভয়", "ভয়", "অপমান", "হত্যা", "গুম",
    "চাঁদাবাজ", "চাঁদাবাজি", "চান্দাবাজ", "সহিংসতা", "বিচারহীনতা", "নাটক", "পতন", "ধ্বংস",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_TEXT_RE = re.compile(r"[^0-9a-zA-Z\u0980-\u09FF\u200c\u200d\s]")
DIGIT_RE = re.compile(r"[0-9০-৯]+")
SPACE_RE = re.compile(r"\s+")
LEADING_MARKS_RE = re.compile(r"^[\u0981-\u0983\u09bc\u09be-\u09c4\u09c7-\u09c8\u09cb-\u09cc\u09cd\u09d7]+")
VALID_BASE_CHAR_RE = re.compile(r"[a-zA-Z\u0985-\u09b9\u09dc-\u09df\u09f0-\u09f1]")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "assets" / "fonts"
NOTO_FONT_FILE = ASSETS_DIR / "NotoSansBengali-Regular.ttf"
NOTO_FONT_URL = (
    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
    "NotoSansBengali/NotoSansBengali-Regular.ttf"
)


def set_font_stack(primary_font: str) -> None:
    fallback_fonts = [
        "Arial Unicode MS",
        "Noto Sans",
        "DejaVu Sans",
        "Arial",
    ]
    stack = [primary_font] + [name for name in fallback_fonts if name != primary_font]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = stack
    plt.rcParams["axes.unicode_minus"] = False


def configure_font() -> str:
    candidates = [
        "Arial Unicode MS",
        "Noto Sans Bengali",
        "Noto Serif Bengali",
        "Vrinda",
        "Mukti",
        "Kalpurush",
        "Siyam Rupali",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            set_font_stack(name)
            return name
    set_font_stack("DejaVu Sans")
    return "DejaVu Sans"


def get_font_path(font_name: str) -> str | None:
    for font in font_manager.fontManager.ttflist:
        if font.name == font_name:
            return font.fname
    try:
        return font_manager.findfont(font_name, fallback_to_default=True)
    except Exception:
        return None


def download_font_if_missing(target_path: Path, url: str) -> bool:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return True
    try:
        urllib.request.urlretrieve(url, target_path)
        return True
    except Exception:
        return False


def ensure_bangla_font(font_name: str) -> tuple[str, str | None]:
    if download_font_if_missing(NOTO_FONT_FILE, NOTO_FONT_URL):
        try:
            font_manager.fontManager.addfont(str(NOTO_FONT_FILE))
            return font_name, str(NOTO_FONT_FILE)
        except Exception:
            pass

    font_path = get_font_path(font_name)
    if font_path:
        return font_name, font_path

    return font_name, None


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / path).resolve()


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\u0980-\u09FF]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "dataset"


def normalize_dataset_label(stem: str) -> str:
    label = stem.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label).strip()
    if not label:
        return "Dataset"
    return label.title()


def dataset_sort_key(label: str) -> tuple[int, str]:
    low = label.lower()
    if "before" in low:
        return (0, low)
    if "after election" in low:
        return (1, low)
    if "after forming" in low or "forming government" in low:
        return (2, low)
    return (9, low)


def ensure_unique_label(label: str, existing: set[str]) -> str:
    if label not in existing:
        return label
    idx = 2
    while f"{label} ({idx})" in existing:
        idx += 1
    return f"{label} ({idx})"


def discover_dataset_files(
    data_dir: Path,
    input_files: list[Path] | None,
) -> dict[str, Path]:
    dataset_files: dict[str, Path] = {}
    labels_in_use: set[str] = set()

    chosen_files: list[Path]
    if input_files:
        chosen_files = [resolve_input_path(path) for path in input_files]
    else:
        chosen_files = sorted(
            [path for path in data_dir.glob("*.csv") if not path.name.startswith(".")],
            key=lambda p: dataset_sort_key(normalize_dataset_label(p.stem)),
        )

    if not chosen_files:
        raise FileNotFoundError(f"No CSV files found. Checked: {data_dir}")

    for path in chosen_files:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        label = normalize_dataset_label(path.stem)
        label = ensure_unique_label(label, labels_in_use)
        labels_in_use.add(label)
        dataset_files[label] = path
    return dataset_files


def cleanup_legacy_outputs(output_dir: Path) -> None:
    legacy_files = [
        "top_terms_before.csv",
        "top_terms_after.csv",
        "topics_before.csv",
        "topics_after.csv",
        "document_topics_before.csv",
        "document_topics_after.csv",
    ]
    for name in legacy_files:
        target = output_dir / name
        if target.exists():
            target.unlink()


def load_single_column_csv(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                rows.append("")
                continue
            rows.append(row[0].strip())
    return rows


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    # Preserve word integrity: remove joiners instead of turning them into spaces.
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = NON_TEXT_RE.sub(" ", text)
    text = DIGIT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    cleaned_tokens: list[str] = []
    for token in text.split():
        token = token.replace("\u200c", "").replace("\u200d", "")
        token = LEADING_MARKS_RE.sub("", token)
        token = token.rstrip("\u09cd")
        if len(token) <= 1:
            continue
        if token in stopwords:
            continue
        if not VALID_BASE_CHAR_RE.search(token):
            continue
        cleaned_tokens.append(token)
    return cleaned_tokens


def build_dataframe(dataset_files: dict[str, Path]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset, path in dataset_files.items():
        rows = load_single_column_csv(path)
        for idx, raw in enumerate(rows):
            clean = normalize_text(raw)
            tokens = tokenize(clean, BANGLA_STOPWORDS)
            records.append(
                {
                    "dataset": dataset,
                    "dataset_slug": slugify(dataset),
                    "doc_id": idx,
                    "raw_text": raw,
                    "clean_text": clean,
                    "tokens": tokens,
                    "token_text": " ".join(tokens),
                    "char_len": len(raw),
                    "token_len": len(tokens),
                    "is_empty_raw": int(raw.strip() == ""),
                    "is_empty_tokenized": int(len(tokens) == 0),
                }
            )
    return pd.DataFrame(records)


def score_sentiment(tokens: list[str]) -> tuple[int, int, float, str]:
    if not tokens:
        return 0, 0, 0.0, "neutral"

    pos_hits = sum(1 for token in tokens if token in BANGLA_POSITIVE_WORDS)
    neg_hits = sum(1 for token in tokens if token in BANGLA_NEGATIVE_WORDS)
    score = (pos_hits - neg_hits) / max(1, len(tokens))

    if pos_hits > neg_hits and score > 0:
        label = "positive"
    elif neg_hits > pos_hits and score < 0:
        label = "negative"
    else:
        label = "neutral"
    return pos_hits, neg_hits, score, label


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    sentiment_rows = df["tokens"].apply(score_sentiment)
    sentiment_df = pd.DataFrame(
        sentiment_rows.tolist(),
        columns=["positive_hits", "negative_hits", "sentiment_score", "sentiment_label"],
        index=df.index,
    )
    return pd.concat([df, sentiment_df], axis=1)


def build_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "sentiment_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    total_per_dataset = summary.groupby("dataset")["count"].transform("sum")
    summary["percentage"] = summary["count"] / total_per_dataset
    label_order = {"negative": 0, "neutral": 1, "positive": 2}
    summary["label_order"] = summary["sentiment_label"].map(label_order).fillna(99)
    summary = summary.sort_values(["dataset", "label_order"]).drop(columns=["label_order"])
    return summary


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("dataset", as_index=False)
        .agg(
            total_rows=("doc_id", "count"),
            empty_raw_rows=("is_empty_raw", "sum"),
            empty_after_clean=("is_empty_tokenized", "sum"),
            avg_char_length=("char_len", "mean"),
            median_char_length=("char_len", "median"),
            avg_token_length=("token_len", "mean"),
            median_token_length=("token_len", "median"),
        )
    )
    summary["non_empty_rows"] = summary["total_rows"] - summary["empty_raw_rows"]
    return summary[
        [
            "dataset",
            "total_rows",
            "non_empty_rows",
            "empty_raw_rows",
            "empty_after_clean",
            "avg_char_length",
            "median_char_length",
            "avg_token_length",
            "median_token_length",
        ]
    ]


def term_counter(text_tokens: Iterable[list[str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for tokens in text_tokens:
        counter.update(tokens)
    return counter


def build_top_terms(df: pd.DataFrame, dataset: str, top_n: int = 40) -> pd.DataFrame:
    subset = df[df["dataset"] == dataset]
    counter = term_counter(subset["tokens"])
    total = sum(counter.values()) or 1
    rows = []
    for term, count in counter.most_common(top_n):
        rows.append(
            {
                "dataset": dataset,
                "term": term,
                "count": count,
                "relative_freq": count / total,
            }
        )
    return pd.DataFrame(rows)


def build_top_terms_all(df: pd.DataFrame, datasets: list[str], top_n: int = 40) -> pd.DataFrame:
    frames = [build_top_terms(df, dataset, top_n=top_n) for dataset in datasets]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_distinctive_terms(df: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    counters: dict[str, Counter[str]] = {}
    totals: dict[str, int] = {}
    all_terms: set[str] = set()

    for dataset in datasets:
        dataset_counter = term_counter(df[df["dataset"] == dataset]["tokens"])
        counters[dataset] = dataset_counter
        totals[dataset] = sum(dataset_counter.values())
        all_terms |= set(dataset_counter.keys())

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        others = [name for name in datasets if name != dataset]
        for term in all_terms:
            count = counters[dataset][term]
            rel = count / totals[dataset] if totals[dataset] else 0.0

            if others:
                others_rel = [
                    counters[other][term] / totals[other] if totals[other] else 0.0
                    for other in others
                ]
                avg_others_rel = sum(others_rel) / len(others_rel)
            else:
                avg_others_rel = 0.0

            rows.append(
                {
                    "dataset": dataset,
                    "term": term,
                    "count": count,
                    "relative_freq": rel,
                    "others_avg_relative_freq": avg_others_rel,
                    "distinctiveness_score": rel - avg_others_rel,
                }
            )

    distinctive = pd.DataFrame(rows)
    return distinctive.sort_values(["dataset", "distinctiveness_score"], ascending=[True, False])


def run_lda_for_dataset(
    df: pd.DataFrame,
    dataset: str,
    requested_topics: int,
    top_words: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[(df["dataset"] == dataset) & (df["token_text"].str.strip() != "")].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        min_df=2,
        max_df=0.95,
        max_features=1500,
    )
    document_term = vectorizer.fit_transform(subset["token_text"])
    if document_term.shape[1] == 0:
        vectorizer = CountVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            min_df=1,
            max_df=1.0,
            max_features=1500,
        )
        document_term = vectorizer.fit_transform(subset["token_text"])

    num_topics = max(2, min(requested_topics, document_term.shape[0], document_term.shape[1]))
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        learning_method="batch",
        max_iter=25,
    )
    lda.fit(document_term)
    doc_topic = lda.transform(document_term)
    terms = vectorizer.get_feature_names_out()

    topic_rows: list[dict[str, object]] = []
    topic_prevalence = doc_topic.mean(axis=0)
    for topic_idx, topic_weights in enumerate(lda.components_):
        top_indices = topic_weights.argsort()[::-1][:top_words]
        topic_record: dict[str, object] = {
            "dataset": dataset,
            "dataset_slug": slugify(dataset),
            "topic_id": int(topic_idx),
            "topic_prevalence": float(topic_prevalence[topic_idx]),
            "top_terms": ", ".join(terms[i] for i in top_indices),
        }
        for rank, term_idx in enumerate(top_indices, start=1):
            topic_record[f"term_{rank}"] = terms[term_idx]
            topic_record[f"weight_{rank}"] = float(topic_weights[term_idx])
        topic_rows.append(topic_record)

    doc_rows: list[dict[str, object]] = []
    for i, (_, row) in enumerate(subset.iterrows()):
        dominant_topic = int(doc_topic[i].argmax())
        doc_rows.append(
            {
                "dataset": dataset,
                "dataset_slug": slugify(dataset),
                "doc_id": int(row["doc_id"]),
                "raw_text": row["raw_text"],
                "clean_text": row["clean_text"],
                "dominant_topic": dominant_topic,
                "dominant_topic_prob": float(doc_topic[i][dominant_topic]),
            }
        )
    return pd.DataFrame(topic_rows), pd.DataFrame(doc_rows)


def subplot_grid(total: int, max_cols: int = 2, width: float = 8.0, height: float = 5.0):
    cols = min(max_cols, max(1, total))
    rows = max(1, math.ceil(total / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), constrained_layout=True)
    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    else:
        axes_list = list(axes.flatten())
    return fig, axes_list


def hide_unused_axes(axes: list[plt.Axes], used: int) -> None:
    for ax in axes[used:]:
        ax.axis("off")


def plot_top_terms_all(top_terms_all: pd.DataFrame, datasets: list[str], output: Path) -> None:
    fig, axes = subplot_grid(len(datasets), max_cols=2, width=10.0, height=6.5)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = top_terms_all[top_terms_all["dataset"] == dataset].copy()
        if subset.empty:
            ax.set_title(f"{dataset}: Top Terms")
            ax.text(0.5, 0.5, "No terms available", ha="center", va="center")
            ax.axis("off")
            continue
        plot_df = subset.sort_values("count", ascending=True).tail(12)
        ax.barh(plot_df["term"], plot_df["count"], color="#1f77b4", alpha=0.85)
        ax.set_title(f"{dataset}: Top Terms")
        ax.set_xlabel("Term Frequency")
        ax.tick_params(axis="y", labelsize=12)
    hide_unused_axes(axes, len(datasets))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_length_distribution_all(df: pd.DataFrame, datasets: list[str], output: Path) -> None:
    fig, axes = subplot_grid(len(datasets), max_cols=2, width=8.0, height=4.5)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        series = df[df["dataset"] == dataset]["char_len"]
        ax.hist(series, bins=25, color="#d62728", alpha=0.75)
        ax.set_title(f"{dataset}: Character Length")
        ax.set_xlabel("Characters per comment")
        ax.set_ylabel("Count")
    hide_unused_axes(axes, len(datasets))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_wordclouds_all(
    df: pd.DataFrame,
    datasets: list[str],
    output: Path,
    font_path: str | None = None,
    per_dataset_dir: Path | None = None,
    random_seed: int | None = None,
) -> None:
    fig, axes = subplot_grid(len(datasets), max_cols=2, width=9.0, height=6.5)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_tokens = df[df["dataset"] == dataset]["tokens"]
        counter = Counter()
        for tokens in dataset_tokens:
            counter.update(tokens)

        if not counter:
            ax.text(0.5, 0.5, "No text available", ha="center", va="center")
            ax.axis("off")
            ax.set_title(f"{dataset}: Wordcloud")
            continue

        wc = WordCloud(
            width=1800,
            height=1100,
            background_color="white",
            max_words=180,
            max_font_size=140,
            min_font_size=10,
            margin=4,
            relative_scaling=0.4,
            prefer_horizontal=1.0,
            collocations=False,
            font_path=font_path,
            random_state=random_seed,
        ).generate_from_frequencies(counter)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{dataset}: Wordcloud")

        if per_dataset_dir is not None:
            per_dataset_dir.mkdir(parents=True, exist_ok=True)
            single_fig, single_ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
            single_ax.imshow(wc, interpolation="bilinear")
            single_ax.axis("off")
            single_ax.set_title(f"{dataset}: Wordcloud")
            single_fig.savefig(per_dataset_dir / f"plot_wordcloud_{slugify(dataset)}.png", dpi=240)
            plt.close(single_fig)
    hide_unused_axes(axes, len(datasets))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_sentiment_distribution(sentiment_summary: pd.DataFrame, output: Path) -> None:
    pivot = (
        sentiment_summary.pivot(index="dataset", columns="sentiment_label", values="count")
        .fillna(0)
        .reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    )
    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    cumulative = [0] * len(pivot.index)
    for label in ["negative", "neutral", "positive"]:
        values = pivot[label].values
        ax.bar(pivot.index, values, bottom=cumulative, color=colors[label], label=label.title(), alpha=0.9)
        cumulative = [c + v for c, v in zip(cumulative, values)]

    ax.set_title("Sentiment Distribution by Dataset")
    ax.set_ylabel("Number of Comments")
    ax.legend(loc="upper right")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_distinctive_terms_all(distinctive_df: pd.DataFrame, datasets: list[str], output: Path) -> None:
    fig, axes = subplot_grid(len(datasets), max_cols=2, width=8.0, height=5.0)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = distinctive_df[distinctive_df["dataset"] == dataset].copy()
        subset = subset[subset["distinctiveness_score"] > 0].head(12)
        if subset.empty:
            ax.set_title(f"{dataset}: Distinctive Terms")
            ax.text(0.5, 0.5, "No distinctive terms", ha="center", va="center")
            ax.axis("off")
            continue
        plot_df = subset.sort_values("distinctiveness_score", ascending=True)
        ax.barh(plot_df["term"], plot_df["distinctiveness_score"], color="#9467bd", alpha=0.9)
        ax.set_title(f"{dataset}: Distinctive Terms")
        ax.set_xlabel("Distinctiveness vs other datasets")
    hide_unused_axes(axes, len(datasets))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_topic_prevalence_all(topics_all: pd.DataFrame, datasets: list[str], output: Path) -> None:
    fig, axes = subplot_grid(len(datasets), max_cols=2, width=8.0, height=4.5)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = topics_all[topics_all["dataset"] == dataset].copy()
        if subset.empty:
            ax.set_title(f"{dataset}: Topic Prevalence")
            ax.text(0.5, 0.5, "No topics extracted", ha="center", va="center")
            ax.axis("off")
            continue
        labels = [f"T{int(topic_id)}" for topic_id in subset["topic_id"]]
        ax.bar(labels, subset["topic_prevalence"], color="#2ca02c", alpha=0.85)
        ax.set_ylim(0, max(0.2, float(subset["topic_prevalence"].max() * 1.2)))
        ax.set_title(f"{dataset}: Topic Prevalence")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Avg topic probability")
    hide_unused_axes(axes, len(datasets))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_report(
    summary_df: pd.DataFrame,
    sentiment_summary_df: pd.DataFrame,
    distinctive_df: pd.DataFrame,
    topics_all_df: pd.DataFrame,
    datasets: list[str],
    output_path: Path,
) -> None:
    def dataframe_to_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows_"
        columns = [str(col) for col in df.columns]
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body_lines = []
        for _, row in df.iterrows():
            values = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in df.columns]
            body_lines.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator] + body_lines)

    lines: list[str] = []
    lines.append("# Election Text Exploration Report")
    lines.append("")
    lines.append("## Datasets Processed")
    for dataset in datasets:
        lines.append(f"- {dataset}")
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append(dataframe_to_markdown(summary_df))
    lines.append("")

    lines.append("## Sentiment Summary")
    lines.append("Lexicon-based sentiment over cleaned tokens (`positive`, `neutral`, `negative`).")
    lines.append("")
    lines.append(dataframe_to_markdown(sentiment_summary_df))
    lines.append("")

    lines.append("## Top Distinctive Terms")
    lines.append("Distinctiveness score is each term's relative frequency in one dataset minus")
    lines.append("the average relative frequency across other datasets.")
    lines.append("")
    for dataset in datasets:
        lines.append(f"### {dataset}")
        subset = (
            distinctive_df[distinctive_df["dataset"] == dataset]
            .head(10)[["term", "count", "relative_freq", "distinctiveness_score"]]
        )
        lines.append(dataframe_to_markdown(subset))
        lines.append("")

    lines.append("## Topics")
    for dataset in datasets:
        lines.append(f"### {dataset}")
        subset = topics_all_df[topics_all_df["dataset"] == dataset][["topic_id", "topic_prevalence", "top_terms"]]
        if subset.empty:
            lines.append("No topics extracted.")
        else:
            lines.append(dataframe_to_markdown(subset))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA + Topic modeling for election text datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing CSV files (used when --input-files is not provided).",
    )
    parser.add_argument(
        "--input-files",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit CSV files to analyze.",
    )
    parser.add_argument("--topics", type=int, default=5, help="Requested number of LDA topics per dataset.")
    parser.add_argument("--top-words", type=int, default=10, help="Number of top words per topic.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/election_text_analysis",
        help="Output folder for CSV/plots/report.",
    )
    parser.add_argument(
        "--wordcloud-seed",
        type=int,
        default=7,
        help="Random seed for wordcloud layout. Use a different value for a new layout.",
    )
    args = parser.parse_args()

    data_dir = resolve_input_path(args.data_dir)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_outputs(output_dir)

    dataset_files = discover_dataset_files(data_dir=data_dir, input_files=args.input_files)
    datasets = list(dataset_files.keys())
    print("Datasets:")
    for dataset, path in dataset_files.items():
        print(f"  - {dataset}: {path}")
    print(f"Output dir: {output_dir}")

    font_name = configure_font()
    font_name, font_path = ensure_bangla_font(font_name)
    print(f"Using font: {font_name}")
    if font_path:
        print(f"Wordcloud font path: {font_path}")
    else:
        print("Warning: No explicit Bangla font path found. Rendering quality may degrade.")

    if not pil_features.check("raqm"):
        print(
            "Warning: Pillow is missing libraqm (raqm=False). "
            "Complex Bangla shaping can appear broken in wordcloud text."
        )

    data_df = build_dataframe(dataset_files)
    data_df = add_sentiment_features(data_df)
    data_df.to_csv(output_dir / "cleaned_documents.csv", index=False, encoding="utf-8")

    summary_df = compute_summary(data_df)
    summary_df.to_csv(output_dir / "summary_stats.csv", index=False, encoding="utf-8")

    sentiment_summary_df = build_sentiment_summary(data_df)
    sentiment_summary_df.to_csv(output_dir / "sentiment_summary.csv", index=False, encoding="utf-8")

    data_df[
        [
            "dataset",
            "doc_id",
            "raw_text",
            "clean_text",
            "sentiment_label",
            "sentiment_score",
            "positive_hits",
            "negative_hits",
        ]
    ].to_csv(output_dir / "document_sentiment.csv", index=False, encoding="utf-8")

    top_terms_all_df = build_top_terms_all(data_df, datasets, top_n=40)
    top_terms_all_df.to_csv(output_dir / "top_terms_all.csv", index=False, encoding="utf-8")
    for dataset in datasets:
        slug = slugify(dataset)
        top_terms_dataset = top_terms_all_df[top_terms_all_df["dataset"] == dataset]
        top_terms_dataset.to_csv(output_dir / f"top_terms_{slug}.csv", index=False, encoding="utf-8")

    distinctive_df = build_distinctive_terms(data_df, datasets)
    distinctive_df.to_csv(output_dir / "term_comparison_full.csv", index=False, encoding="utf-8")

    topics_frames: list[pd.DataFrame] = []
    doc_topic_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        topics_df, doc_topics_df = run_lda_for_dataset(
            data_df,
            dataset=dataset,
            requested_topics=args.topics,
            top_words=args.top_words,
        )
        slug = slugify(dataset)
        topics_df.to_csv(output_dir / f"topics_{slug}.csv", index=False, encoding="utf-8")
        doc_topics_df.to_csv(output_dir / f"document_topics_{slug}.csv", index=False, encoding="utf-8")
        if not topics_df.empty:
            topics_frames.append(topics_df)
        if not doc_topics_df.empty:
            doc_topic_frames.append(doc_topics_df)

    topics_all_df = pd.concat(topics_frames, ignore_index=True) if topics_frames else pd.DataFrame()
    doc_topics_all_df = pd.concat(doc_topic_frames, ignore_index=True) if doc_topic_frames else pd.DataFrame()
    topics_all_df.to_csv(output_dir / "topics_all.csv", index=False, encoding="utf-8")
    doc_topics_all_df.to_csv(output_dir / "document_topics_all.csv", index=False, encoding="utf-8")

    plot_top_terms_all(top_terms_all_df, datasets, output_dir / "plot_top_terms.png")
    plot_wordclouds_all(
        data_df,
        datasets,
        output=output_dir / "plot_wordcloud.png",
        font_path=font_path,
        per_dataset_dir=output_dir,
        random_seed=args.wordcloud_seed,
    )
    plot_sentiment_distribution(sentiment_summary_df, output_dir / "plot_sentiment_distribution.png")
    plot_length_distribution_all(data_df, datasets, output_dir / "plot_length_distribution.png")
    plot_distinctive_terms_all(distinctive_df, datasets, output_dir / "plot_distinctive_terms.png")
    plot_topic_prevalence_all(topics_all_df, datasets, output_dir / "plot_topic_prevalence.png")

    write_report(
        summary_df=summary_df,
        sentiment_summary_df=sentiment_summary_df,
        distinctive_df=distinctive_df,
        topics_all_df=topics_all_df,
        datasets=datasets,
        output_path=output_dir / "report.md",
    )

    print(f"Saved outputs to: {output_dir}")
    print("Key files:")
    print(f"  - {output_dir / 'report.md'}")
    print(f"  - {output_dir / 'summary_stats.csv'}")
    print(f"  - {output_dir / 'sentiment_summary.csv'}")
    print(f"  - {output_dir / 'topics_all.csv'}")
    print(f"  - {output_dir / 'plot_top_terms.png'}")
    print(f"  - {output_dir / 'plot_wordcloud.png'}")
    print(f"  - {output_dir / 'plot_sentiment_distribution.png'}")


if __name__ == "__main__":
    main()
