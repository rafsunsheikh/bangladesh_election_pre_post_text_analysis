#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
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

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_TEXT_RE = re.compile(r"[^0-9a-zA-Z\u0980-\u09FF\s]")
DIGIT_RE = re.compile(r"[0-9০-৯]+")
SPACE_RE = re.compile(r"\s+")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def configure_font() -> str:
    candidates = [
        "Noto Sans Bengali",
        "Noto Serif Bengali",
        "Vrinda",
        "Mukti",
        "Kalpurush",
        "Siyam Rupali",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def get_font_path(font_name: str) -> str | None:
    for font in font_manager.fontManager.ttflist:
        if font.name == font_name:
            return font.fname
    try:
        return font_manager.findfont(font_name, fallback_to_default=True)
    except Exception:
        return None


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
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = NON_TEXT_RE.sub(" ", text)
    text = DIGIT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    tokens = text.split()
    return [token for token in tokens if len(token) > 1 and token not in stopwords]


def build_dataframe(before_file: Path, after_file: Path) -> pd.DataFrame:
    before_rows = load_single_column_csv(before_file)
    after_rows = load_single_column_csv(after_file)

    records: list[dict[str, object]] = []
    for dataset, rows in [("Before Election", before_rows), ("After Election", after_rows)]:
        for idx, raw in enumerate(rows):
            clean = normalize_text(raw)
            tokens = tokenize(clean, BANGLA_STOPWORDS)
            records.append(
                {
                    "dataset": dataset,
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


def build_top_terms(df: pd.DataFrame, dataset: str, top_n: int = 30) -> pd.DataFrame:
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


def build_term_comparison(df: pd.DataFrame) -> pd.DataFrame:
    before_counter = term_counter(df[df["dataset"] == "Before Election"]["tokens"])
    after_counter = term_counter(df[df["dataset"] == "After Election"]["tokens"])
    before_total = sum(before_counter.values()) or 1
    after_total = sum(after_counter.values()) or 1

    all_terms = set(before_counter) | set(after_counter)
    rows = []
    for term in all_terms:
        before_rel = before_counter[term] / before_total
        after_rel = after_counter[term] / after_total
        rows.append(
            {
                "term": term,
                "before_count": before_counter[term],
                "after_count": after_counter[term],
                "before_relative_freq": before_rel,
                "after_relative_freq": after_rel,
                "after_minus_before": after_rel - before_rel,
            }
        )
    comp = pd.DataFrame(rows)
    return comp.sort_values("after_minus_before", ascending=False)


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
        max_features=1000,
    )
    document_term = vectorizer.fit_transform(subset["token_text"])
    if document_term.shape[1] == 0:
        vectorizer = CountVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            min_df=1,
            max_df=1.0,
            max_features=1000,
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
            "topic_id": int(topic_idx),
            "topic_prevalence": float(topic_prevalence[topic_idx]),
            "top_terms": ", ".join(terms[i] for i in top_indices),
        }
        for rank, term_idx in enumerate(top_indices, start=1):
            topic_record[f"term_{rank}"] = terms[term_idx]
            topic_record[f"weight_{rank}"] = float(topic_weights[term_idx])
        topic_rows.append(topic_record)

    doc_rows: list[dict[str, object]] = []
    for i, (row_idx, row) in enumerate(subset.iterrows()):
        dominant_topic = int(doc_topic[i].argmax())
        doc_rows.append(
            {
                "dataset": dataset,
                "doc_id": int(row["doc_id"]),
                "raw_text": row["raw_text"],
                "clean_text": row["clean_text"],
                "dominant_topic": dominant_topic,
                "dominant_topic_prob": float(doc_topic[i][dominant_topic]),
            }
        )
    return pd.DataFrame(topic_rows), pd.DataFrame(doc_rows)


def plot_top_terms(before_terms: pd.DataFrame, after_terms: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, df, title, color in [
        (axes[0], before_terms, "Before Election: Top Terms", "#1f77b4"),
        (axes[1], after_terms, "After Election: Top Terms", "#d62728"),
    ]:
        plot_df = df.sort_values("count", ascending=True).tail(15)
        ax.barh(plot_df["term"], plot_df["count"], color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Term Frequency")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_length_distribution(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    before = df[df["dataset"] == "Before Election"]["char_len"]
    after = df[df["dataset"] == "After Election"]["char_len"]
    axes[0].hist(before, bins=25, color="#1f77b4", alpha=0.75)
    axes[0].set_title("Before Election: Character Length")
    axes[0].set_xlabel("Characters per comment")
    axes[0].set_ylabel("Count")
    axes[1].hist(after, bins=25, color="#d62728", alpha=0.75)
    axes[1].set_title("After Election: Character Length")
    axes[1].set_xlabel("Characters per comment")
    axes[1].set_ylabel("Count")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_distinctive_terms(term_comp: pd.DataFrame, output: Path) -> None:
    top_after = term_comp.head(10).copy()
    top_after["direction"] = "After > Before"
    top_before = term_comp.tail(10).copy()
    top_before["direction"] = "Before > After"
    top_before = top_before.sort_values("after_minus_before", ascending=False)

    plot_df = pd.concat([top_after, top_before], ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    colors = plot_df["direction"].map({"After > Before": "#d62728", "Before > After": "#1f77b4"})
    ax.barh(plot_df["term"], plot_df["after_minus_before"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Relative Frequency Difference (After - Before)")
    ax.set_title("Most Distinctive Terms")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_topic_prevalence(before_topics: pd.DataFrame, after_topics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, tdf, title, color in [
        (axes[0], before_topics, "Before Election: Topic Prevalence", "#1f77b4"),
        (axes[1], after_topics, "After Election: Topic Prevalence", "#d62728"),
    ]:
        if tdf.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No topics extracted", ha="center", va="center")
            ax.axis("off")
            continue
        x_labels = [f"T{int(i)}" for i in tdf["topic_id"]]
        ax.bar(x_labels, tdf["topic_prevalence"], color=color, alpha=0.85)
        ax.set_ylim(0, max(0.2, float(tdf["topic_prevalence"].max() * 1.2)))
        ax.set_title(title)
        ax.set_xlabel("Topic")
        ax.set_ylabel("Average Topic Probability")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_wordclouds(df: pd.DataFrame, output: Path, font_path: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, dataset, title in [
        (axes[0], "Before Election", "Before Election: Wordcloud"),
        (axes[1], "After Election", "After Election: Wordcloud"),
    ]:
        text = " ".join(df[df["dataset"] == dataset]["token_text"].astype(str).tolist()).strip()
        if not text:
            ax.text(0.5, 0.5, "No text available", ha="center", va="center")
            ax.axis("off")
            ax.set_title(title)
            continue

        wc = WordCloud(
            width=1400,
            height=800,
            background_color="white",
            max_words=250,
            collocations=False,
            font_path=font_path,
            regexp=r"[\w\u0980-\u09FF]+",
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title)

    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_report(
    summary_df: pd.DataFrame,
    before_topics: pd.DataFrame,
    after_topics: pd.DataFrame,
    term_comp: pd.DataFrame,
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
    lines.append("## Dataset Summary")
    lines.append(dataframe_to_markdown(summary_df))
    lines.append("")

    lines.append("## Top Distinctive Terms")
    lines.append("Terms with highest positive score are more frequent in *After Election*.")
    lines.append("Terms with highest negative score are more frequent in *Before Election*.")
    lines.append("")
    lines.append(dataframe_to_markdown(term_comp.head(10)))
    lines.append("")
    lines.append(dataframe_to_markdown(term_comp.tail(10)))
    lines.append("")

    lines.append("## Topics: Before Election")
    if before_topics.empty:
        lines.append("No topics extracted.")
    else:
        lines.append(dataframe_to_markdown(before_topics[["topic_id", "topic_prevalence", "top_terms"]]))
    lines.append("")

    lines.append("## Topics: After Election")
    if after_topics.empty:
        lines.append("No topics extracted.")
    else:
        lines.append(dataframe_to_markdown(after_topics[["topic_id", "topic_prevalence", "top_terms"]]))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA + Topic modeling for Before/After election text files.")
    parser.add_argument(
        "--before",
        type=Path,
        default=PROJECT_ROOT / "data/Before Election  - Sheet1.csv",
        help="Path to before-election CSV",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=PROJECT_ROOT / "data/After Election - Sheet1.csv",
        help="Path to after-election CSV",
    )
    parser.add_argument(
        "--topics",
        type=int,
        default=5,
        help="Requested number of LDA topics per dataset",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=10,
        help="Number of top words per topic",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/election_text_analysis",
        help="Output folder for CSV/plots/report",
    )
    args = parser.parse_args()

    before_path = resolve_input_path(args.before)
    after_path = resolve_input_path(args.after)
    output_dir = resolve_output_path(args.output_dir)

    if not before_path.exists():
        raise FileNotFoundError(f"Before-election file not found: {before_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"After-election file not found: {after_path}")

    print(f"Before file: {before_path}")
    print(f"After file: {after_path}")
    print(f"Output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    font_name = configure_font()
    print(f"Using font: {font_name}")
    font_path = get_font_path(font_name)
    if font_path:
        print(f"Wordcloud font path: {font_path}")

    data_df = build_dataframe(before_path, after_path)
    data_df.to_csv(output_dir / "cleaned_documents.csv", index=False, encoding="utf-8")

    summary_df = compute_summary(data_df)
    summary_df.to_csv(output_dir / "summary_stats.csv", index=False, encoding="utf-8")

    before_terms = build_top_terms(data_df, "Before Election", top_n=40)
    after_terms = build_top_terms(data_df, "After Election", top_n=40)
    before_terms.to_csv(output_dir / "top_terms_before.csv", index=False, encoding="utf-8")
    after_terms.to_csv(output_dir / "top_terms_after.csv", index=False, encoding="utf-8")

    term_comp = build_term_comparison(data_df)
    term_comp.to_csv(output_dir / "term_comparison_full.csv", index=False, encoding="utf-8")

    before_topics, before_doc_topics = run_lda_for_dataset(
        data_df,
        dataset="Before Election",
        requested_topics=args.topics,
        top_words=args.top_words,
    )
    after_topics, after_doc_topics = run_lda_for_dataset(
        data_df,
        dataset="After Election",
        requested_topics=args.topics,
        top_words=args.top_words,
    )
    before_topics.to_csv(output_dir / "topics_before.csv", index=False, encoding="utf-8")
    after_topics.to_csv(output_dir / "topics_after.csv", index=False, encoding="utf-8")
    before_doc_topics.to_csv(output_dir / "document_topics_before.csv", index=False, encoding="utf-8")
    after_doc_topics.to_csv(output_dir / "document_topics_after.csv", index=False, encoding="utf-8")

    plot_top_terms(before_terms, after_terms, output_dir / "plot_top_terms.png")
    plot_wordclouds(data_df, output_dir / "plot_wordcloud.png", font_path=font_path)
    plot_length_distribution(data_df, output_dir / "plot_length_distribution.png")
    plot_distinctive_terms(term_comp, output_dir / "plot_distinctive_terms.png")
    plot_topic_prevalence(before_topics, after_topics, output_dir / "plot_topic_prevalence.png")

    write_report(
        summary_df=summary_df,
        before_topics=before_topics,
        after_topics=after_topics,
        term_comp=term_comp,
        output_path=output_dir / "report.md",
    )

    print(f"Saved outputs to: {output_dir}")
    print("Key files:")
    print(f"  - {output_dir / 'report.md'}")
    print(f"  - {output_dir / 'topics_before.csv'}")
    print(f"  - {output_dir / 'topics_after.csv'}")
    print(f"  - {output_dir / 'plot_top_terms.png'}")
    print(f"  - {output_dir / 'plot_wordcloud.png'}")
    print(f"  - {output_dir / 'plot_topic_prevalence.png'}")


if __name__ == "__main__":
    main()
