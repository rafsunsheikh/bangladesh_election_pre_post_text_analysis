#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


DISPLAY_NAME_MAP = {
    "Before Election Some": "Before Election Data",
    "Post Election Data Updated With Location 09 March": "Post Election Data upto Forming Government",
    "After Forming Government Data With Location": "After Forming Government Data",
}

DISPLAY_ORDER = [
    "Before Election Data",
    "Post Election Data upto Forming Government",
    "After Forming Government Data",
]


def sort_key(name: str) -> tuple[int, str]:
    if name in DISPLAY_ORDER:
        return (DISPLAY_ORDER.index(name), name)
    return (99, name)


def build_html(df: pd.DataFrame) -> str:
    df = df.copy()
    df["dataset_display"] = df["dataset"].map(lambda x: DISPLAY_NAME_MAP.get(str(x), str(x)))
    df["topic_prevalence_pct"] = df["topic_prevalence"] * 100.0
    groups = sorted(df["dataset_display"].dropna().unique().tolist(), key=sort_key)

    sections: list[str] = []
    for dataset in groups:
        sub = (
            df[df["dataset_display"] == dataset][["topic_id", "topic_prevalence_pct", "top_terms"]]
            .sort_values("topic_prevalence_pct", ascending=False)
            .reset_index(drop=True)
        )
        max_val = max(float(sub["topic_prevalence_pct"].max()), 0.0001) if not sub.empty else 1.0
        rows: list[str] = []
        for r in sub.itertuples(index=False):
            width = max(2.0, (float(r.topic_prevalence_pct) / max_val) * 100.0)
            rows.append(
                "<tr>"
                f"<td>T{int(r.topic_id)}</td>"
                f"<td>{float(r.topic_prevalence_pct):.2f}%</td>"
                f'<td><div class="bar-wrap"><div class="bar" style="width:{width:.2f}%"></div></div></td>'
                f"<td>{html.escape(str(r.top_terms))}</td>"
                "</tr>"
            )
        section = (
            '<section class="card">'
            f"<h2>{html.escape(dataset)}</h2>"
            '<table><thead><tr><th>Topic</th><th>Prevalence</th><th>Relative Bar</th><th>Top Terms</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>"
            "</section>"
        )
        sections.append(section)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Topics by Dataset</title>
  <style>
    body {{
      margin: 0;
      padding: 18px;
      background: #f8fafc;
      color: #1f2937;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    .container {{
      max-width: 1320px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 14px 0;
      font-size: 1.6rem;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 14px;
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }}
    h2 {{
      margin: 4px 0 10px 0;
      font-size: 1.1rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 0.93rem;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    th:nth-child(1), td:nth-child(1) {{ width: 64px; }}
    th:nth-child(2), td:nth-child(2) {{ width: 110px; }}
    th:nth-child(3), td:nth-child(3) {{ width: 220px; }}
    .bar-wrap {{
      height: 12px;
      width: 100%;
      background: #e5e7eb;
      border-radius: 8px;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      background: #2563eb;
      border-radius: 8px;
    }}
    .footnote {{
      margin-top: 10px;
      padding: 12px 14px;
      background: #eef2ff;
      border: 1px solid #c7d2fe;
      border-radius: 10px;
      font-size: 0.95rem;
      line-height: 1.45;
    }}
    .footnote p {{
      margin: 0 0 8px 0;
    }}
    .footnote p:last-child {{
      margin-bottom: 0;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Topics and Topic Prevalence by Dataset</h1>
    {''.join(sections)}
    <div class="footnote">
      <p>Topic prevalence means how much a topic appears overall in a dataset.</p>
      <p>In each dataset, each comment gets a probability distribution across topics (from LDA), and prevalence is the average probability of a topic across all comments in that dataset.</p>
      <p>So if Topic 2 has prevalence of 21%, it means roughly 21% of the context of data from the particular dataset aligns with Topic 2.</p>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML page showing topic prevalence by dataset.")
    parser.add_argument(
        "--topics-csv",
        type=Path,
        default=Path("outputs/election_text_analysis/topics_all.csv"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("outputs/election_text_analysis/topics_prevalence_by_dataset.html"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.topics_csv)
    required = {"dataset", "topic_id", "topic_prevalence", "top_terms"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {args.topics_csv}: {missing}")

    out_html = build_html(df)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(out_html, encoding="utf-8")
    print(f"Saved topics prevalence page: {args.output_html}")


if __name__ == "__main__":
    main()
