#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
import re
from pathlib import Path


def slugify(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return out or "tab"


def relpath_for_html(from_html: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=from_html.parent.resolve())


def make_embed_block(path_ref: str, media_type: str) -> str:
    safe_ref = html.escape(path_ref, quote=True)
    if media_type == "iframe":
        return f'<iframe src="{safe_ref}" class="viz-frame" loading="lazy"></iframe>'
    if media_type == "image":
        return f'<img src="{safe_ref}" class="viz-image" alt="Visualization"/>'
    return f'<object data="{safe_ref}" type="image/svg+xml" class="viz-object"></object>'


def resolve_map_path(base: Path, map_path: Path | None) -> Path:
    if map_path is not None:
        return map_path
    preferred = base / "outputs/notebook_assets/bangladesh_interactive_location_map.html"
    fallback = base / "outputs/notebook_assets/bangladesh_interactive_location_map_test.html"
    return preferred if preferred.exists() else fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tabbed HTML dashboard from generated map and chart assets.")
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("outputs/dashboard/tabbed_visual_dashboard.html"),
        help="Output HTML file for the dashboard.",
    )
    parser.add_argument(
        "--map-html",
        type=Path,
        default=None,
        help="Path to interactive map HTML. If omitted, script auto-detects common map paths.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_html = (project_root / args.output_html).resolve() if not args.output_html.is_absolute() else args.output_html.resolve()
    map_html = resolve_map_path(project_root, args.map_html.resolve() if args.map_html else None)

    visuals: list[dict[str, str]] = []

    candidates = [
        ("Bangladesh Map", map_html, "iframe"),
        (
            "Bangladesh Sentiment Map",
            project_root / "outputs/notebook_assets/bangladesh_interactive_sentiment_map.html",
            "iframe",
        ),
        ("Top Locations Overall", project_root / "outputs/location_analytics/dashboard/chart_top_locations_overall.svg", "svg"),
        ("Top Locations by Period", project_root / "outputs/location_analytics/dashboard/chart_top_locations_by_period.svg", "svg"),
        ("Sentiment by Top Locations", project_root / "outputs/location_analytics/dashboard/chart_sentiment_top_locations.svg", "svg"),
        ("Text Sentiment Distribution", project_root / "outputs/election_text_analysis/plot_sentiment_distribution.png", "image"),
        ("Text Top Terms", project_root / "outputs/election_text_analysis/plot_top_terms.png", "image"),
        ("Topics by Dataset", project_root / "outputs/election_text_analysis/topics_prevalence_by_dataset.html", "iframe"),
        ("Text Length Distribution", project_root / "outputs/election_text_analysis/plot_length_distribution.png", "image"),
        ("Text Wordcloud", project_root / "outputs/election_text_analysis/plot_wordcloud.png", "image"),
    ]

    for title, path, media in candidates:
        if path.exists():
            ref = relpath_for_html(out_html, path)
            visuals.append({"title": title, "ref": ref, "media": media})

    if not visuals:
        raise FileNotFoundError("No visual assets found. Generate map/charts first, then run this script.")

    tab_buttons = []
    tab_panels = []
    for idx, v in enumerate(visuals):
        tab_id = slugify(v["title"])
        active_class = "active" if idx == 0 else ""
        selected = "true" if idx == 0 else "false"
        tab_buttons.append(
            f'<button class="nav-link {active_class}" id="tab-{tab_id}" data-bs-toggle="tab" '
            f'data-bs-target="#panel-{tab_id}" type="button" role="tab" '
            f'aria-controls="panel-{tab_id}" aria-selected="{selected}">{html.escape(v["title"])}</button>'
        )

        panel_active = "show active" if idx == 0 else ""
        embed = make_embed_block(v["ref"], v["media"])
        tab_panels.append(
            f'<div class="tab-pane fade {panel_active}" id="panel-{tab_id}" role="tabpanel" '
            f'aria-labelledby="tab-{tab_id}">{embed}</div>'
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bangladesh Election Visual Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
  <style>
    body {{
      background: linear-gradient(135deg, #f7f9fc 0%, #eef3ff 100%);
      min-height: 100vh;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    .dashboard-wrap {{
      max-width: 1440px;
      margin: 16px auto;
      padding: 12px;
    }}
    .title-bar {{
      background: #1e293b;
      color: #fff;
      border-radius: 14px;
      padding: 16px 20px;
      margin-bottom: 12px;
    }}
    .title-bar h1 {{
      margin: 0;
      font-size: 1.4rem;
      font-weight: 700;
      text-align: center;
    }}
    .nav-tabs {{
      border-bottom: 0;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .nav-tabs .nav-link {{
      border: 0;
      border-radius: 10px;
      background: #e8eefc;
      color: #1e293b;
      font-weight: 600;
      padding: 8px 12px;
    }}
    .nav-tabs .nav-link.active {{
      background: #2563eb;
      color: #fff;
    }}
    .tab-content {{
      background: #fff;
      border-radius: 14px;
      padding: 12px;
      margin-top: 10px;
      min-height: 760px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }}
    .viz-frame, .viz-object, .viz-image {{
      width: 100%;
      height: 740px;
      border: 0;
      border-radius: 10px;
      background: #fff;
    }}
    .viz-image {{
      object-fit: contain;
      height: 740px;
    }}
    @media (max-width: 991px) {{
      .dashboard-wrap {{
        margin: 8px auto;
        padding: 8px;
      }}
      .title-bar {{
        padding: 12px 14px;
      }}
      .title-bar h1 {{
        font-size: 1.15rem;
      }}
      .tab-content {{
        min-height: auto;
        padding: 8px;
      }}
      .viz-frame, .viz-object, .viz-image {{
        height: 68vh;
        min-height: 420px;
      }}
    }}
    @media (max-width: 768px) {{
      .nav-tabs {{
        flex-wrap: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        -webkit-overflow-scrolling: touch;
        padding-bottom: 2px;
      }}
      .nav-tabs .nav-link {{
        white-space: nowrap;
        font-size: 0.95rem;
        padding: 8px 10px;
      }}
      .tab-content {{
        border-radius: 10px;
      }}
      .viz-frame, .viz-object, .viz-image {{
        height: 62vh;
        min-height: 360px;
      }}
    }}
    @media (max-width: 480px) {{
      .title-bar h1 {{
        font-size: 1.02rem;
      }}
      .viz-frame, .viz-object, .viz-image {{
        height: 58vh;
        min-height: 320px;
      }}
    }}
  </style>
</head>
<body>
  <div class="dashboard-wrap">
    <div class="title-bar">
      <h1>মানুষের মতামতের বিশ্লেষণ</h1>
    </div>
    <ul class="nav nav-tabs" id="vizTabs" role="tablist">
      {''.join(tab_buttons)}
    </ul>
    <div class="tab-content" id="vizTabsContent">
      {''.join(tab_panels)}
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")
    print(f"Saved tabbed dashboard: {out_html}")
    print(f"Tabs included: {len(visuals)}")


if __name__ == "__main__":
    main()
