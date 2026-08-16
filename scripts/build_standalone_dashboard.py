#!/usr/bin/env python3
"""Build a single self-contained HTML dashboard.

`build_tabbed_dashboard.py` emits a small shell that pulls in ten sibling assets
over relative paths, so publishing it means copying the whole `outputs/` tree.
This script inlines everything instead — SVGs as markup, PNGs as data URIs,
folium maps and the topics page as `iframe srcdoc` — producing one file that can
be dropped anywhere.

Note: the embedded maps still fetch Leaflet and their basemap tiles from the
network at view time, so the maps need an internet connection to draw. Every
chart, image, and the page itself are fully local.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import re
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower() or "tab"


def encode_image(path: Path, max_width: int) -> str:
    """Return a data URI, downscaling oversized plots so the file stays sane."""
    with Image.open(path) as im:
        if im.width > max_width:
            height = round(im.height * max_width / im.width)
            im = im.convert("RGBA").resize((max_width, height), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def inline_svg(path: Path) -> str:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg).strip()
    # Let the container drive size rather than the file's fixed width/height.
    svg = re.sub(r'<svg([^>]*?)\swidth="[^"]*"', r"<svg\1", svg, count=1)
    svg = re.sub(r'<svg([^>]*?)\sheight="[^"]*"', r"<svg\1", svg, count=1)
    return f'<div class="viz-svg">{svg}</div>'


def inline_iframe(path: Path) -> str:
    doc = path.read_text(encoding="utf-8")
    return f'<iframe class="viz-frame" srcdoc="{html.escape(doc, quote=True)}"></iframe>'


CSS = """
*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0;
  background: linear-gradient(135deg, #f7f9fc 0%, #eef3ff 100%);
  min-height: 100vh;
  font-family: "Segoe UI", "Noto Sans Bengali", Arial, sans-serif;
  color: #1e293b;
}
.dashboard-wrap { max-width: 1440px; margin: 16px auto; padding: 12px; }
.title-bar { background: #1e293b; color: #fff; border-radius: 14px; padding: 16px 20px; margin-bottom: 12px; }
.title-bar h1 { margin: 0; font-size: 1.4rem; font-weight: 700; text-align: center; }
.title-bar p { margin: 6px 0 0; text-align: center; font-size: .82rem; opacity: .75; }
.tabs { display: flex; gap: 6px; flex-wrap: wrap; padding: 0; margin: 0; list-style: none; }
.tabs button {
  border: 0; border-radius: 10px; background: #e8eefc; color: #1e293b;
  font-weight: 600; padding: 8px 12px; cursor: pointer; font-size: .95rem;
  font-family: inherit;
}
.tabs button:hover { background: #d8e3fa; }
.tabs button[aria-selected="true"] { background: #2563eb; color: #fff; }
.panels {
  background: #fff; border-radius: 14px; padding: 12px; margin-top: 10px;
  min-height: 760px; box-shadow: 0 10px 24px rgba(15, 23, 42, .08);
}
.panel { display: none; }
.panel.active { display: block; }
.viz-frame { width: 100%; height: 740px; border: 0; border-radius: 10px; background: #fff; }
.viz-image { width: 100%; height: 740px; object-fit: contain; border-radius: 10px; background: #fff; }
.viz-svg { width: 100%; overflow-x: auto; }
.viz-svg svg { width: 100%; height: auto; max-height: 740px; display: block; }
@media (max-width: 991px) {
  .dashboard-wrap { margin: 8px auto; padding: 8px; }
  .title-bar { padding: 12px 14px; }
  .title-bar h1 { font-size: 1.15rem; }
  .panels { min-height: auto; padding: 8px; border-radius: 10px; }
  .viz-frame, .viz-image { height: 68vh; min-height: 420px; }
}
@media (max-width: 768px) {
  .tabs { flex-wrap: nowrap; overflow-x: auto; -webkit-overflow-scrolling: touch; padding-bottom: 2px; }
  .tabs button { white-space: nowrap; font-size: .9rem; padding: 8px 10px; }
}
"""

JS = """
(function () {
  var buttons = document.querySelectorAll('.tabs button');
  var panels = document.querySelectorAll('.panel');
  function select(id) {
    buttons.forEach(function (b) { b.setAttribute('aria-selected', String(b.dataset.target === id)); });
    panels.forEach(function (p) { p.classList.toggle('active', p.id === id); });
  }
  buttons.forEach(function (b) {
    b.addEventListener('click', function () { select(b.dataset.target); });
  });
})();
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-html", type=Path, default=Path("outputs/dashboard/standalone_dashboard.html")
    )
    parser.add_argument(
        "--max-image-width",
        type=int,
        default=2000,
        help="Downscale embedded PNGs wider than this. Plots render at 3600-4000px, "
        "far beyond what the 740px-tall panel shows.",
    )
    args = parser.parse_args()

    candidates = [
        ("Bangladesh Map", PROJECT_ROOT / "outputs/notebook_assets/bangladesh_interactive_location_map.html", "iframe"),
        ("Bangladesh Sentiment Map", PROJECT_ROOT / "outputs/notebook_assets/bangladesh_interactive_sentiment_map.html", "iframe"),
        ("Top Locations Overall", PROJECT_ROOT / "outputs/location_analytics/dashboard/chart_top_locations_overall.svg", "svg"),
        ("Top Locations by Period", PROJECT_ROOT / "outputs/location_analytics/dashboard/chart_top_locations_by_period.svg", "svg"),
        ("Sentiment by Top Locations", PROJECT_ROOT / "outputs/location_analytics/dashboard/chart_sentiment_top_locations.svg", "svg"),
        ("Text Sentiment Distribution", PROJECT_ROOT / "outputs/election_text_analysis/plot_sentiment_distribution.png", "image"),
        ("Text Top Terms", PROJECT_ROOT / "outputs/election_text_analysis/plot_top_terms.png", "image"),
        ("Topics by Dataset", PROJECT_ROOT / "outputs/election_text_analysis/topics_prevalence_by_dataset.html", "iframe"),
        ("Text Length Distribution", PROJECT_ROOT / "outputs/election_text_analysis/plot_length_distribution.png", "image"),
        ("Text Wordcloud", PROJECT_ROOT / "outputs/election_text_analysis/plot_wordcloud.png", "image"),
    ]

    missing = [title for title, path, _ in candidates if not path.exists()]
    visuals = [(title, path, media) for title, path, media in candidates if path.exists()]
    if not visuals:
        raise FileNotFoundError("No visual assets found. Generate the maps and charts first.")
    for title in missing:
        print(f"  skipped (not generated): {title}")

    buttons, panels = [], []
    for idx, (title, path, media) in enumerate(visuals):
        tab_id = "panel-" + slugify(title)
        if media == "image":
            body = f'<img class="viz-image" src="{encode_image(path, args.max_image_width)}" alt="{html.escape(title)}"/>'
        elif media == "svg":
            body = inline_svg(path)
        else:
            body = inline_iframe(path)

        selected = "true" if idx == 0 else "false"
        active = " active" if idx == 0 else ""
        buttons.append(
            f'<button type="button" role="tab" data-target="{tab_id}" aria-selected="{selected}">'
            f"{html.escape(title)}</button>"
        )
        panels.append(f'<div class="panel{active}" id="{tab_id}" role="tabpanel">{body}</div>')
        print(f"  embedded: {title} ({media})")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bangladesh Election Visual Dashboard</title>
<style>{CSS}</style>
</head>
<body>
<div class="dashboard-wrap">
  <div class="title-bar">
    <h1>মানুষের মতামতের বিশ্লেষণ</h1>
    <p>Bangladesh Election Text &amp; Location Analytics</p>
  </div>
  <div class="tabs" role="tablist">{"".join(buttons)}</div>
  <div class="panels">{"".join(panels)}</div>
</div>
<script>{JS}</script>
</body>
</html>
"""

    out = args.output_html if args.output_html.is_absolute() else PROJECT_ROOT / args.output_html
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"\nSaved standalone dashboard: {out}")
    print(f"  tabs: {len(visuals)}")
    print(f"  size: {out.stat().st_size / 1024 / 1024:.2f} MB (single file, no sibling assets needed)")


if __name__ == "__main__":
    main()
