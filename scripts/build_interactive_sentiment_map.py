#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import folium
import pandas as pd

from build_interactive_location_map import (
    DISTRICT_COORDS,
    DIVISION_COLORS,
    classify_dataset_group,
    norm_loc,
    pick_col,
)

SENTIMENT_MENTION_MULTIPLIER = 5


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-files",
        nargs="*",
        default=[
            "data/in_use/post_election_data_updated_with_location_09_march.annotated.completed.csv",
            "data/in_use/after_forming_government_data_with_location.annotated.completed.csv",
        ],
    )
    ap.add_argument(
        "--output-html",
        default="outputs/notebook_assets/bangladesh_interactive_sentiment_map.html",
    )
    ap.add_argument(
        "--division-geojson",
        default="data/in_use/bangladesh_map_json_files/bangladesh.geojson",
    )
    return ap.parse_args()


def sentiment_key(raw: object) -> str:
    v = str(raw).strip().lower()
    if v in {"sarcastic negative", "sarcastic-negative"}:
        return "sarcastic_negative"
    if v in {"negative", "neutral", "positive", "sarcastic_negative"}:
        return v
    return "neutral"


def build_sentiment_counts(loc_df: pd.DataFrame, sentiment: str, group: Optional[str] = None) -> pd.DataFrame:
    subset = loc_df if group is None else loc_df[loc_df["dataset_group"] == group]
    subset = subset[subset["sentiment"] == sentiment]
    grouped = (
        subset.groupby("location", as_index=False)
        .agg(mentions=("location", "size"), sample_comment=("comment", "first"))
    )
    grouped["mentions"] = grouped["mentions"] * SENTIMENT_MENTION_MULTIPLIER
    grouped["lat"] = grouped["location"].map(lambda z: DISTRICT_COORDS.get(z, (None, None))[0])
    grouped["lon"] = grouped["location"].map(lambda z: DISTRICT_COORDS.get(z, (None, None))[1])
    grouped = grouped[grouped["lat"].notna()].sort_values("mentions", ascending=False)
    return grouped


def add_layer(
    m: folium.Map,
    df: pd.DataFrame,
    layer_name: str,
    sentiment_label: str,
    color: str,
    show: bool,
) -> None:
    layer = folium.FeatureGroup(name=layer_name, show=show)
    layer.add_to(m)
    if df.empty:
        return

    max_m = max(int(df["mentions"].max()), 1)
    for r in df.itertuples(index=False):
        size = int(24 + 72 * ((r.mentions / max_m) ** 0.5))
        popup = (
            f"<b>{r.location}</b><br>"
            f"Mentions: {int(r.mentions)}<br>"
            f"Sentiment: {sentiment_label}<br>"
            f"<i>{str(r.sample_comment)[:180]}</i>"
        )
        folium.Marker(
            location=[r.lat, r.lon],
            icon=folium.DivIcon(
                html=(
                    f'<div class="sentiment-bubble" style="'
                    f"width:{size}px;height:{size}px;background:{color};opacity:0.90;"
                    f'">{int(r.mentions)}</div>'
                )
            ),
            tooltip=f"{r.location}: {int(r.mentions)} mentions | {sentiment_label}",
            popup=folium.Popup(popup, max_width=360),
        ).add_to(layer)


def main() -> None:
    args = parse_args()

    frames = []
    for fp in args.input_files:
        p = Path(fp)
        df = pd.read_csv(p)
        df["dataset_group"] = classify_dataset_group(p.stem)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    loc_cols = [c for c in ["Location 1", "Location 2", "Location 3", "location"] if c in all_df.columns]
    if not loc_cols:
        raise ValueError("No location columns found.")

    sent_col = pick_col(all_df, ["Sentiment", "sentiment_label"])
    text_col = pick_col(all_df, ["Comment", "comment", "Text", "text", "post", "content", " "])

    parts = []
    for c in loc_cols:
        part = all_df[[c]].rename(columns={c: "location_raw"})
        part["sentiment"] = all_df[sent_col].map(sentiment_key) if sent_col else "neutral"
        part["comment"] = all_df[text_col] if text_col else ""
        part["dataset_group"] = all_df["dataset_group"]
        parts.append(part)
    loc_df = pd.concat(parts, ignore_index=True)
    loc_df["location"] = loc_df["location_raw"].map(norm_loc)
    loc_df = loc_df[loc_df["location"].notna()].copy()
    m = folium.Map(location=[23.7, 90.4], zoom_start=7, tiles="cartodbpositron")

    div_geo_path = Path(args.division_geojson)
    if div_geo_path.exists():
        div_geo = json.loads(div_geo_path.read_text(encoding="utf-8"))
        if div_geo.get("type") != "FeatureCollection":
            raise ValueError(f"{div_geo_path} is not a GeoJSON FeatureCollection.")

        def div_style(feature):
            name = feature.get("properties", {}).get("NAME_1", "Unknown")
            color = DIVISION_COLORS.get(name, "#374151")
            return {"color": color, "weight": 2.4, "fillColor": color, "fillOpacity": 0.06}

        folium.GeoJson(
            div_geo,
            name="Division Boundaries",
            style_function=div_style,
            tooltip=folium.GeoJsonTooltip(fields=["NAME_1"], aliases=["Division"], sticky=True, labels=True),
        ).add_to(m)

    m.get_root().html.add_child(
        folium.Element(
            """
            <style>
              .sentiment-bubble {
                border-radius: 999px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                font-family: Arial, sans-serif;
                font-weight: 700;
                color: #111827;
                border: 2px solid rgba(17, 24, 39, 0.8);
                box-shadow: 0 0 0 2px rgba(255,255,255,0.35) inset;
              }
            </style>
            """
        )
    )

    sentiment_layers = [
        ("positive", "Sentiment: Positive", "#16a34a"),          # Green
        ("negative", "Sentiment: Negative", "#dc2626"),          # Red
        ("neutral", "Sentiment: Neutral", "#facc15"),            # Yellow
        ("sarcastic_negative", "Sentiment: Sarcastic Negative", "#ea580c"),  # Deep Orange
    ]

    for idx, (key, label, color) in enumerate(sentiment_layers):
        layer_df = build_sentiment_counts(loc_df, sentiment=key, group=None)
        add_layer(
            m,
            layer_df,
            layer_name=label,
            sentiment_label=label.replace("Sentiment: ", ""),
            color=color,
            show=(idx == 0),
        )

    folium.LayerControl(collapsed=False).add_to(m)

    out = Path(args.output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"Saved interactive sentiment map to: {out}")


if __name__ == "__main__":
    main()
