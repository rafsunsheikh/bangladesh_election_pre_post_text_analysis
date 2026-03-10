#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import folium
import pandas as pd

DISTRICT_COORDS: Dict[str, Tuple[float, float]] = {
    "Dhaka": (23.8103, 90.4125), "Gazipur": (24.0023, 90.4264), "Narayanganj": (23.6238, 90.5000),
    "Narsingdi": (23.9220, 90.7177), "Tangail": (24.2513, 89.9167), "Manikganj": (23.8617, 90.0003),
    "Munshiganj": (23.5422, 90.5305), "Rajbari": (23.7574, 89.6445), "Faridpur": (23.6070, 89.8429),
    "Gopalganj": (23.0051, 89.8266), "Madaripur": (23.1641, 90.1897), "Shariatpur": (23.2423, 90.4348),
    "Kishoreganj": (24.4449, 90.7766), "Mymensingh": (24.7471, 90.4203), "Jamalpur": (24.9375, 89.9378),
    "Sherpur": (25.0205, 90.0153), "Netrokona": (24.8835, 90.7279), "Chattogram": (22.3569, 91.7832),
    "Cox's Bazar": (21.4272, 92.0058), "Rangamati": (22.7324, 92.2985), "Bandarban": (22.1953, 92.2184),
    "Khagrachhari": (23.1193, 91.9847), "Cumilla": (23.4607, 91.1809), "Brahmanbaria": (23.9571, 91.1117),
    "Chandpur": (23.2333, 90.6713), "Feni": (23.0159, 91.3976), "Noakhali": (22.8246, 91.1017),
    "Lakshmipur": (22.9447, 90.8282), "Sylhet": (24.8949, 91.8687), "Moulvibazar": (24.4829, 91.7774),
    "Habiganj": (24.3745, 91.4155), "Sunamganj": (25.0658, 91.3950), "Rajshahi": (24.3745, 88.6042),
    "Naogaon": (24.7936, 88.9318), "Natore": (24.4206, 89.0003), "Chapainawabganj": (24.5965, 88.2775),
    "Pabna": (24.0064, 89.2372), "Sirajganj": (24.4534, 89.7007), "Bogura": (24.8465, 89.3776),
    "Joypurhat": (25.0968, 89.0230), "Rangpur": (25.7439, 89.2752), "Gaibandha": (25.3290, 89.5403),
    "Kurigram": (25.8054, 89.6362), "Lalmonirhat": (25.9923, 89.2847), "Nilphamari": (25.9318, 88.8560),
    "Dinajpur": (25.6279, 88.6332), "Thakurgaon": (26.0337, 88.4617), "Panchagarh": (26.3411, 88.5542),
    "Khulna": (22.8456, 89.5403), "Jessore": (23.1664, 89.2081), "Jhenaidah": (23.5448, 89.1539),
    "Magura": (23.4855, 89.4198), "Narail": (23.1725, 89.5127), "Satkhira": (22.7185, 89.0705),
    "Bagerhat": (22.6516, 89.7859), "Kushtia": (23.9013, 89.1205), "Chuadanga": (23.6402, 88.8418),
    "Meherpur": (23.7622, 88.6318), "Barishal": (22.7010, 90.3535), "Bhola": (22.6859, 90.6482),
    "Patuakhali": (22.3596, 90.3296), "Pirojpur": (22.5781, 89.9787), "Jhalokati": (22.6406, 90.1987),
    "Barguna": (22.1592, 90.1250),
}

ALIASES = {
    "barisal": "Barishal", "বরিশাল": "Barishal", "chittagong": "Chattogram", "চট্টগ্রাম": "Chattogram",
    "comilla": "Cumilla", "কুমিল্লা": "Cumilla", "bogra": "Bogura", "বগুড়া": "Bogura", "বগুড়া": "Bogura",
    "jashore": "Jessore", "যশোর": "Jessore", "নড়াইল": "Narail", "নড়াইল": "Narail",
}

DIVISION_COLORS = {
    "Barisal": "#1f77b4", "Chittagong": "#ff7f0e", "Dhaka": "#2ca02c", "Khulna": "#d62728",
    "Rajshahi": "#9467bd", "Sylhet": "#8c564b", "Rangpur": "#17becf", "Mymensingh": "#bcbd22",
}


def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    d = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c.lower().strip() in d:
            return d[c.lower().strip()]
    return None


def norm_loc(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    k = re.sub(r"\s+", " ", s.lower()).replace(" জেলা", "").strip()
    if k in ALIASES:
        return ALIASES[k]
    if s in DISTRICT_COORDS:
        return s
    t = s.title()
    if t in DISTRICT_COORDS:
        return t
    return ALIASES.get(s, s)


def classify_dataset_group(stem: str) -> str:
    s = stem.lower().strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    if "forming" in s or "government" in s:
        return "post_forming_government"
    if "post election" in s or "after election" in s or "post_election" in stem.lower():
        return "post_election_upto_forming"
    return "other"


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
    ap.add_argument("--output-html", default="outputs/notebook_assets/bangladesh_interactive_location_map.html")
    ap.add_argument("--mention-multiplier", type=float, default=1.0)
    ap.add_argument(
        "--division-geojson",
        default="data/in_use/bangladesh_map_json_files/bangladesh.geojson",
    )
    return ap.parse_args()


def build_mentions(loc_df: pd.DataFrame, group: Optional[str] = None, mention_multiplier: float = 1.0) -> pd.DataFrame:
    subset = loc_df if group is None else loc_df[loc_df["dataset_group"] == group]
    pressure = (
        subset.groupby("location", as_index=False)
        .agg(mentions=("location", "size"), sample_comment=("comment", "first"))
    )
    pressure["mentions"] = (pressure["mentions"] * mention_multiplier).round().astype(int)
    pressure["lat"] = pressure["location"].map(lambda z: DISTRICT_COORDS.get(z, (None, None))[0])
    pressure["lon"] = pressure["location"].map(lambda z: DISTRICT_COORDS.get(z, (None, None))[1])
    pressure = pressure[pressure["lat"].notna()].sort_values("mentions", ascending=False)
    return pressure


def add_mentions_layer(
    m: folium.Map,
    pressure: pd.DataFrame,
    layer_name: str,
    show: bool = True,
) -> None:
    layer = folium.FeatureGroup(name=layer_name, show=show)
    layer.add_to(m)
    if pressure.empty:
        return

    max_m = max(int(pressure["mentions"].max()), 1)
    min_m = min(int(pressure["mentions"].min()), max_m)

    def hex_color_for_mentions(mentions: int) -> str:
        if max_m == min_m:
            t = 1.0
        else:
            t = (mentions - min_m) / (max_m - min_m)
        # Bright gradient: yellow -> orange -> vivid red.
        r = 255
        g = int(220 - 170 * t)
        b = int(60 - 40 * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    for r in pressure.itertuples(index=False):
        norm = (r.mentions / max_m) ** 0.5
        diameter = int(24 + 72 * norm)
        color = hex_color_for_mentions(int(r.mentions))
        popup = (
            f"<b>{r.location}</b><br>Mentions: {r.mentions}<br>"
            f"<i>{str(r.sample_comment)[:180]}</i>"
        )
        folium.Marker(
            location=[r.lat, r.lon],
            icon=folium.DivIcon(
                html=(
                    f'<div class="mention-bubble" style="'
                    f"width:{diameter}px;height:{diameter}px;"
                    f"background:{color};opacity:0.88;"
                    f'">{int(r.mentions)}</div>'
                )
            ),
            tooltip=f"{r.location}: {int(r.mentions)} mentions",
            popup=folium.Popup(popup, max_width=350),
        ).add_to(layer)


def main():
    args = parse_args()

    frames = []
    for fp in args.input_files:
        p = Path(fp)
        df = pd.read_csv(p)
        df["dataset"] = p.stem
        df["dataset_group"] = classify_dataset_group(p.stem)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    loc_cols = [c for c in ["Location 1", "Location 2", "Location 3", "location"] if c in all_df.columns]
    if not loc_cols:
        raise ValueError("No location columns found.")
    sent_col = pick_col(all_df, ["Sentiment"])
    text_col = pick_col(all_df, ["comment", "Comment", "text", "Text", "post", "content"])

    parts = []
    for c in loc_cols:
        part = all_df[[c]].rename(columns={c: "location_raw"})
        part["Sentiment"] = all_df[sent_col] if sent_col else "neutral"
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

    # Bubble labels: always visible mention counts centered in circles.
    m.get_root().html.add_child(
        folium.Element(
            """
            <style>
              .mention-bubble {
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

    multiplier = max(0.0, float(args.mention_multiplier))
    post_election = build_mentions(loc_df, group="post_election_upto_forming", mention_multiplier=multiplier)
    post_forming = build_mentions(loc_df, group="post_forming_government", mention_multiplier=multiplier)
    all_data = build_mentions(loc_df, group=None, mention_multiplier=multiplier)

    add_mentions_layer(
        m,
        post_election,
        layer_name="Mentions: Post Election (Upto Forming Government)",
        show=False,
    )
    add_mentions_layer(
        m,
        post_forming,
        layer_name="Mentions: Post Forming Government",
        show=False,
    )
    add_mentions_layer(
        m,
        all_data,
        layer_name="Mentions: All Available Data",
        show=True,
    )

    folium.LayerControl(collapsed=False).add_to(m)
    out = Path(args.output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"Saved interactive map to: {out}")


if __name__ == "__main__":
    main()
