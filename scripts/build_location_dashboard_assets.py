#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import pandas as pd

DISTRICT_COORDS = {
    "Dhaka": (23.8103, 90.4125),
    "Gazipur": (24.0023, 90.4264),
    "Narayanganj": (23.6238, 90.5000),
    "Narsingdi": (23.9220, 90.7177),
    "Tangail": (24.2513, 89.9167),
    "Manikganj": (23.8617, 90.0003),
    "Munshiganj": (23.5422, 90.5305),
    "Rajbari": (23.7574, 89.6445),
    "Faridpur": (23.6070, 89.8429),
    "Gopalganj": (23.0051, 89.8266),
    "Madaripur": (23.1641, 90.1897),
    "Shariatpur": (23.2423, 90.4348),
    "Kishoreganj": (24.4449, 90.7766),
    "Mymensingh": (24.7471, 90.4203),
    "Jamalpur": (24.9375, 89.9378),
    "Sherpur": (25.0205, 90.0153),
    "Netrokona": (24.8835, 90.7279),
    "Chattogram": (22.3569, 91.7832),
    "Cox's Bazar": (21.4272, 92.0058),
    "Rangamati": (22.7324, 92.2985),
    "Bandarban": (22.1953, 92.2184),
    "Khagrachhari": (23.1193, 91.9847),
    "Cumilla": (23.4607, 91.1809),
    "Brahmanbaria": (23.9571, 91.1117),
    "Chandpur": (23.2333, 90.6713),
    "Feni": (23.0159, 91.3976),
    "Noakhali": (22.8246, 91.1017),
    "Lakshmipur": (22.9447, 90.8282),
    "Sylhet": (24.8949, 91.8687),
    "Moulvibazar": (24.4829, 91.7774),
    "Habiganj": (24.3745, 91.4155),
    "Sunamganj": (25.0658, 91.3950),
    "Rajshahi": (24.3745, 88.6042),
    "Naogaon": (24.7936, 88.9318),
    "Natore": (24.4206, 89.0003),
    "Chapainawabganj": (24.5965, 88.2775),
    "Pabna": (24.0064, 89.2372),
    "Sirajganj": (24.4534, 89.7007),
    "Bogura": (24.8465, 89.3776),
    "Joypurhat": (25.0968, 89.0230),
    "Rangpur": (25.7439, 89.2752),
    "Gaibandha": (25.3290, 89.5403),
    "Kurigram": (25.8054, 89.6362),
    "Lalmonirhat": (25.9923, 89.2847),
    "Nilphamari": (25.9318, 88.8560),
    "Dinajpur": (25.6279, 88.6332),
    "Thakurgaon": (26.0337, 88.4617),
    "Panchagarh": (26.3411, 88.5542),
    "Khulna": (22.8456, 89.5403),
    "Jessore": (23.1664, 89.2081),
    "Jhenaidah": (23.5448, 89.1539),
    "Magura": (23.4855, 89.4198),
    "Narail": (23.1725, 89.5127),
    "Satkhira": (22.7185, 89.0705),
    "Bagerhat": (22.6516, 89.7859),
    "Kushtia": (23.9013, 89.1205),
    "Chuadanga": (23.6402, 88.8418),
    "Meherpur": (23.7622, 88.6318),
    "Barishal": (22.7010, 90.3535),
    "Bhola": (22.6859, 90.6482),
    "Patuakhali": (22.3596, 90.3296),
    "Pirojpur": (22.5781, 89.9787),
    "Jhalokati": (22.6406, 90.1987),
    "Barguna": (22.1592, 90.1250),
}

LOCATION_ALIASES = {
    "barisal": "Barishal",
    "barishal": "Barishal",
    "বরিশাল": "Barishal",
    "barguna": "Barguna",
    "বরগুনা": "Barguna",
    "bhola": "Bhola",
    "ভোলা": "Bhola",
    "patuakhali": "Patuakhali",
    "পটুয়াখালী": "Patuakhali",
    "patuakhali ": "Patuakhali",
    "pirojpur": "Pirojpur",
    "পিরোজপুর": "Pirojpur",
    "jhalokati": "Jhalokati",
    "ঝালকাঠি": "Jhalokati",
    "dhaka": "Dhaka",
    "ঢাকা": "Dhaka",
    "gazipur": "Gazipur",
    "গাজীপুর": "Gazipur",
    "narayanganj": "Narayanganj",
    "নারায়ণগঞ্জ": "Narayanganj",
    "নারায়ণগঞ্জ": "Narayanganj",
    "narsingdi": "Narsingdi",
    "নরসিংদী": "Narsingdi",
    "tangail": "Tangail",
    "টাঙ্গাইল": "Tangail",
    "manikganj": "Manikganj",
    "মানিকগঞ্জ": "Manikganj",
    "munshiganj": "Munshiganj",
    "মুন্সিগঞ্জ": "Munshiganj",
    "rajbari": "Rajbari",
    "রাজবাড়ী": "Rajbari",
    "faridpur": "Faridpur",
    "ফরিদপুর": "Faridpur",
    "gopalganj": "Gopalganj",
    "গোপালগঞ্জ": "Gopalganj",
    "madaripur": "Madaripur",
    "মাদারীপুর": "Madaripur",
    "shariatpur": "Shariatpur",
    "শরীয়তপুর": "Shariatpur",
    "শরিয়তপুর": "Shariatpur",
    "kishoreganj": "Kishoreganj",
    "কিশোরগঞ্জ": "Kishoreganj",
    "mymensingh": "Mymensingh",
    "ময়মনসিংহ": "Mymensingh",
    "ময়মনসিংহ": "Mymensingh",
    "jamalpur": "Jamalpur",
    "জামালপুর": "Jamalpur",
    "sherpur": "Sherpur",
    "শেরপুর": "Sherpur",
    "netrokona": "Netrokona",
    "নেত্রকোনা": "Netrokona",
    "chittagong": "Chattogram",
    "chattogram": "Chattogram",
    "চট্টগ্রাম": "Chattogram",
    "coxsbazar": "Cox's Bazar",
    "cox's bazar": "Cox's Bazar",
    "কক্সবাজার": "Cox's Bazar",
    "rangamati": "Rangamati",
    "রাঙ্গামাটি": "Rangamati",
    "bandarban": "Bandarban",
    "বান্দরবান": "Bandarban",
    "khagrachhari": "Khagrachhari",
    "খাগড়াছড়ি": "Khagrachhari",
    "comilla": "Cumilla",
    "cumilla": "Cumilla",
    "কুমিল্লা": "Cumilla",
    "brahmanbaria": "Brahmanbaria",
    "ব্রাহ্মণবাড়িয়া": "Brahmanbaria",
    "chandpur": "Chandpur",
    "চাঁদপুর": "Chandpur",
    "feni": "Feni",
    "ফেনী": "Feni",
    "noakhali": "Noakhali",
    "নোয়াখালী": "Noakhali",
    "নোয়াখালী": "Noakhali",
    "lakshmipur": "Lakshmipur",
    "লক্ষ্মীপুর": "Lakshmipur",
    "sylhet": "Sylhet",
    "সিলেট": "Sylhet",
    "moulvibazar": "Moulvibazar",
    "moulvibazar ": "Moulvibazar",
    "মৌলভীবাজার": "Moulvibazar",
    "habiganj": "Habiganj",
    "হবিগঞ্জ": "Habiganj",
    "sunamganj": "Sunamganj",
    "সুনামগঞ্জ": "Sunamganj",
    "rajshahi": "Rajshahi",
    "রাজশাহী": "Rajshahi",
    "naogaon": "Naogaon",
    "নওগাঁ": "Naogaon",
    "natore": "Natore",
    "নাটোর": "Natore",
    "chapainawabganj": "Chapainawabganj",
    "চাঁপাইনবাবগঞ্জ": "Chapainawabganj",
    "pabna": "Pabna",
    "পাবনা": "Pabna",
    "sirajganj": "Sirajganj",
    "সিরাজগঞ্জ": "Sirajganj",
    "bogra": "Bogura",
    "bogura": "Bogura",
    "বগুড়া": "Bogura",
    "বগুড়া": "Bogura",
    "joypurhat": "Joypurhat",
    "জয়পুরহাট": "Joypurhat",
    "rangpur": "Rangpur",
    "রংপুর": "Rangpur",
    "gaibandha": "Gaibandha",
    "গাইবান্ধা": "Gaibandha",
    "kurigram": "Kurigram",
    "কুড়িগ্রাম": "Kurigram",
    "lalmonirhat": "Lalmonirhat",
    "লালমনিরহাট": "Lalmonirhat",
    "nilphamari": "Nilphamari",
    "নীলফামারী": "Nilphamari",
    "dinajpur": "Dinajpur",
    "দিনাজপুর": "Dinajpur",
    "thakurgaon": "Thakurgaon",
    "ঠাকুরগাঁও": "Thakurgaon",
    "panchagarh": "Panchagarh",
    "পঞ্চগড়": "Panchagarh",
    "khulna": "Khulna",
    "খুলনা": "Khulna",
    "jashore": "Jessore",
    "jessore": "Jessore",
    "যশোর": "Jessore",
    "jhenaidah": "Jhenaidah",
    "ঝিনাইদহ": "Jhenaidah",
    "magura": "Magura",
    "মাগুরা": "Magura",
    "narail": "Narail",
    "নড়াইল": "Narail",
    "নড়াইল": "Narail",
    "satkhira": "Satkhira",
    "সাতক্ষীরা": "Satkhira",
    "bagerhat": "Bagerhat",
    "বাগেরহাট": "Bagerhat",
    "kushtia": "Kushtia",
    "কুষ্টিয়া": "Kushtia",
    "কুষ্টিয়া": "Kushtia",
    "chuadanga": "Chuadanga",
    "চুয়াডাঙ্গা": "Chuadanga",
    "চুয়াডাঙ্গা": "Chuadanga",
    "meherpur": "Meherpur",
    "মেহেরপুর": "Meherpur",
}

COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_GREEN = "#2ca02c"
COLOR_RED = "#d62728"
COLOR_NEUTRAL = "#7f7f7f"


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


SPACE_RE = re.compile(r"\s+")


def normalize_location_name(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    key = SPACE_RE.sub(" ", raw.lower()).strip()
    key = key.replace("’", "'")
    if key.endswith(" জেলা"):
        key = key[:-5].strip()
    key_compact = key.replace(" ", "")

    if key in LOCATION_ALIASES:
        return LOCATION_ALIASES[key]
    if key_compact in LOCATION_ALIASES:
        return LOCATION_ALIASES[key_compact]

    if raw in DISTRICT_COORDS:
        return raw
    title_case = raw.title()
    if title_case in DISTRICT_COORDS:
        return title_case
    return raw


def svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937} .small{font-size:11px} .axis{stroke:#9ca3af;stroke-width:1}</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_horizontal_bar_chart(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    output: Path,
    color: str = COLOR_BLUE,
    top_n: int = 15,
) -> None:
    d = df.sort_values(value_col, ascending=False).head(top_n).copy()
    width, height = 1200, 780
    ml, mr, mt, mb = 260, 70, 80, 70
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    lines = svg_start(width, height)
    lines.append(f'<text x="{ml}" y="45" font-size="26" font-weight="700">{esc(title)}</text>')

    if d.empty:
        lines.append('<text x="260" y="120" font-size="16">No data available</text>')
        write_svg(output, lines)
        return

    max_v = float(d[value_col].max()) or 1.0
    n = len(d)
    row_h = plot_h / n

    lines.append(f'<line class="axis" x1="{ml}" y1="{mt+plot_h}" x2="{width-mr}" y2="{mt+plot_h}"/>')
    for i in range(6):
        tick_v = max_v * i / 5
        x = ml + (tick_v / max_v) * plot_w
        lines.append(f'<line class="axis" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+plot_h}" stroke="#f3f4f6"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{mt+plot_h+20}" text-anchor="middle">{tick_v:.0f}</text>')

    for i, row in enumerate(d.itertuples(index=False)):
        y = mt + i * row_h + row_h * 0.14
        bar_h = row_h * 0.72
        value = float(getattr(row, value_col))
        bar_w = (value / max_v) * plot_w
        label = getattr(row, label_col)
        lines.append(f'<rect x="{ml}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.9"/>')
        lines.append(f'<text class="small" x="{ml-12}" y="{y + bar_h*0.72:.1f}" text-anchor="end">{esc(label)}</text>')
        lines.append(f'<text class="small" x="{ml + bar_w + 8:.1f}" y="{y + bar_h*0.72:.1f}">{value:.0f}</text>')

    write_svg(output, lines)


def draw_grouped_bar(df: pd.DataFrame, output: Path, top_n: int = 12) -> None:
    periods = list(df["period"].dropna().unique())
    if len(periods) != 2:
        raise ValueError("Expected exactly 2 periods in location_frequency_by_period.csv")

    pivot = (
        df.pivot_table(index="location", columns="period", values="location_mentions", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    pivot["total"] = pivot[periods[0]] + pivot[periods[1]]
    pivot = pivot.sort_values("total", ascending=False).head(top_n)

    width, height = 1300, 760
    ml, mr, mt, mb = 120, 80, 90, 170
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    lines = svg_start(width, height)
    lines.append('<text x="120" y="45" font-size="26" font-weight="700">Top Locations by Period</text>')
    lines.append(f'<text x="120" y="70" class="small">{esc(periods[0])} vs {esc(periods[1])}</text>')

    if pivot.empty:
        lines.append('<text x="120" y="120" font-size="16">No data available</text>')
        write_svg(output, lines)
        return

    max_v = float(max(pivot[periods[0]].max(), pivot[periods[1]].max())) or 1.0
    n = len(pivot)
    group_w = plot_w / n
    bar_w = group_w * 0.34

    lines.append(f'<line class="axis" x1="{ml}" y1="{mt+plot_h}" x2="{width-mr}" y2="{mt+plot_h}"/>')
    lines.append(f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}"/>')

    for i in range(6):
        tick_v = max_v * i / 5
        y = mt + plot_h - (tick_v / max_v) * plot_h
        lines.append(f'<line class="axis" x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#f3f4f6"/>')
        lines.append(f'<text class="small" x="{ml-8}" y="{y+4:.1f}" text-anchor="end">{tick_v:.0f}</text>')

    for i, (_, row) in enumerate(pivot.iterrows()):
        gx = ml + i * group_w
        v1 = float(row[periods[0]])
        v2 = float(row[periods[1]])
        label = str(row["location"])
        h1 = (v1 / max_v) * plot_h
        h2 = (v2 / max_v) * plot_h
        x1 = gx + group_w * 0.13
        x2 = x1 + bar_w + group_w * 0.08
        y1 = mt + plot_h - h1
        y2 = mt + plot_h - h2
        lines.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{h1:.1f}" fill="{COLOR_BLUE}"/>')
        lines.append(f'<rect x="{x2:.1f}" y="{y2:.1f}" width="{bar_w:.1f}" height="{h2:.1f}" fill="{COLOR_ORANGE}"/>')
        lines.append(f'<text class="small" x="{gx+group_w/2:.1f}" y="{mt+plot_h+18}" text-anchor="middle" transform="rotate(30 {gx+group_w/2:.1f},{mt+plot_h+18})">{esc(label)}</text>')

    lx = width - mr - 260
    ly = 40
    lines.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{COLOR_BLUE}"/><text class="small" x="{lx+20}" y="{ly+12}">{esc(periods[0])}</text>')
    lines.append(f'<rect x="{lx}" y="{ly+22}" width="14" height="14" fill="{COLOR_ORANGE}"/><text class="small" x="{lx+20}" y="{ly+34}">{esc(periods[1])}</text>')

    write_svg(output, lines)


def draw_growth_chart(growth_df: pd.DataFrame, output: Path, top_n: int = 20) -> None:
    d = growth_df.copy()
    d = d[d["mention_delta"] != 0].sort_values("mention_delta", ascending=False)
    d = pd.concat([d.head(top_n // 2), d.tail(top_n // 2)]).drop_duplicates(subset=["location"])
    d = d.sort_values("mention_delta", ascending=True)

    width, height = 1200, 820
    ml, mr, mt, mb = 280, 80, 85, 70
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    lines = svg_start(width, height)
    lines.append('<text x="280" y="45" font-size="26" font-weight="700">Largest Mention Changes (Period B - Period A)</text>')

    if d.empty:
        lines.append('<text x="280" y="120" font-size="16">No growth data available</text>')
        write_svg(output, lines)
        return

    min_v = float(d["mention_delta"].min())
    max_v = float(d["mention_delta"].max())
    max_abs = max(abs(min_v), abs(max_v), 1.0)

    n = len(d)
    row_h = plot_h / n
    zero_x = ml + (0 - (-max_abs)) / (2 * max_abs) * plot_w

    lines.append(f'<line class="axis" x1="{zero_x:.1f}" y1="{mt}" x2="{zero_x:.1f}" y2="{mt+plot_h}" stroke="#374151"/>')

    for i, row in enumerate(d.itertuples(index=False)):
        delta = float(row.mention_delta)
        y = mt + i * row_h + row_h * 0.12
        bh = row_h * 0.75
        bw = abs(delta) / max_abs * (plot_w / 2)
        if delta >= 0:
            x = zero_x
            color = COLOR_GREEN
            value_x = x + bw + 8
            anchor = "start"
        else:
            x = zero_x - bw
            color = COLOR_RED
            value_x = x - 8
            anchor = "end"

        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.88"/>')
        lines.append(f'<text class="small" x="{ml-10}" y="{y + bh*0.72:.1f}" text-anchor="end">{esc(row.location)}</text>')
        lines.append(f'<text class="small" x="{value_x:.1f}" y="{y + bh*0.72:.1f}" text-anchor="{anchor}">{delta:.0f}</text>')

    write_svg(output, lines)


def draw_sentiment_stacked(freq_df: pd.DataFrame, sent_df: pd.DataFrame, output: Path, top_n: int = 12) -> None:
    top_locs = (
        freq_df.groupby("location", as_index=False)["location_mentions"].sum().sort_values("location_mentions", ascending=False).head(top_n)
    )
    d = sent_df[sent_df["location"].isin(top_locs["location"])].copy()
    pivot = d.pivot_table(index="location", columns="sentiment", values="comments", aggfunc="sum", fill_value=0)
    for col in ["negative", "sarcastic_negative", "neutral", "positive"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["negative", "sarcastic_negative", "neutral", "positive"]]
    pivot = pivot.loc[top_locs["location"]]

    width, height = 1200, 780
    ml, mr, mt, mb = 260, 90, 90, 70
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    lines = svg_start(width, height)
    lines.append('<text x="260" y="45" font-size="26" font-weight="700">Sentiment Composition by Top Locations</text>')

    n = len(pivot)
    if n == 0:
        lines.append('<text x="260" y="120" font-size="16">No sentiment data available</text>')
        write_svg(output, lines)
        return

    row_h = plot_h / n
    colors = {
        "negative": COLOR_RED,
        "sarcastic_negative": "#8b0000",
        "neutral": COLOR_NEUTRAL,
        "positive": COLOR_GREEN,
    }

    for i, (loc, vals) in enumerate(pivot.iterrows()):
        total = float(vals.sum()) or 1.0
        y = mt + i * row_h + row_h * 0.15
        bh = row_h * 0.7
        x = ml
        for senti in ["negative", "sarcastic_negative", "neutral", "positive"]:
            frac = float(vals[senti]) / total
            w = frac * plot_w
            if w > 0:
                lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bh:.1f}" fill="{colors[senti]}"/>')
            x += w
        lines.append(f'<text class="small" x="{ml-10}" y="{y + bh*0.72:.1f}" text-anchor="end">{esc(loc)}</text>')

    lx = width - mr - 190
    ly = 35
    for idx, senti in enumerate(["negative", "sarcastic_negative", "neutral", "positive"]):
        yy = ly + idx * 20
        lines.append(f'<rect x="{lx}" y="{yy}" width="12" height="12" fill="{colors[senti]}"/>')
        lines.append(f'<text class="small" x="{lx+18}" y="{yy+10}">{senti}</text>')

    write_svg(output, lines)


def map_projection(lat: float, lon: float, width: int, height: int, ml: int, mr: int, mt: int, mb: int) -> tuple[float, float]:
    # Bangladesh bounding box approx.
    lon_min, lon_max = 88.0, 92.8
    lat_min, lat_max = 20.6, 26.7
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    x = ml + (lon - lon_min) / (lon_max - lon_min) * plot_w
    y = mt + (lat_max - lat) / (lat_max - lat_min) * plot_h
    return x, y


def draw_bubble_map(df: pd.DataFrame, value_col: str, title: str, output: Path, top_label_n: int = 12) -> pd.DataFrame:
    width, height = 1200, 840
    ml, mr, mt, mb = 110, 80, 90, 90

    data = df.copy()
    data = data.groupby("location", as_index=False)[value_col].sum()
    data = data[data["location"].notna()].copy()
    data["location"] = data["location"].map(normalize_location_name)
    data = data[data["location"].notna()].copy()
    data["lat"] = data["location"].map(lambda x: DISTRICT_COORDS.get(x, (None, None))[0])
    data["lon"] = data["location"].map(lambda x: DISTRICT_COORDS.get(x, (None, None))[1])
    mapped = data[data["lat"].notna()].copy()
    unmapped = data[data["lat"].isna()].copy()

    lines = svg_start(width, height)
    lines.append(f'<text x="110" y="45" font-size="26" font-weight="700">{esc(title)}</text>')
    lines.append('<text x="110" y="68" class="small">Approximate district coordinates, bubble size reflects metric value.</text>')

    x0, y0 = map_projection(26.7, 88.0, width, height, ml, mr, mt, mb)
    x1, y1 = map_projection(20.6, 92.8, width, height, ml, mr, mt, mb)
    rect_x = min(x0, x1)
    rect_y = min(y0, y1)
    rect_w = abs(x1 - x0)
    rect_h = abs(y1 - y0)
    lines.append(f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{rect_w:.1f}" height="{rect_h:.1f}" fill="#f9fafb" stroke="#d1d5db"/>')

    if mapped.empty:
        lines.append('<text x="110" y="120" font-size="16">No mappable locations found</text>')
        write_svg(output, lines)
        return unmapped

    max_abs = float(mapped[value_col].abs().max()) or 1.0
    labels_df = mapped.reindex(mapped[value_col].abs().sort_values(ascending=False).index).head(top_label_n)

    for row in mapped.itertuples(index=False):
        x, y = map_projection(float(row.lat), float(row.lon), width, height, ml, mr, mt, mb)
        val = float(getattr(row, value_col))
        radius = 4 + (abs(val) / max_abs) ** 0.5 * 22
        if val >= 0:
            fill = "#2563eb"
        else:
            fill = "#dc2626"
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" opacity="0.55" stroke="white" stroke-width="1"/>')

    for row in labels_df.itertuples(index=False):
        x, y = map_projection(float(row.lat), float(row.lon), width, height, ml, mr, mt, mb)
        lines.append(f'<text class="small" x="{x+8:.1f}" y="{y-8:.1f}">{esc(row.location)} ({float(getattr(row, value_col)):.0f})</text>')

    write_svg(output, lines)
    return unmapped


def detect_periods(freq_df: pd.DataFrame) -> tuple[str, str]:
    periods = sorted(freq_df["period"].dropna().unique().tolist())
    if len(periods) != 2:
        raise ValueError(f"Expected exactly 2 periods, found {len(periods)}: {periods}")
    return periods[0], periods[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SVG charts/maps for location analytics outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/location_analytics"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/location_analytics/dashboard"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    freq = pd.read_csv(args.input_dir / "location_frequency_by_period.csv")
    overall = pd.read_csv(args.input_dir / "location_frequency_overall.csv")
    growth = pd.read_csv(args.input_dir / "location_growth.csv")
    cooc = pd.read_csv(args.input_dir / "location_cooccurrence.csv")
    sent = pd.read_csv(args.input_dir / "location_sentiment.csv")

    # Normalize location labels across Bangla/English variants before chart/map aggregation.
    freq["location"] = freq["location"].map(normalize_location_name)
    overall["location"] = overall["location"].map(normalize_location_name)
    growth["location"] = growth["location"].map(normalize_location_name)
    cooc["location_a"] = cooc["location_a"].map(normalize_location_name)
    cooc["location_b"] = cooc["location_b"].map(normalize_location_name)
    sent["location"] = sent["location"].map(normalize_location_name)

    freq = freq[freq["location"].notna()].copy()
    overall = overall[overall["location"].notna()].copy()
    growth = growth[growth["location"].notna()].copy()
    cooc = cooc[cooc["location_a"].notna() & cooc["location_b"].notna()].copy()
    sent = sent[sent["location"].notna()].copy()

    # Re-aggregate after normalization to merge duplicate variants.
    freq = (
        freq.groupby(["period", "location"], as_index=False)
        .agg(location_mentions=("location_mentions", "sum"), unique_comments=("unique_comments", "sum"), comment_share=("comment_share", "sum"))
    )
    overall = (
        overall.groupby("location", as_index=False)
        .agg(total_mentions=("total_mentions", "sum"), total_unique_comments=("total_unique_comments", "sum"))
    )
    sent = (
        sent.groupby(["period", "location", "sentiment"], as_index=False)
        .agg(comments=("comments", "sum"), share=("share", "sum"))
    )

    period_a, period_b = detect_periods(freq)

    draw_horizontal_bar_chart(
        overall,
        label_col="location",
        value_col="total_mentions",
        title="Top Locations Overall (Mentions)",
        output=args.output_dir / "chart_top_locations_overall.svg",
    )

    draw_grouped_bar(freq, output=args.output_dir / "chart_top_locations_by_period.svg")

    draw_growth_chart(growth, output=args.output_dir / "chart_growth_top_delta.svg")

    draw_sentiment_stacked(freq, sent, output=args.output_dir / "chart_sentiment_top_locations.svg")

    unmapped_a = draw_bubble_map(
        freq[freq["period"] == period_a],
        value_col="location_mentions",
        title=f"District Bubble Map: Mentions in {period_a}",
        output=args.output_dir / "map_mentions_period_a.svg",
    )

    unmapped_b = draw_bubble_map(
        freq[freq["period"] == period_b],
        value_col="location_mentions",
        title=f"District Bubble Map: Mentions in {period_b}",
        output=args.output_dir / "map_mentions_period_b.svg",
    )

    unmapped_delta = draw_bubble_map(
        growth[["location", "mention_delta"]].rename(columns={"mention_delta": "value"}),
        value_col="value",
        title="District Bubble Map: Mention Delta (Period B - Period A)",
        output=args.output_dir / "map_growth_delta.svg",
    )

    def norm_unmapped(df: pd.DataFrame, source: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["source", "location", "value"])
        metric_cols = [c for c in df.columns if c not in {"location", "lat", "lon"}]
        metric_col = metric_cols[0] if metric_cols else None
        out = pd.DataFrame()
        out["source"] = [source] * len(df)
        out["location"] = df["location"]
        out["value"] = df[metric_col] if metric_col else 0
        out = out[out["location"].notna()].copy()
        out["location"] = out["location"].astype(str).str.strip()
        out = out[(out["location"] != "") & (out["location"].str.lower() != "nan")].copy()
        return out

    unmapped = pd.concat(
        [
            norm_unmapped(unmapped_a, f"mentions_{period_a}"),
            norm_unmapped(unmapped_b, f"mentions_{period_b}"),
            norm_unmapped(unmapped_delta, "delta"),
        ],
        ignore_index=True,
    ).drop_duplicates()
    unmapped.to_csv(args.output_dir / "map_unmapped_locations.csv", index=False, encoding="utf-8")

    cooc_top = cooc.sort_values("co_mentions", ascending=False).head(25)
    cooc_top.to_csv(args.output_dir / "cooccurrence_top25.csv", index=False, encoding="utf-8")

    readme = args.output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Location Dashboard Assets",
                "",
                "Generated files:",
                "- chart_top_locations_overall.svg",
                "- chart_top_locations_by_period.svg",
                "- chart_growth_top_delta.svg",
                "- chart_sentiment_top_locations.svg",
                "- map_mentions_period_a.svg",
                "- map_mentions_period_b.svg",
                "- map_growth_delta.svg",
                "- map_unmapped_locations.csv",
                "- cooccurrence_top25.csv",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved dashboard assets to: {args.output_dir}")


if __name__ == "__main__":
    main()
