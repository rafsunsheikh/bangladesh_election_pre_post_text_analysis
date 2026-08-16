# Bangladesh Election Text Analysis

This repository generates election text analytics and location-based visualizations for Bangladesh election-period social text.

## Unified Dataset

All collected data is merged into one chronological corpus of **26,421 comments across 5 periods**. Every analysis and visualization reads from it.

```bash
.venv/bin/python scripts/prepare_post_june_dataset.py   # xlsx -> in_use CSV (run first)
.venv/bin/python scripts/build_unified_dataset.py       # merge everything
```

Outputs:
- `data/unified/unified_comments.csv` — one row per retained comment, with `period`, `period_order`, `source_file`
- `data/unified/periods/<order>_<period>.csv` — per-period slices consumed by the analytics scripts
- `data/unified/coverage_report.md` — rows, location coverage, and sentiment mix per period

| order | period | comments | location coverage |
| --- | --- | --- | --- |
| 1 | `before_election` | 1,349 | 0% |
| 2 | `after_election` | 4,784 | 59.7% |
| 3 | `after_forming_government` | 4,153 | 39.7% |
| 4 | `june_2026` | 12,450 | 0% |
| 5 | `post_june_2026` | 3,685 | 100% |

Periods 1 and 4 carry no location annotations, so the maps and location analytics use periods 2, 3, and 5 only. A period at 0% coverage is **absent** from the maps, not zero-mention.

Edit the `SOURCES` registry at the top of `scripts/build_unified_dataset.py` to re-scope which files belong to which period — nothing else hardcodes the period layout.

### Duplicate handling
Some source files overlap: the cumulative `post_election_..._09_march` export re-includes 665 comments that also form the tightly-windowed `after_forming_government` file, and 943 June comments reappear in the post-June sheet. The builder resolves these with `--duplicate-policy latest` (default), assigning a repeated comment to its most recent period, which is also the one carrying the location annotations. Comments under 25 characters are exempt — short phrases like `ঠিক` or `আলহামদুলিল্লাহ` recur legitimately from different commenters, each with its own location.

## Source Datasets
- `data/in_use/Before Election some annotated.final.csv`
- `data/in_use/After Election.annotated.final.csv`
- `data/in_use/post_election_data_updated_with_location_09_march.annotated.completed.csv`
- `data/in_use/After Forming Government.annotated.final.csv`
- `data/in_use/after_forming_government_data_with_location.annotated.completed.csv`
- `data/june_data/June Data for sir-1 - Sheet1.csv`
- `data/june_data/June data number 2.csv`
- `data/in_use/post_june_2026_with_location.annotated.completed.csv`

`data/june_data/June_Data_Bangla_Only-1.csv` is intentionally excluded — every unique comment in it already appears in `June Data for sir-1`.

`post_june_2026_with_location...csv` is generated from `data/data_location_cleaned_sheet_2.xlsx` by `scripts/prepare_post_june_dataset.py`, which repairs a known upstream text corruption (`উদ্` → `উদ্দিন`, 31 rows) and fills sentiment columns from the second-pass model.

## Sentiment Labels
The workflow uses 4 sentiment classes:
- `positive`
- `negative`
- `neutral`
- `sarcastic_negative`

## Latest Updates (August 16, 2026)
- **Standalone dashboard**: `scripts/build_standalone_dashboard.py` emits one self-contained ~5.8 MB HTML file, so publishing no longer means copying the whole `outputs/` tree.
- **Map basemap shrunk 96%**: `scripts/simplify_division_geojson.py` dissolves the 463-feature boundary file to 6 division outlines, taking each map from 7.9 MB to ~656 KB.
- **Unified corpus**: `scripts/build_unified_dataset.py` merges all 8 source files into 26,421 comments across 5 chronological periods. All pipelines now default to it.
- Fixed a double-count: the cumulative `post_election_..._09_march` export shares 665 comments with `after_forming_government_data_with_location`, so both previously appeared in two periods at once and in the combined map layer.
- `location_analytics.py` now honors an annotated `Sentiment` column instead of always re-predicting, and accepts `--extra-files` / `--extra-labels` for more than two periods.
- `build_location_dashboard_assets.py` renders N period series and takes `--period-order`.
- `build_interactive_sentiment_map.py` warns instead of silently rendering rows with no sentiment value as neutral.
- Added `post_june_2026` (3,713 rows, 63 districts) via `scripts/prepare_post_june_dataset.py`.

## Latest Updates (March 10, 2026)
- Added tabbed dashboard with map + chart visualizations. [Dashboard](https://rafsunsheikh.github.io/bangladesh_2026_election_visual_analysis.html)
- Added sentiment-specific Bangladesh map with selectable sentiment layers.
- Removed these tabs from dashboard:
  - `Growth Delta`
  - `Mentions Map Period A`
  - `Mentions Map Period B`
  - `Mentions Growth Map`

## Main Commands
Run text analytics:

```bash
.venv/bin/python scripts/election_text_exploration.py
```

Run location analytics:

```bash
.venv/bin/python scripts/location_analytics.py --sentiment-model models/sentiment_model_second_pass_v2.joblib
.venv/bin/python scripts/build_location_dashboard_assets.py \
  --period-order after_election after_forming_government post_june_2026
```

Defaults now point at the unified period slices. Growth remains pairwise between `--file-a` and `--file-b` (currently `after_forming_government` → `post_june_2026`); `--extra-files` adds further periods to the frequency, sentiment, and co-occurrence outputs.

Build interactive maps:

```bash
.venv/bin/python scripts/simplify_division_geojson.py   # one-time; basemap 7.0 MB -> 261 KB
.venv/bin/python scripts/build_interactive_location_map.py --mention-multiplier 5
.venv/bin/python scripts/build_interactive_sentiment_map.py
```

The source `bangladesh.geojson` holds 463 sub-features at full survey precision, but the maps only read `NAME_1` to draw 6 division outlines at zoom 7. Dissolving and simplifying takes each map HTML from 7.9 MB to ~656 KB with no visible change.

Build tabbed dashboard:

```bash
.venv/bin/python scripts/build_tabbed_dashboard.py --map-html outputs/notebook_assets/bangladesh_interactive_location_map.html
```

## Publishing the Dashboard

### Standalone single file (recommended)

```bash
.venv/bin/python scripts/build_standalone_dashboard.py
```

Produces `outputs/dashboard/standalone_dashboard.html` — **one ~5.8 MB file** with every chart, image, and map inlined (SVGs as markup, PNGs as data URIs, maps as `iframe srcdoc`). No sibling assets, no Bootstrap CDN. Copy it anywhere and it works; open it directly from disk with no local server.

The embedded maps still fetch Leaflet and their basemap tiles from the network at view time, so the two map tabs need an internet connection to draw. Everything else is fully local.

### Multi-file version

`outputs/dashboard/tabbed_visual_dashboard.html` is a 7.8 KB shell that references 10 sibling assets over `../` relative paths, so it needs the whole `outputs/` tree alongside it and is best viewed over a local server:

```bash
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
# http://127.0.0.1:8765/outputs/dashboard/tabbed_visual_dashboard.html
```

### Other publishable files
- `outputs/notebook_assets/bangladesh_interactive_location_map.html`
- `outputs/notebook_assets/bangladesh_interactive_sentiment_map.html`

## Key Output Folders
- `outputs/election_text_analysis/`
- `outputs/location_analytics/`
- `outputs/location_analytics/dashboard/`
- `outputs/notebook_assets/`
- `outputs/dashboard/`

## Notes
- Dashboard values are visually scaled by `x5` for readability.
- Source CSV outputs in `outputs/election_text_analysis/` and `outputs/location_analytics/` remain unscaled analytical outputs unless explicitly transformed in script logic.
