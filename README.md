# Bangladesh Election Text Analysis

This repository generates election text analytics and location-based visualizations for Bangladesh election-period social text.

## Current Datasets
- `data/in_use/Before Election some annotated.final.csv`
- `data/in_use/post_election_data_updated_with_location_09_march.annotated.completed.csv`
- `data/in_use/after_forming_government_data_with_location.annotated.completed.csv`

## Sentiment Labels
The workflow uses 4 sentiment classes:
- `positive`
- `negative`
- `neutral`
- `sarcastic_negative`

## Latest Updates (March 10, 2026)
- Added tabbed dashboard with map + chart visualizations.
- Added sentiment-specific Bangladesh map with selectable sentiment layers.
- Applied `x5` scaling to all dashboard visualization values.
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
.venv/bin/python scripts/location_analytics.py
.venv/bin/python scripts/build_location_dashboard_assets.py
```

Build interactive maps:

```bash
.venv/bin/python scripts/build_interactive_location_map.py --mention-multiplier 5
.venv/bin/python scripts/build_interactive_sentiment_map.py
```

Build tabbed dashboard:

```bash
.venv/bin/python scripts/build_tabbed_dashboard.py --map-html outputs/notebook_assets/bangladesh_interactive_location_map.html
```

## Publishable HTML Files
- `outputs/dashboard/tabbed_visual_dashboard.html`
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
