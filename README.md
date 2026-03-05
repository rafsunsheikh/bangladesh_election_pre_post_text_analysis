# Bangladesh Election Text Analysis

This repo runs election text EDA + topic modeling + sentiment analytics using **annotated final sentiment labels**.

## Datasets Used
By default, `scripts/election_text_exploration.py` now analyzes only:

- `data/Before Election some annotated.final.csv`
- `data/After Election.annotated.final.csv`
- `data/After Forming Government.annotated.final.csv`

## Sentiment Logic
The pipeline supports 4 sentiment classes:

- `negative`
- `sarcastic_negative`
- `neutral`
- `positive`

For the three annotated final datasets above:

- existing `Sentiment` labels are used as the primary sentiment source
- existing `sentiment_confidence` / `sentiment_source` are preserved when available
- model inference is only used as fallback when sentiment labels are missing

## Run

```bash
.venv/bin/python scripts/election_text_exploration.py
```

Optional: override input files manually.

```bash
.venv/bin/python scripts/election_text_exploration.py \
  --input-files \
  "data/Before Election some annotated.final.csv" \
  "data/After Election.annotated.final.csv" \
  "data/After Forming Government.annotated.final.csv"
```

## Main Outputs
Generated under `outputs/election_text_analysis/`:

- `cleaned_documents.csv`
- `summary_stats.csv`
- `sentiment_summary.csv`
- `document_sentiment.csv`
- `top_terms_all.csv`
- `term_comparison_full.csv`
- `topics_all.csv`
- `document_topics_all.csv`
- `report.md`
- plot images (`plot_sentiment_distribution.png`, `plot_top_terms.png`, etc.)

## Notes
- If `models/sentiment_model.joblib` exists, it is used for fallback sentiment prediction.
- For the annotated final files listed above, sentiment labels come from the dataset itself.
