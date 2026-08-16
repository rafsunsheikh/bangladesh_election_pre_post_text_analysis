# Unified Dataset Coverage

Total retained comments: **26,421** across 5 periods and 8 source files.

## Per Period
| order | period | comments | with_location | location_coverage | sources |
| --- | --- | --- | --- | --- | --- |
| 1 | before_election | 1349 | 0 | 0.0% | 1 |
| 2 | after_election | 4784 | 2854 | 59.7% | 2 |
| 3 | after_forming_government | 4153 | 1648 | 39.7% | 2 |
| 4 | june_2026 | 12450 | 0 | 0.0% | 2 |
| 5 | post_june_2026 | 3685 | 3685 | 100.0% | 1 |

## Sentiment Mix (% of period)
| period | Negative | Neutral | Positive | Sarcastic_negative |
| --- | --- | --- | --- | --- |
| before_election | 38.5 | 8.5 | 19.6 | 33.4 |
| after_election | 27.2 | 7.8 | 17.9 | 47.1 |
| after_forming_government | 32.6 | 9.8 | 19.7 | 37.9 |
| june_2026 | 27.8 | 5.4 | 21.9 | 44.8 |
| post_june_2026 | 28.0 | 4.8 | 25.8 | 41.4 |

## Duplicate Resolution
- Policy: `latest` (which period a repeated comment is assigned to)
- Minimum length treated as a real duplicate: 25 characters
- Short comments exempted from de-duplication: 10,054
- Repeats dropped within a single source file: 89
- Repeats dropped across source files: 991

## Caveats
- Location analysis can only use periods with non-zero location coverage.
  A period at 0% is absent from the maps, not zero-mention.
- Sentiment marked `model_unified_fill` is inferred, not hand-annotated.
