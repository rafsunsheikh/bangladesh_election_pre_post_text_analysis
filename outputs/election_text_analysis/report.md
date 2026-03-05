# Election Text Exploration Report

## Datasets Processed
- Before Election
- After Election
- After Forming Government

## Dataset Summary
| dataset | total_rows | non_empty_rows | empty_raw_rows | empty_after_clean | avg_char_length | median_char_length | avg_token_length | median_token_length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| After Election | 2044 | 1935 | 109 | 116 | 36.82583170254403 | 26.0 | 5.167318982387475 | 4.0 |
| After Forming Government | 2559 | 2517 | 42 | 93 | 48.95974990230559 | 33.0 | 6.533411488862837 | 4.0 |
| Before Election | 1370 | 1358 | 12 | 18 | 42.93576642335766 | 33.5 | 6.015328467153284 | 5.0 |

## Sentiment Summary
Final-annotation sentiment across four classes (`negative`, `sarcastic_negative`, `neutral`, `positive`).

| dataset | sentiment_label | count | percentage |
| --- | --- | --- | --- |
| After Election | negative | 445 | 0.2299741602067183 |
| After Election | sarcastic_negative | 885 | 0.4573643410852713 |
| After Election | neutral | 114 | 0.0589147286821705 |
| After Election | positive | 491 | 0.2537467700258398 |
| After Forming Government | negative | 806 | 0.3202224870878029 |
| After Forming Government | sarcastic_negative | 886 | 0.3520063567739372 |
| After Forming Government | neutral | 265 | 0.1052840683353198 |
| After Forming Government | positive | 560 | 0.22248708780294 |
| Before Election | negative | 522 | 0.381021897810219 |
| Before Election | sarcastic_negative | 453 | 0.3306569343065693 |
| Before Election | neutral | 129 | 0.0941605839416058 |
| Before Election | positive | 266 | 0.1941605839416058 |

![Sentiment distribution](outputs/election_text_analysis/plot_sentiment_distribution.svg)

## Location Analysis
Location analytics are computed from finalized location-labeled period datasets.

### Top Locations (All Periods)

| location | total_mentions | pressure_share |
| --- | --- | --- |
| Dhaka | 1568 | 0.7698 |
| Chattogram | 363 | 0.7989 |
| Barishal | 166 | 0.8253 |
| Khulna | 149 | 0.7785 |
| Sylhet | 118 | 0.839 |
| Cumilla | 102 | 0.7745 |
| Gazipur | 99 | 0.8485 |
| Mymensingh | 81 | 0.7901 |
| Rangpur | 80 | 0.75 |
| Noakhali | 76 | 0.7895 |
| Rajshahi | 72 | 0.6944 |
| Bogura | 47 | 0.6596 |

### Top Locations by Period

| period | location | mentions |
| --- | --- | --- |
| 12.02.26 —- 17.02.26 | Dhaka | 527 |
| 12.02.26 —- 17.02.26 | Chattogram | 98 |
| 12.02.26 —- 17.02.26 | Barishal | 46 |
| 12.02.26 —- 17.02.26 | Sylhet | 40 |
| 12.02.26 —- 17.02.26 | Gazipur | 27 |
| 12.02.26 —- 17.02.26 | Mymensingh | 26 |
| 12.02.26 —- 17.02.26 | Khulna | 25 |
| 12.02.26 —- 17.02.26 | Cumilla | 23 |
| 18.02.26 —--- 01.03.26 | Dhaka | 1041 |
| 18.02.26 —--- 01.03.26 | Chattogram | 265 |
| 18.02.26 —--- 01.03.26 | Khulna | 124 |
| 18.02.26 —--- 01.03.26 | Barishal | 120 |
| 18.02.26 —--- 01.03.26 | Cumilla | 79 |
| 18.02.26 —--- 01.03.26 | Sylhet | 78 |
| 18.02.26 —--- 01.03.26 | Gazipur | 72 |
| 18.02.26 —--- 01.03.26 | Noakhali | 57 |

### Top Co-occurring Location Pairs

| period | location_a | location_b | co_mentions |
| --- | --- | --- | --- |
| 12.02.26 —- 17.02.26 | Dhaka | Gazipur | 24 |
| 12.02.26 —- 17.02.26 | Chattogram | Dhaka | 23 |
| 12.02.26 —- 17.02.26 | Barishal | Dhaka | 19 |
| 12.02.26 —- 17.02.26 | Dhaka | Mymensingh | 17 |
| 12.02.26 —- 17.02.26 | Chattogram | Noakhali | 15 |
| 12.02.26 —- 17.02.26 | Dhaka | Rangpur | 14 |
| 12.02.26 —- 17.02.26 | Dhaka | Kishoreganj | 11 |
| 12.02.26 —- 17.02.26 | Barishal | Bhola | 9 |
| 18.02.26 —--- 01.03.26 | Chattogram | Dhaka | 62 |
| 18.02.26 —--- 01.03.26 | Dhaka | Gazipur | 62 |
| 18.02.26 —--- 01.03.26 | Chattogram | Noakhali | 45 |
| 18.02.26 —--- 01.03.26 | Barishal | Dhaka | 42 |
| 18.02.26 —--- 01.03.26 | Dhaka | Khulna | 41 |
| 18.02.26 —--- 01.03.26 | Dhaka | Mymensingh | 37 |
| 18.02.26 —--- 01.03.26 | Dhaka | Rangpur | 30 |
| 18.02.26 —--- 01.03.26 | Dhaka | Pabna | 27 |

### Interactive Bangladesh Map

- Open interactive map: [outputs/notebook_assets/bangladesh_interactive_location_map.html](outputs/notebook_assets/bangladesh_interactive_location_map.html)

## Top Distinctive Terms
Distinctiveness score is each term's relative frequency in one dataset minus
the average relative frequency across other datasets.

### Before Election
| term | count | relative_freq | distinctiveness_score |
| --- | --- | --- | --- |
| আব্বাস | 64 | 0.007766047809731828 | 0.007766047809731828 |
| ধানের | 84 | 0.010192937750273025 | 0.007516979343178214 |
| বিএনপি | 82 | 0.009950248756218905 | 0.0064489676416340964 |
| ঢাকা | 39 | 0.004732435384055333 | 0.00441849217501028 |
| শীষ | 43 | 0.0052178133721635725 | 0.004133965187936954 |
| চান্দা | 35 | 0.004247057395947094 | 0.004127433016498563 |
| মির্জা | 33 | 0.004004368401892974 | 0.0039744623070308415 |
| আওয়ামী | 39 | 0.004732435384055333 | 0.003867709796798109 |
| মানুষ | 51 | 0.006188569348380051 | 0.0036967251159032415 |
| চাঁদাবাজ | 29 | 0.003518990413784735 | 0.0031551565219645654 |

### After Election
| term | count | relative_freq | distinctiveness_score |
| --- | --- | --- | --- |
| ভোট | 188 | 0.01779965915546298 | 0.00936555251605517 |
| ইনশাআল্লাহ | 150 | 0.014201855709145995 | 0.007366533038059453 |
| বিজয় | 63 | 0.005964779397841318 | 0.004783820698800645 |
| দালাল | 55 | 0.005207347093353532 | 0.004755315435264482 |
| খবর | 49 | 0.0046392728649876916 | 0.004366677776071439 |
| দাদু | 46 | 0.0043552357508047715 | 0.004082640661888519 |
| সাদা | 42 | 0.003976519598560878 | 0.003976519598560878 |
| কাকু | 39 | 0.0036924824843779587 | 0.0033284489932967794 |
| শকুন | 31 | 0.0029350501798901724 | 0.0029350501798901724 |
| আল্লাহ | 55 | 0.005207347093353532 | 0.0027131003967468034 |

### After Forming Government
| term | count | relative_freq | distinctiveness_score |
| --- | --- | --- | --- |
| মাফিয়া | 94 | 0.005622345834080986 | 0.005622345834080986 |
| শুরু | 109 | 0.006519528679944973 | 0.005506074651920024 |
| শপথ | 90 | 0.005383097075183922 | 0.005261752578156862 |
| সংবিধান | 82 | 0.004904599557389796 | 0.004904599557389796 |
| কাজে | 75 | 0.004485914229319935 | 0.004209218445718372 |
| জুলাই | 78 | 0.004665350798492732 | 0.0038885943656202164 |
| তাহলে | 99 | 0.005921406782702315 | 0.003469792987057707 |
| সংস্কার | 55 | 0.0032896704348346194 | 0.0032896704348346194 |
| যুদ্ধ | 45 | 0.002691548537591961 | 0.0026442090185614746 |
| করতে | 95 | 0.005682158023805252 | 0.002413789606838998 |

## Topics
### Before Election
| topic_id | topic_prevalence | top_terms |
| --- | --- | --- |
| 0 | 0.20895145078887953 | ধানের, শীষ, মানুষ, ইনশাআল্লাহ, নাই, তারেক, চোর, খুনি, তুই, বিজয় |
| 1 | 0.1930597281948385 | মানুষ, ভাই, হাদী, ঢাকা, বাটপার, জনগণ, চাই, দল, নাহিদ, রাম |
| 2 | 0.2048374734874389 | ভোট, জিন্দাবাদ, ধানের, চাঁদাবাজ, বাংলাদেশ, তারেক, রহমান, বলে, দিবে, শীষে |
| 3 | 0.18783954417918117 | আব্বাস, চান্দা, মির্জা, মনে, বিএনপি, থাকবে, খাম্বা, আপনারা, ভাই, ঠিক |
| 4 | 0.2053118033496599 | ইনশাআল্লাহ, বিএনপি, জিতবে, নাই, করতে, বি, জয়, দলে, জনগণ, ভালো |

### After Election
| topic_id | topic_prevalence | top_terms |
| --- | --- | --- |
| 0 | 0.18966908798828752 | মনে, দাঁড়িপাল্লা, নাই, বলে, আলহামদুলিল্লাহ, বাটপার, ইনশাআল্লাহ, তাহলে, ভোট, তামিম |
| 1 | 0.21002895457377554 | ভোট, বাংলাদেশ, খবর, আজ, ঠিক, নতুন, শেষ, পাল্লা, কাক্কু, ভালো |
| 2 | 0.1819045636813587 | দাদু, ভোট, কাকু, নাটক, হয়ে, নাকি, রহমান, মুখে, আবার, খবর |
| 3 | 0.2073838219677034 | ভোট, দালাল, আল্লাহ, সাদা, প্রথম, শকুন, করতে, করেন, দেশের, জনগণ |
| 4 | 0.21101357178887334 | ইনশাআল্লাহ, বিজয়, ধানের, জয়, একটু, মন, শীষ, এবার, খারাপ, শীষের |

### After Forming Government
| topic_id | topic_prevalence | top_terms |
| --- | --- | --- |
| 0 | 0.20440745718418746 | শপথ, সংবিধান, আলহামদুলিল্লাহ, সংস্কার, ভোট, নির্বাচন, তাহলে, শুরু, হয়ে, অনুযায়ী |
| 1 | 0.21551754641820314 | ভালো, কাজে, চাই, দেখতে, তারেক, এটাই, ইনশাআল্লাহ, আবার, ধন্যবাদ, নয় |
| 2 | 0.20684508261093412 | করতে, যুদ্ধ, বিরুদ্ধে, জনগণ, নিজের, ঠিক, এগিয়ে, পল্টি, আপনারা, দেশের |
| 3 | 0.1755485650530031 | আল্লাহ, বলে, ভাই, নাই, বড়, করেন, কাজ, প্রথম, বাবা, জুলাই |
| 4 | 0.19768134873366952 | মাফিয়া, শুরু, মনে, নিজেই, সিদ্ধান্ত, জুলাই, বিএনপি, তাহলে, নিজে, সে |
