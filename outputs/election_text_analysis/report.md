# Election Text Exploration Report

## Datasets Processed
- Before Election Some
- Post Election Data Updated With Location 09 March
- After Forming Government Data With Location

## Dataset Summary
| dataset | total_rows | non_empty_rows | empty_raw_rows | empty_after_clean | avg_char_length | median_char_length | avg_token_length | median_token_length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Before Election Some | 1370 | 1358 | 12 | 18 | 42.941605839416056 | 33.5 | 6.015328467153284 | 5.0 |
| After Forming Government Data With Location | 1655 | 1655 | 0 | 3 | 38.721450151057404 | 31.0 | 5.495468277945619 | 4.0 |
| Post Election Data Updated With Location 09 March | 3250 | 3249 | 1 | 7 | 38.43415384615385 | 30.0 | 5.4 | 4.0 |

## Sentiment Summary
Model-based sentiment over cleaned text (`negative`, `sarcastic_negative`, `neutral`, `positive`).

| dataset | sentiment_label | count | percentage |
| --- | --- | --- | --- |
| Before Election Some | negative | 522 | 0.381021897810219 |
| Before Election Some | sarcastic_negative | 453 | 0.33065693430656934 |
| Before Election Some | neutral | 129 | 0.09416058394160584 |
| Before Election Some | positive | 266 | 0.19416058394160585 |
| After Forming Government Data With Location | negative | 559 | 0.3377643504531722 |
| After Forming Government Data With Location | sarcastic_negative | 692 | 0.41812688821752264 |
| After Forming Government Data With Location | neutral | 144 | 0.08700906344410876 |
| After Forming Government Data With Location | positive | 260 | 0.15709969788519637 |
| Post Election Data Updated With Location 09 March | negative | 1000 | 0.3076923076923077 |
| Post Election Data Updated With Location 09 March | sarcastic_negative | 1490 | 0.4584615384615385 |
| Post Election Data Updated With Location 09 March | neutral | 305 | 0.09384615384615384 |
| Post Election Data Updated With Location 09 March | positive | 455 | 0.14 |

## Top Distinctive Terms
Distinctiveness score is each term's relative frequency in one dataset minus
the average relative frequency across other datasets.

### Before Election Some
| term | count | relative_freq | distinctiveness_score |
| --- | --- | --- | --- |
| ভোট | 105 | 0.012741172187841281 | 0.011538153717093288 |
| ইনশাআল্লাহ | 89 | 0.010799660235408324 | 0.01018290358786553 |
| ধানের | 84 | 0.010192937750273025 | 0.009608680722915392 |
| বিএনপি | 82 | 0.009950248756218905 | 0.008134483229623252 |
| আব্বাস | 64 | 0.007766047809731828 | 0.0077375577812417995 |
| শীষ | 43 | 0.0052178133721635725 | 0.005189323343673544 |
| ঢাকা | 39 | 0.004732435384055333 | 0.0045350099804727 |
| আওয়ামী | 39 | 0.004732435384055333 | 0.004030208651042454 |
| চান্দা | 35 | 0.004247057395947094 | 0.0038217117644442334 |
| মির্জা | 33 | 0.004004368401892974 | 0.003668502447555332 |

### Post Election Data Updated With Location 09 March
| term | count | relative_freq | distinctiveness_score |
| --- | --- | --- | --- |
| সমঝোতা | 164 | 0.009344729344729345 | 0.007640496249622143 |
| রাষ্ট্রপতি | 96 | 0.00547008547008547 | 0.00541511020895298 |
| মন্ত্রী | 74 | 0.004216524216524216 | 0.004051598433126745 |
| হলো | 81 | 0.004615384615384616 | 0.003932893532270573 |
| চাঁদা | 76 | 0.004330484330484331 | 0.0037351558201111105 |
| শেষ | 72 | 0.0041025641025641026 | 0.0028968107323524487 |
| নোটিশ | 50 | 0.002849002849002849 | 0.002849002849002849 |
| লাভ | 53 | 0.00301994301994302 | 0.0026730004910049594 |
| তেল | 51 | 0.002905982905982906 | 0.0022898610587634327 |
| মুক্তিযোদ্ধা | 40 | 0.002279202279202279 | 0.002279202279202279 |

### After Forming Government Data With Location
| term | count | relative_freq | distinctiveness_score |
| --- | --- | --- | --- |
| কষ্ট | 105 | 0.01154480483782298 | 0.011138560055895636 |
| খাও | 63 | 0.006926882902693787 | 0.00606848985645946 |
| হাট | 64 | 0.007036833424958769 | 0.005865050065334128 |
| খা | 61 | 0.006706981858163826 | 0.005766810917992886 |
| ঠিক | 81 | 0.008905992303463442 | 0.005428113820124456 |
| ভাই | 91 | 0.010005497526113249 | 0.005360077554883304 |
| ঘাট | 57 | 0.006267179769103903 | 0.005270028771952906 |
| খাবে | 54 | 0.005937328202308961 | 0.004818832708130903 |
| তুই | 40 | 0.0043980208905992305 | 0.0032674013180429996 |
| খেতে | 35 | 0.0038482682792743265 | 0.0031893054324701985 |

## Topics
### Before Election Some
| topic_id | topic_prevalence | top_terms |
| --- | --- | --- |
| 0 | 0.20895145078887953 | ধানের, শীষ, মানুষ, ইনশাআল্লাহ, নাই, তারেক, চোর, খুনি, তুই, বিজয় |
| 1 | 0.1930597281948385 | মানুষ, ভাই, হাদী, ঢাকা, বাটপার, জনগণ, চাই, দল, নাহিদ, রাম |
| 2 | 0.2048374734874389 | ভোট, জিন্দাবাদ, ধানের, চাঁদাবাজ, বাংলাদেশ, তারেক, রহমান, বলে, দিবে, শীষে |
| 3 | 0.18783954417918117 | আব্বাস, চান্দা, মির্জা, মনে, বিএনপি, থাকবে, খাম্বা, আপনারা, ভাই, ঠিক |
| 4 | 0.2053118033496599 | ইনশাআল্লাহ, বিএনপি, জিতবে, নাই, করতে, বি, জয়, দলে, জনগণ, ভালো |

### Post Election Data Updated With Location 09 March
| topic_id | topic_prevalence | top_terms |
| --- | --- | --- |
| 0 | 0.23186592901822967 | ঠিক, হলো, কাজ, মনে, লাভ, তেল, এত, নাই, নোটিশ, কোন |
| 1 | 0.1986605643941385 | রাষ্ট্রপতি, ভালো, দরকার, মন্ত্রী, নতুন, চাই, উনি, বিষয়ক, সংবিধান, জুলাই |
| 2 | 0.172761075969543 | আলহামদুলিল্লাহ, শেষ, দেশের, ঘাট, হাট, মারা, ঘেরাও, সারা, নাম, good |
| 3 | 0.1877596007606989 | হোক, দেওয়া, সে, কোন, করতে, আবার, মানুষ, ডিম, বড়, শাহরিয়ার |
| 4 | 0.2089528298573906 | সমঝোতা, চাঁদা, বলে, সমঝোতার, ভাই, নাই, খেলা, চাদা, খা, হা |

### After Forming Government Data With Location
| topic_id | topic_prevalence | top_terms |
| --- | --- | --- |
| 0 | 0.20997327499023483 | খাও, খা, ভালো, তুই, নাই, নেতা, মনে, জুলাই, হোক, পিছনে |
| 1 | 0.1983658016464394 | হাট, ঘাট, খাইতে, বছর, ভাই, আপনারা, কোন, খেয়ে, দরদ, আহারে |
| 2 | 0.1713302949905066 | খাবে, খাবেন, তাহলে, আপনাদের, টয়লেট, অবশ্যই, বাহ, সমস্যা, সমঝোতা, সুন্দর |
| 3 | 0.2200787468136823 | কষ্ট, বলে, তোর, খান, বেশি, তোমাদের, ভাই, নাই, গাড়ির, দেয় |
| 4 | 0.20025188155913567 | ঠিক, ভাই, খাওয়ার, জ্বালা, চাই, অধিকার, খেতে, বলেছেন, যান, ভোট |
