import pandas as pd
import re
import string

# =========================
# 1) ANALYSIS / INSPECTION
# =========================
def analyze_current_data(df: pd.DataFrame):
    """Analyze current state of data to identify normalization needs"""
    print("=== CURRENT DATA ANALYSIS ===")
    print(f"Total articles: {len(df)}")

    for col in df.columns:
        print(f"\n📊 Column: {col}")
        print(f"   Non-null values: {df[col].notna().sum()}")
        print(f"   Null values: {df[col].isna().sum()}")

        if col in ["title", "main_text"]:
            lengths = df[col].astype(str).str.len()
            print(f"   Avg length: {lengths.mean():.0f} chars")
            print(f"   Min length: {lengths.min()}")
            print(f"   Max length: {lengths.max()}")

            text_sample = " ".join(df[col].dropna().astype(str).head(100))
            special_chars = [
                c for c in text_sample
                if c not in string.ascii_letters + string.digits + string.whitespace + string.punctuation
            ]
            if special_chars:
                print(f"   Special characters found: {set(special_chars)}")

        elif col == "publication_name":
            unique_pubs = df[col].value_counts()
            print(f"   Unique publications: {len(unique_pubs)}")
            print(f"   Top 5 publications:")
            for pub, count in unique_pubs.head().items():
                print(f"     {pub}: {count}")

        elif col == "year":
            try:
                year_series = df[col].astype(str)
                year_counts = year_series.value_counts().sort_index()
                print(f"   Year range (raw strings): {year_counts.index.min()} to {year_counts.index.max()}")

                invalid_years = df[~df[col].astype(str).str.match(r"^\d{4}$", na=False)][col].value_counts()
                if len(invalid_years) > 0:
                    print(f"   Invalid year formats (sample):")
                    print(invalid_years.head(10))
            except Exception as e:
                print(f"   Error analyzing years: {e}")
                print(f"   Sample year values: {df[col].head().tolist()}")
                print(f"   Year data types: {df[col].apply(type).value_counts()}")


# ======================
# 2) TEXT NORMALIZATION
# ======================
def basic_text_cleaning(text: str) -> str:
    """Basic text cleaning operations"""
    if pd.isna(text) or text == "":
        return ""

    text = str(text)

    # Remove extra whitespace
    text = " ".join(text.split())

    # Fix common encoding issues
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "—",
        "â€\"": "—",
        "Â": "",
        "\x92": "'",
        "\x93": '"',
        "\x94": '"',
        "\x96": "-",
        "\x97": "—",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove non-ASCII and normalize spaces
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_publication_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize publication names to handle variations"""
    print("\n=== NORMALIZING PUBLICATION NAMES ===")

    original_pubs = df["publication_name"].value_counts(dropna=False)
    print(f"Original unique publications: {len(original_pubs)}")

    df["publication_normalized"] = df["publication_name"].astype(str)

    # Apply common normalization rules
    normalization_rules = {
        r";\s*.*$": "",        # remove trailing semicolon segments
        r",\s*.*$": "",        # remove trailing comma segments
        r"\s*\(.*\)": "",      # remove parentheses content
        r"^the\s+": "",        # drop leading "the "
        r"\s+": " ",           # collapse whitespace
    }
    for pattern, replacement in normalization_rules.items():
        df["publication_normalized"] = df["publication_normalized"].str.replace(
            pattern, replacement, regex=True
        )

    df["publication_normalized"] = df["publication_normalized"].str.strip().str.title()

    new_pubs = df["publication_normalized"].value_counts()
    print(f"Normalized unique publications: {len(new_pubs)}")
    print(f"Reduction: {len(original_pubs) - len(new_pubs)} publications")

    print("\nExamples of normalization:")
    comparison_sample = (
        df[["publication_name", "publication_normalized"]]
        .drop_duplicates()
        .head(10)
    )
    for _, row in comparison_sample.iterrows():
        if row["publication_name"] != row["publication_normalized"]:
            print(f"  '{row['publication_name']}' → '{row['publication_normalized']}'")

    return df


def advanced_text_normalization(text: str, remove_stopwords: bool = True, to_lowercase: bool = True) -> str:
    """Advanced text normalization with NLP-like steps (no external deps)"""
    if pd.isna(text) or text == "":
        return ""

    text = basic_text_cleaning(str(text))
    if to_lowercase:
        text = text.lower()

    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }
    for c, e in contractions.items():
        text = text.replace(c, e)

    # Keep basic punctuation
    text = re.sub(r"[^\w\s\.\!\?\,\;\:\-]", " ", text)

    if remove_stopwords:
        stop_words = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","up",
            "about","into","through","during","before","after","above","below","between","among",
            "this","that","these","those","i","me","my","myself","we","our","ours","ourselves",
            "you","your","yours","yourself","yourselves","he","him","his","himself","she","her",
            "hers","herself","it","its","itself","they","them","their","theirs","themselves",
            "what","which","who","whom","whose","am","is","are","was","were","be","been","being",
            "have","has","had","having","do","does","did","doing","will","would","should","could",
            "can","may","might","must","shall",
        }
        words = text.split()
        text = " ".join(w for w in words if w.lower() not in stop_words)

    return " ".join(text.split())


# =========================
# 3) YEAR NORMALIZATION
# =========================
def normalize_years(df: pd.DataFrame, min_year: int = 1990, max_year: int = 2030) -> pd.DataFrame:
    """Create df['year_normalized'] with cleaned 4-digit years in a reasonable range."""
    print("\n=== NORMALIZING YEARS ===")

    def fix_spaced_year(year_val):
        """Fix years with spaces or noise like '20 21' -> '2021', '200 3' -> '2003'."""
        if pd.isna(year_val):
            return None
        s = str(year_val).strip()
        if s.lower() == "nan":
            return None

        # 1) Prefer already-correct 4-digit patterns
        m = re.search(r"\b(\d{4})\b", s)
        if m:
            cand = m.group(1)
            yi = int(cand)
            if min_year <= yi <= max_year:
                return cand

        # 2) Remove non-digits to reconstruct potential 4-digit year
        digits = re.sub(r"[^\d]", "", s)
        if len(digits) == 4:
            yi = int(digits)
            if min_year <= yi <= max_year:
                return digits

        return None

    # Start from string representation
    df["year_normalized"] = df["year"].astype(str)

    # Inspect original invalids (for logging only)
    print("Original year values (sample of invalids):")
    year_counts = df["year_normalized"].value_counts(dropna=False)
    invalid_mask_preview = ~year_counts.index.str.match(r"^\d{4}$", na=False)
    if invalid_mask_preview.any():
        print(year_counts[invalid_mask_preview].head(10))

    # Apply fixer
    df["year_normalized"] = df["year_normalized"].apply(fix_spaced_year)

    # Coerce to numeric (NaN where invalid)
    df["year_normalized"] = pd.to_numeric(df["year_normalized"], errors="coerce")

    # Reasonable range filter
    reasonable = (df["year_normalized"] >= min_year) & (df["year_normalized"] <= max_year)
    invalid = df["year_normalized"].isna() | ~reasonable

    print("\nResults:")
    print(f"   Successfully normalized years: {(~invalid).sum()}")
    print(f"   Still invalid/missing years: {invalid.sum()}")

    if invalid.sum() > 0:
        print("\nRemaining invalid year values (sample):")
        print(df.loc[invalid, "year"].value_counts().head(10))

    # Optional: convert to Int64 to allow NaN while being 'int-like'
    df["year_normalized"] = df["year_normalized"].astype("Int64")
    return df


# =========================
# 4) PIPELINE + REPORTING
# =========================
def full_normalization_pipeline(df: pd.DataFrame, normalization_level: str = "moderate") -> pd.DataFrame:
    """
    Complete normalization pipeline
    Levels: 'basic', 'moderate', 'advanced'
    """
    print(f"🔧 APPLYING '{normalization_level.upper()}' NORMALIZATION")
    print("=" * 50)

    original_size = len(df)
    df_normalized = df.copy()

    # 1) Basic text cleaning
    print("1. Basic text cleaning...")
    df_normalized["title"] = df_normalized["title"].apply(basic_text_cleaning)
    df_normalized["main_text"] = df_normalized["main_text"].apply(basic_text_cleaning)

    # 2) Publication names
    print("2. Publication name normalization...")
    df_normalized = normalize_publication_names(df_normalized)

    # 3) Years
    print("3. Year normalization...")
    df_normalized = normalize_years(df_normalized)

    # 4) Advanced text normalization (optional)
    if normalization_level in ["moderate", "advanced"]:
        print("4. Advanced text normalization...")
        remove_stops = normalization_level == "advanced"
        to_lower = normalization_level == "advanced"

        df_normalized["title_normalized"] = df_normalized["title"].apply(
            lambda x: advanced_text_normalization(x, remove_stopwords=remove_stops, to_lowercase=to_lower)
        )

        if normalization_level == "advanced":
            print("   Normalizing main text (may take a while)...")
            df_normalized["main_text_normalized"] = df_normalized["main_text"].apply(
                lambda x: advanced_text_normalization(x, remove_stopwords=False, to_lowercase=to_lower)
            )

    # Remove rows that are completely empty across essential fields
    before_cleanup = len(df_normalized)
    essential_cols = ["title", "main_text", "publication_normalized"]
    df_normalized = df_normalized[df_normalized[essential_cols].notna().any(axis=1)]
    after_cleanup = len(df_normalized)
    if before_cleanup > after_cleanup:
        print(f"5. Removed {before_cleanup - after_cleanup} completely empty articles")

    print(f"\n✅ NORMALIZATION COMPLETE")
    print(f"   Original articles: {original_size}")
    print(f"   Final articles: {len(df_normalized)}")
    return df_normalized


def compare_before_after(df_original: pd.DataFrame, df_normalized: pd.DataFrame, num_samples: int = 3):
    """Show before/after comparison"""
    print(f"\n📋 BEFORE/AFTER COMPARISON ({num_samples} samples)")
    print("=" * 60)

    sample_indices = df_original.head(num_samples).index
    for i, idx in enumerate(sample_indices, 1):
        print(f"\nSAMPLE {i}:")
        print(f"Original title: {str(df_original.loc[idx, 'title'])[:100]}...")
        if "title_normalized" in df_normalized.columns:
            print(f"Normalized title: {str(df_normalized.loc[idx, 'title_normalized'])[:100]}...")

        print(f"Original publication: {df_original.loc[idx, 'publication_name']}")
        print(f"Normalized publication: {df_normalized.loc[idx, 'publication_normalized']}")

        print(f"Original year: {df_original.loc[idx, 'year']}")
        print(f"Normalized year: {df_normalized.loc[idx, 'year_normalized']}")


# ======================
# 5) ENTRYPOINT / I/O
# ======================
def normalize_newspaper_data(
    csv_file: str = "cleaned_newspaper_data.csv",
    output_file: str = "normalized_newspaper_data.csv",
    level: str = "moderate",
) -> pd.DataFrame | None:
    """Main normalization function"""
    print("📖 Loading data...")
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"Loaded {len(df)} articles from {csv_file}")
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None

    analyze_current_data(df)
    df_normalized = full_normalization_pipeline(df, normalization_level=level)

    df_normalized.to_csv(output_file, index=False, encoding="utf-8")
    print(f"💾 Normalized data saved to: {output_file}")

    compare_before_after(df, df_normalized)
    return df_normalized


# ======================
# 6) SCRIPT USAGE
# ======================
if __name__ == "__main__":
    normalize_newspaper_data(
        csv_file="cleaned_newspaper_data.csv",
        output_file="normalized_newspaper_data.csv",
        level="advanced",  # Options: 'basic', 'moderate', 'advanced'
    )
