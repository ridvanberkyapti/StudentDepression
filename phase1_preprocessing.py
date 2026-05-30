"""
=============================================================================
STUDENT DEPRESSION RISK ANALYSIS — PHASE 1: DATA PREPROCESSING & CLEANING
=============================================================================
Author  : [Your Name]
Course  : Probability and Statistics — Term Project
Model   : Naive Bayes Classifier (Probabilistic Modeling)
Dataset : Student Depression Dataset (N=27,901)
=============================================================================

TECHNICAL OVERVIEW
------------------
This module handles the complete data preprocessing pipeline:
  1. Raw data ingestion and structural validation
  2. Ordinal encoding for ordered categorical variables
  3. Binary & label encoding for nominal categorical variables
  4. Z-Score based outlier detection and removal for continuous features
  5. Export of a clean, model-ready DataFrame

All functions are designed to be imported by downstream modules
(EDA, Naive Bayes training, SHAP analysis, Streamlit dashboard).
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS — Centralised configuration for reproducibility
# ---------------------------------------------------------------------------

DATA_PATH = "Student_Depression_Dataset.csv"

# Continuous features subject to Z-Score outlier filtering
CONTINUOUS_FEATURES = ["Age", "CGPA", "Work/Study Hours"]

# Ordinal features with a natural, meaningful order
ORDINAL_MAPPINGS = {
    "Sleep Duration": {
        "Less than 5 hours": 0,
        "5-6 hours":         1,
        "7-8 hours":         2,
        "More than 8 hours": 3,
        "Others":            1,   # treated as middle-tier
    },
    "Dietary Habits": {
        "Unhealthy":  0,
        "Moderate":   1,
        "Healthy":    2,
        "Others":     1,
    },
}

# Nominal binary features — encoded directly to 0/1
BINARY_MAPPINGS = {
    "Gender":                           {"Male": 0, "Female": 1},
    "Have you ever had suicidal thoughts ?": {"No": 0, "Yes": 1},
    "Family History of Mental Illness": {"No": 0, "Yes": 1},
}

# High-cardinality nominal features handled via LabelEncoder
HIGH_CARDINALITY_COLS = ["City", "Profession", "Degree"]

# Target column
TARGET_COL = "Depression"

# Z-Score threshold for outlier detection (standard: ±3σ)
Z_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# STEP 1 — Data Ingestion
# ---------------------------------------------------------------------------

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV dataset and perform initial structural validation.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with shape and dtype summary printed to stdout.
    """
    df = pd.read_csv(path)
    print("=" * 60)
    print("  DATASET LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"  Missing : {df.isnull().sum().sum()} total null values")
    print("=" * 60)
    return df


# ---------------------------------------------------------------------------
# STEP 2 — Missing Value Imputation
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using statistically appropriate strategies:
      - Continuous features  → median imputation (robust to skew)
      - Categorical features → mode imputation

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with zero null values.
    """
    df = df.copy()

    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    category_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    before = df.isnull().sum().sum()

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [IMPUTE] '{col}' — filled {df[col].isnull().sum()} NaN "
                  f"with median={median_val:.4f}")

    for col in category_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [IMPUTE] '{col}' — filled NaN with mode='{mode_val}'")

    after = df.isnull().sum().sum()
    print(f"\n  Missing values: {before} --> {after}")
    return df


# ---------------------------------------------------------------------------
# STEP 3 — Encoding
# ---------------------------------------------------------------------------

def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply predefined ordinal encoding to features that carry a natural rank
    order (e.g., Sleep Duration: <5h < 5-6h < 7-8h < >8h).

    Ordinal encoding preserves the monotonic relationship, which is critical
    for distance-based and probabilistic models.
    """
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        df[col] = df[col].map(mapping)
        print(f"  [ORDINAL] '{col}' encoded: {mapping}")
    return df


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map binary nominal features to {0, 1} using predefined dictionaries.
    This avoids unnecessary integer ranges from LabelEncoder on binary vars.
    """
    df = df.copy()
    for col, mapping in BINARY_MAPPINGS.items():
        df[col] = df[col].map(mapping)
        print(f"  [BINARY] '{col}' encoded: {mapping}")
    return df


def encode_high_cardinality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply scikit-learn LabelEncoder to high-cardinality nominal columns.

    NOTE: For a Naive Bayes model, LabelEncoding is acceptable since GaussianNB
    models each feature's distribution independently (no ordinal assumptions
    are implied by the model's likelihood computation).
    """
    df = df.copy()
    le = LabelEncoder()
    for col in HIGH_CARDINALITY_COLS:
        original_uniq = df[col].nunique()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  [LABEL]  '{col}' — {original_uniq} unique categories encoded "
              f"to [0, {df[col].max()}]")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and enforce binary {0, 1} encoding on the target variable.

    The 'Depression' column already contains integer values in the raw data,
    but this step guarantees type integrity and documents intent explicitly.
    """
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    dist = df[TARGET_COL].value_counts().to_dict()
    ratio = df[TARGET_COL].mean() * 100
    print(f"  [TARGET] '{TARGET_COL}': {dist} | "
          f"Depression rate = {ratio:.2f}%")
    return df


# ---------------------------------------------------------------------------
# STEP 4 — Z-Score Outlier Detection & Removal
# ---------------------------------------------------------------------------

def remove_outliers_zscore(df: pd.DataFrame,
                           features: list = CONTINUOUS_FEATURES,
                           threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    """
    Remove rows where any continuous feature's Z-Score exceeds ±threshold.

    Mathematical Background
    -----------------------
    For a feature X with mean μ and standard deviation σ, the Z-Score is:

        Z = (x_i - μ) / σ

    A data point x_i is classified as an outlier if |Z| > threshold (3.0).
    Under a Normal distribution, |Z| > 3 captures only 0.27% of observations,
    making this a conservative and statistically principled filter.

    Parameters
    ----------
    df        : Input DataFrame (post-encoding)
    features  : List of continuous numeric columns to evaluate
    threshold : Cutoff value for Z-Score (default = 3.0)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with outlier rows removed.
    """
    df = df.copy()
    n_before = len(df)

    print(f"\n  Z-Score Outlier Removal (threshold = ±{threshold})")
    print(f"  {'Feature':<25} {'Mean':>10} {'Std':>10} "
          f"{'Min Z':>10} {'Max Z':>10} {'Outliers':>10}")
    print(f"  {'-'*75}")

    mask_clean = pd.Series([True] * len(df), index=df.index)

    for col in features:
        z_scores  = np.abs(stats.zscore(df[col].dropna()))
        col_mask  = np.abs(stats.zscore(df[col])) <= threshold
        n_outlier = (~col_mask).sum()

        print(f"  {col:<25} {df[col].mean():>10.3f} {df[col].std():>10.3f} "
              f"{z_scores.min():>10.3f} {z_scores.max():>10.3f} "
              f"{n_outlier:>10,}")

        mask_clean &= col_mask

    df_clean = df[mask_clean].reset_index(drop=True)
    n_after  = len(df_clean)

    print(f"\n  Rows before : {n_before:,}")
    print(f"  Rows after  : {n_after:,}")
    print(f"  Removed     : {n_before - n_after:,} "
          f"({(n_before - n_after) / n_before * 100:.2f}%)")
    return df_clean


# ---------------------------------------------------------------------------
# STEP 5 — Drop Irrelevant Columns
# ---------------------------------------------------------------------------

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that carry no predictive signal:
      - 'id'  : Arbitrary row identifier — not a feature
    """
    df = df.copy()
    cols_to_drop = ["id"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  [DROP] Removed columns: {cols_to_drop}")
    return df


# ---------------------------------------------------------------------------
# MASTER PIPELINE — Orchestrates all preprocessing steps
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline in sequence.

    Pipeline Steps
    --------------
    1. Load raw data
    2. Drop irrelevant identifiers
    3. Impute missing values
    4. Ordinal encode ordered categorical features
    5. Binary encode two-class nominal features
    6. Label encode high-cardinality nominal features
    7. Validate and enforce target encoding
    8. Z-Score outlier removal on continuous features

    Returns
    -------
    pd.DataFrame
        A fully cleaned, encoded, and model-ready DataFrame.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1 — PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_data(path)

    print("\n[Step 1] Dropping irrelevant columns...")
    df = drop_irrelevant_columns(df)

    print("\n[Step 2] Handling missing values...")
    df = handle_missing_values(df)

    print("\n[Step 3] Ordinal encoding...")
    df = encode_ordinal_features(df)

    print("\n[Step 4] Binary encoding...")
    df = encode_binary_features(df)

    print("\n[Step 5] Label encoding (high cardinality)...")
    df = encode_high_cardinality_features(df)

    print("\n[Step 6] Target variable validation...")
    df = encode_target(df)

    print("\n[Step 7] Z-Score outlier removal...")
    df = remove_outliers_zscore(df)

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print(f"  Final shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Features    : {[c for c in df.columns if c != TARGET_COL]}")
    print("=" * 60)

    return df


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_clean = run_preprocessing_pipeline(DATA_PATH)

    # Persist cleaned dataset for downstream modules
    output_path = "df_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n  Cleaned dataset saved --> '{output_path}'")
    print("\n  Head of cleaned dataset:")
    print(df_clean.head(3).to_string())
