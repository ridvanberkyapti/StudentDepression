"""
=============================================================================
STUDENT DEPRESSION RISK ANALYSIS — PHASE 3: STREAMLIT DASHBOARD
=============================================================================
Author  : [Your Name]
Course  : Probability and Statistics — Term Project
Model   : Naive Bayes Classifier (GaussianNB) + SHAP Explainability
=============================================================================

ARCHITECTURE
------------
  phase1_preprocessing  →  run_preprocessing_pipeline()  →  clean DataFrame
  app.py                →  GaussianNB training, interactive UI, SHAP analysis

USAGE
-----
  streamlit run app.py

NOTE: Student_Depression_Dataset.csv and phase1_preprocessing.py must reside
      in the same working directory as this file.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Student Depression Risk Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS — mirrors phase1_preprocessing.py mappings exactly
# ---------------------------------------------------------------------------

SLEEP_MAP = {
    "Less than 5 hours": 0,
    "5-6 hours":         1,
    "7-8 hours":         2,
    "More than 8 hours": 3,
    "Others":            1,
}

DIET_MAP = {
    "Unhealthy": 0,
    "Moderate":  1,
    "Healthy":   2,
    "Others":    1,
}

SUICIDAL_MAP = {"No": 0, "Yes": 1}
FAMILY_MAP   = {"No": 0, "Yes": 1}

# Feature columns expected by the model (same order as training DataFrame)
# High-cardinality cols (City, Profession, Degree, Gender) are excluded from
# the sidebar for usability — they are set to their dataset median/mode below.
FEATURE_COLS = [
    "Gender", "Age", "City", "Profession", "Academic Pressure",
    "Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction",
    "Sleep Duration", "Dietary Habits",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours", "Financial Stress",
    "Family History of Mental Illness",
]

TARGET_COL = "Depression"


# ---------------------------------------------------------------------------
# MODEL TRAINING  —  cached so it runs only once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training Naive Bayes model… this may take a moment.")
def train_model(csv_path: str = "Student_Depression_Dataset.csv"):
    """
    Execute Phase 1 pipeline, split data, train GaussianNB, return artefacts.

    Returns
    -------
    dict with keys:
        model        : fitted GaussianNB
        X_test       : test feature matrix (DataFrame)
        y_test       : test labels (Series)
        accuracy     : float
        report       : str  (classification_report)
        feature_cols : list of feature column names
        medians      : dict of median values (for sidebar defaults)
    """
    # Import here so Streamlit's module watcher doesn't re-run the heavy import
    from phase1_preprocessing import run_preprocessing_pipeline

    df = run_preprocessing_pipeline(csv_path)

    # Guard: fill any residual NaNs that slip through ordinal/label mapping
    # (e.g. unseen category values mapped to NaN by phase1 .map() calls)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == object:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 70 / 30 stratified split — preserves class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=["No Depression", "Depression"])

    # Median values used as defaults / fill-ins for hidden sidebar fields
    medians = X_train.median().to_dict()

    return {
        "model":        model,
        "X_train":      X_train,
        "X_test":       X_test,
        "y_test":       y_test,
        "accuracy":     accuracy,
        "report":       report,
        "feature_cols": list(X.columns),
        "medians":      medians,
    }


# ---------------------------------------------------------------------------
# SIDEBAR — Interactive student profile builder
# ---------------------------------------------------------------------------

def build_sidebar(feature_cols: list, medians: dict) -> pd.DataFrame:
    """
    Render sidebar inputs and return a single-row DataFrame ready for
    model inference.  All encoding mirrors phase1_preprocessing.py exactly.
    """
    st.sidebar.header("🎓 Student Profile")
    st.sidebar.markdown("Adjust the sliders below to define a student profile.")

    # ---- Continuous features -----------------------------------------------
    age = st.sidebar.slider(
        "Age", min_value=16, max_value=35, value=21, step=1
    )
    cgpa = st.sidebar.slider(
        "CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1
    )
    work_study_hours = st.sidebar.slider(
        "Work / Study Hours per Day", min_value=0, max_value=16, value=8, step=1
    )

    # ---- Likert-scale features (1-5) ----------------------------------------
    st.sidebar.markdown("---")
    academic_pressure = st.sidebar.slider(
        "Academic Pressure (1 = Low, 5 = High)",
        min_value=1, max_value=5, value=3, step=1
    )
    financial_stress = st.sidebar.slider(
        "Financial Stress (1 = Low, 5 = High)",
        min_value=1, max_value=5, value=3, step=1
    )
    study_satisfaction = st.sidebar.slider(
        "Study Satisfaction (1 = Low, 5 = High)",
        min_value=1, max_value=5, value=3, step=1
    )

    # ---- Ordinal categorical ------------------------------------------------
    st.sidebar.markdown("---")
    sleep_label = st.sidebar.selectbox(
        "Sleep Duration",
        options=list(SLEEP_MAP.keys()),
        index=2,          # default: "7-8 hours"
    )
    diet_label = st.sidebar.selectbox(
        "Dietary Habits",
        options=list(DIET_MAP.keys()),
        index=1,          # default: "Moderate"
    )

    # ---- Binary categorical --------------------------------------------------
    st.sidebar.markdown("---")
    suicidal_label = st.sidebar.selectbox(
        "History of Suicidal Thoughts",
        options=["No", "Yes"],
    )
    family_label = st.sidebar.selectbox(
        "Family History of Mental Illness",
        options=["No", "Yes"],
    )

    # ---- Encode inputs using Phase 1 mappings --------------------------------
    encoded = {
        "Age":                                    age,
        "CGPA":                                   cgpa,
        "Work/Study Hours":                       work_study_hours,
        "Academic Pressure":                      academic_pressure,
        "Financial Stress":                       financial_stress,
        "Study Satisfaction":                     study_satisfaction,
        "Sleep Duration":                         SLEEP_MAP[sleep_label],
        "Dietary Habits":                         DIET_MAP[diet_label],
        "Have you ever had suicidal thoughts ?":  SUICIDAL_MAP[suicidal_label],
        "Family History of Mental Illness":       FAMILY_MAP[family_label],
    }

    # Fill hidden high-cardinality / irrelevant features with training medians
    row = {}
    for col in feature_cols:
        if col in encoded:
            row[col] = encoded[col]
        else:
            row[col] = medians.get(col, 0)

    return pd.DataFrame([row])[feature_cols]   # enforce column order


# ---------------------------------------------------------------------------
# SHAP ANALYSIS — Explainable AI for the individual prediction
# ---------------------------------------------------------------------------

def render_shap(model: GaussianNB, X_train: pd.DataFrame,
                input_df: pd.DataFrame, feature_cols: list) -> None:
    """
    Compute SHAP values for the given input vector using a KernelExplainer
    (model-agnostic, works with any sklearn estimator) and render a
    Waterfall plot showing per-feature contribution to P(Depression=1).
    """
    st.subheader("🔍 Explainable AI — SHAP Feature Contribution")
    st.markdown(
        "The chart below shows **which features push the Depression risk "
        "up (red) or down (blue)** for the current student profile. "
        "SHAP values are additive: their sum equals the log-odds shift "
        "from the model's baseline probability."
    )

    with st.spinner("Computing SHAP values… (first run may take ~20 s)"):

        # Model wrapper: KernelExplainer needs a callable → probabilities
        def predict_proba_depression(X: np.ndarray) -> np.ndarray:
            """Return P(Depression=1) for an array of input rows."""
            df_tmp = pd.DataFrame(X, columns=feature_cols)
            return model.predict_proba(df_tmp)[:, 1]

        # Use a small background sample for speed (100 rows from training set)
        background = shap.sample(X_train, min(100, len(X_train)), random_state=42)

        explainer   = shap.KernelExplainer(predict_proba_depression, background)
        shap_values = explainer.shap_values(input_df, nsamples=200)

        # Build SHAP Explanation object for Waterfall plot
        base_value = explainer.expected_value
        sv_row     = shap_values[0] if shap_values.ndim == 2 else shap_values

        explanation = shap.Explanation(
            values      = sv_row,
            base_values = base_value,
            data        = input_df.values[0],
            feature_names = feature_cols,
        )

    # ---- Waterfall plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    shap.waterfall_plot(explanation, max_display=12, show=False)

    # Style tweaks for readability inside Streamlit
    for text_obj in fig.findobj(plt.Text):
        text_obj.set_color("#E6EDF3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262D")
    ax.tick_params(colors="#8B949E")
    ax.xaxis.label.set_color("#E6EDF3")

    plt.title(
        "SHAP Waterfall — Feature Contributions to P(Depression | Profile)",
        color="#E6EDF3", fontsize=13, pad=12,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---- Bar plot (magnitude ranking) ----------------------------------------
    st.markdown("**Feature Importance — Absolute SHAP Magnitudes**")
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.patch.set_facecolor("#0D1117")
    ax2.set_facecolor("#161B22")

    shap.bar_plot(explanation, max_display=12, show=False)

    for text_obj in fig2.findobj(plt.Text):
        text_obj.set_color("#E6EDF3")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#21262D")
    ax2.tick_params(colors="#8B949E")

    plt.title(
        "SHAP Bar Plot — Absolute Feature Impact",
        color="#E6EDF3", fontsize=13, pad=12,
    )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------------

def main() -> None:

    # ---- Header ---------------------------------------------------------------
    st.title("🧠 Student Depression Risk Analyzer")
    st.markdown(
        "**Probabilistic Analysis of Student Lifestyle & Depression Risk** — "
        "Probability & Statistics Term Project | Naive Bayes Classifier"
    )
    st.divider()

    # ---- Load / train model ---------------------------------------------------
    try:
        artefacts = train_model("Student_Depression_Dataset.csv")
    except FileNotFoundError:
        st.error(
            "❌ **Dataset not found.** "
            "Place `Student_Depression_Dataset.csv` and `phase1_preprocessing.py` "
            "in the same directory as `app.py`, then restart Streamlit."
        )
        st.stop()
    except Exception as exc:
        st.error(f"❌ **Pipeline error:** {exc}")
        st.stop()

    model        = artefacts["model"]
    X_train      = artefacts["X_train"]
    X_test       = artefacts["X_test"]
    y_test       = artefacts["y_test"]
    accuracy     = artefacts["accuracy"]
    report       = artefacts["report"]
    feature_cols = artefacts["feature_cols"]
    medians      = artefacts["medians"]

    # ---- Sidebar inputs -------------------------------------------------------
    input_df = build_sidebar(feature_cols, medians)

    # ---- Model prediction for current profile ---------------------------------
    prob_vector  = model.predict_proba(input_df)[0]   # [P(0), P(1)]
    risk_pct     = prob_vector[1] * 100               # P(Depression=1) in %
    prediction   = model.predict(input_df)[0]

    # ---- Top KPI row ----------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric(
        label  = "📊 Model Accuracy (Test Set)",
        value  = f"{accuracy * 100:.2f}%",
        help   = "GaussianNB accuracy on the 30% held-out test split.",
    )

    risk_label = "🔴 HIGH RISK" if prediction == 1 else "🟢 LOW RISK"
    col2.metric(
        label = "🎯 Depression Risk — Current Profile",
        value = f"{risk_pct:.1f}%",
        delta = risk_label,
        delta_color = "inverse",
        help  = "P(Depression=1 | entered features) computed by GaussianNB.",
    )

    col3.metric(
        label = "📐 Train / Test Split",
        value = "70% / 30%",
        help  = "Stratified split preserving class balance.",
    )

    st.divider()

    # ---- Conditional probability formula display ------------------------------
    st.subheader("📐 Conditional Probability Output")

    st.latex(
        r"P(\text{Depression} \mid \mathbf{x}) = "
        r"\frac{P(\mathbf{x} \mid \text{Depression}) \cdot P(\text{Depression})}"
        r"{P(\mathbf{x})} \quad \longrightarrow \quad"
        rf"\hat{{P}} = {risk_pct:.2f}\%"
    )

    # Risk gauge — colour-coded progress bar
    bar_color = (
        "#F85149" if risk_pct >= 70
        else "#D29922" if risk_pct >= 40
        else "#3FB950"
    )
    st.markdown(
        f"""
        <div style="margin:8px 0 16px 0;">
          <div style="background:#21262D; border-radius:8px; height:22px; overflow:hidden;">
            <div style="width:{risk_pct:.1f}%; background:{bar_color};
                        height:100%; border-radius:8px;
                        transition: width 0.4s ease;">
            </div>
          </div>
          <p style="color:#8B949E; font-size:0.85em; margin:4px 0;">
            Risk level: <b style="color:{bar_color};">{risk_pct:.1f}%</b>
            — {'High Risk: professional support is recommended.'
               if risk_pct >= 70 else
               'Moderate Risk: monitoring advised.'
               if risk_pct >= 40 else
               'Low Risk: no immediate concern.'}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Classification report ------------------------------------------------
    with st.expander("📋 Full Classification Report (Test Set)", expanded=False):
        st.code(report, language="text")

    st.divider()

    # ---- SHAP Explainability --------------------------------------------------
    render_shap(model, X_train, input_df, feature_cols)

    st.divider()

    # ---- Dataset statistics footer -------------------------------------------
    st.subheader("📊 Dataset & Model Summary")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    n_train = len(X_train)
    n_test  = len(X_test)
    dep_rate = y_test.mean() * 100

    info_col1.metric("Training Samples", f"{n_train:,}")
    info_col2.metric("Test Samples",     f"{n_test:,}")
    info_col3.metric("Features Used",    len(feature_cols))
    info_col4.metric("Depression Rate (Test)", f"{dep_rate:.1f}%")

    st.caption(
        "Model: GaussianNB | Encoding: Ordinal + Binary + LabelEncoder (Phase 1) | "
        "Outlier removal: Z-Score ±3σ | SHAP: KernelExplainer (nsamples=200)"
    )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
