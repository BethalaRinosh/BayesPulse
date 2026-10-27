"""Streamlit interface for the educational BayesPulse model.

Run with:
    streamlit run app.py

This interface is a classroom demonstration, not a clinical decision tool.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from bayespulse.naive_bayes import HybridNaiveBayes

DATA_PATH = ROOT / "data" / "cleveland.csv"
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = CONTINUOUS + CATEGORICAL

FEATURE_HELP = {
    "age": "Age in years.",
    "trestbps": "Resting blood pressure (mm Hg).",
    "chol": "Serum cholesterol (mg/dL).",
    "thalach": "Maximum heart rate recorded in the dataset.",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "sex": "Dataset code: 0 = female, 1 = male.",
    "cp": "Chest-pain type code: 1–4; the model uses the original numeric category code.",
    "fbs": "Fasting blood sugar > 120 mg/dL: 0 = no, 1 = yes.",
    "restecg": "Resting ECG code: 0 = normal, 1 = ST-T abnormality, 2 = probable/definite LV hypertrophy.",
    "exang": "Exercise-induced angina: 0 = no, 1 = yes.",
    "slope": "Peak exercise ST segment slope: 1 = upsloping, 2 = flat, 3 = downsloping.",
    "ca": "Number of major vessels (0–3); this feature has missing values in the dataset.",
    "thal": "Original thal code: 3 = normal, 6 = fixed defect, 7 = reversible defect.",
}
CATEGORICAL_LABELS = {
    "sex": {0: "0 · female", 1: "1 · male"},
    "cp": {1: "1 · typical angina", 2: "2 · atypical angina", 3: "3 · non-anginal pain", 4: "4 · asymptomatic"},
    "fbs": {0: "0 · no", 1: "1 · yes"},
    "restecg": {0: "0 · normal", 1: "1 · ST-T abnormality", 2: "2 · LV hypertrophy"},
    "exang": {0: "0 · no", 1: "1 · yes"},
    "slope": {1: "1 · upsloping", 2: "2 · flat", 3: "3 · downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "3 · normal", 6: "6 · fixed defect", 7: "7 · reversible defect"},
}


@st.cache_data
def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH, header=0, na_values=["?"])
    if frame.shape != (303, 14) or frame["num"].isna().any():
        raise ValueError("Expected the complete Cleveland dataset with 303 rows and target num.")
    return frame


@st.cache_resource
def train_demo_model() -> HybridNaiveBayes:
    frame = load_dataset()
    X = frame[FEATURES].to_numpy(dtype=float)
    y = (frame["num"] > 0).astype(int).to_numpy()
    return HybridNaiveBayes(
        continuous_features=tuple(range(len(CONTINUOUS))),
        categorical_features=tuple(range(len(CONTINUOUS), len(FEATURES))),
        alpha=1.0,
        variance_smoothing=1e-9,
    ).fit(X, y)


def main() -> None:
    st.set_page_config(page_title="BayesPulse", page_icon="🫀", layout="wide")
    st.title("BayesPulse")
    st.subheader("Probabilistic feature modelling")
    st.write(
        "Explore how a hybrid Naive Bayes model combines class priors and feature "
        "likelihoods to produce a posterior probability."
    )
    st.warning(
        "Educational demonstration only. This model is not clinically validated, "
        "must not be used to diagnose or assess a real person's health, and does "
        "not provide medical advice."
    )

    frame = load_dataset()
    model = train_demo_model()
    st.caption(
        f"Model trained on {len(frame)} records from the UCI Cleveland heart-disease "
        "subset. Target: original num > 0. This app fits on the full dataset "
        "for demonstration; displayed probabilities are not independent test estimates."
    )

    st.markdown("### Enter a hypothetical dataset record")
    st.write(
        "Inputs default to the dataset median/mode. The categorical variables are "
        "original dataset codes, not continuous measurements."
    )
    defaults = frame[FEATURES]
    input_values: dict[str, float] = {}
    left, right = st.columns(2)

    with left:
        st.markdown("#### Continuous features")
        for feature in CONTINUOUS:
            series = pd.to_numeric(defaults[feature], errors="coerce").dropna()
            low = float(series.min())
            high = float(series.max())
            default = float(series.median())
            step = 0.1 if feature == "oldpeak" else 1.0
            input_values[feature] = st.number_input(
                feature,
                min_value=low,
                max_value=high,
                value=default,
                step=step,
                help=FEATURE_HELP[feature],
            )

    with right:
        st.markdown("#### Categorical-coded features")
        for feature in CATEGORICAL:
            observed = sorted(
                int(value) for value in pd.to_numeric(defaults[feature], errors="coerce").dropna().unique()
            )
            mode_values = defaults[feature].mode(dropna=True)
            default_value = int(mode_values.iloc[0]) if not mode_values.empty else observed[0]
            selected = st.selectbox(
                feature,
                options=observed,
                index=observed.index(default_value),
                format_func=lambda value, name=feature: CATEGORICAL_LABELS.get(name, {}).get(value, str(value)),
                help=FEATURE_HELP[feature],
            )
            input_values[feature] = float(selected)

    missing_features = st.multiselect(
        "Optionally mark predictors as missing",
        options=FEATURES,
        default=[],
        help="Missing predictors are skipped by the model. Do not use this to represent unknown real-world medical values.",
    )
    for feature in missing_features:
        input_values[feature] = np.nan

    if st.button("Run Bayesian demonstration", type="primary", use_container_width=True):
        row = np.array([[input_values[name] for name in FEATURES]], dtype=float)
        posterior = model.predict_proba(row)[0]
        class_one_position = int(np.flatnonzero(model.classes_ == 1)[0])
        probability = float(posterior[class_one_position])
        explanation = model.explain_log_odds(row, positive_class=1)
        contributions = np.asarray(explanation["feature_log_likelihood_ratios"], dtype=float)[0]
        feature_indices = tuple(explanation["feature_indices"])
        evidence_names = [FEATURES[index] for index in feature_indices]
        prior_log_odds = float(explanation["prior_log_odds"])
        total_log_odds = float(np.asarray(explanation["total_log_odds"])[0])
        progression = model.predict_proba_progression(row)[0]

        st.markdown("### Model output")
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Modelled P(class 1)", f"{probability:.1%}")
        metric_right.metric("Modelled P(class 0)", f"{float(posterior[1 - class_one_position]):.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            "These are model posterior probabilities for the dataset's recorded binary label, "
            "not a calibrated estimate of future disease risk."
        )

        st.markdown("### Feature evidence")
        st.write(
            "Positive log-likelihood ratios support class 1 relative to class 0 under this "
            "fitted model; negative values support class 0. They are not causal effects."
        )
        evidence = pd.DataFrame({
            "Feature": evidence_names,
            "Observed value": [input_values[name] for name in evidence_names],
            "Log-likelihood ratio": contributions,
            "Model evidence": [
                "Missing, skipped" if not np.isfinite(input_values[name])
                else "Toward class 1" if value > 0
                else "Toward class 0" if value < 0
                else "Neutral"
                for name, value in zip(evidence_names, contributions)
            ],
        })
        evidence = evidence.sort_values("Log-likelihood ratio", key=lambda col: col.abs(), ascending=False)
        st.dataframe(evidence, use_container_width=True, hide_index=True)
        st.bar_chart(evidence.set_index("Feature")[["Log-likelihood ratio"]])

        st.markdown("### Bayesian updating trace")
        trace_names = ["Prior"] + evidence_names
        trace = pd.DataFrame({
            "Evidence included": trace_names,
            "P(class 0)": progression[:, 0],
            "P(class 1)": progression[:, 1],
        })
        st.line_chart(trace.set_index("Evidence included"))
        st.dataframe(trace, use_container_width=True, hide_index=True)

        st.caption(
            f"Prior log-odds: {prior_log_odds:.6f} · Total posterior log-odds: "
            f"{total_log_odds:.6f}. Contributions sum to the posterior log-odds."
        )

    with st.expander("About the dataset and model limitations"):
        st.markdown(
            "- Dataset: UCI Heart Disease, Cleveland subset (303 records). "
            "The binary label is 0 when the original num value is 0, otherwise 1.\n"
            "- Continuous features use Gaussian likelihoods; coded categorical features use "
            "Laplace-smoothed categorical likelihoods.\n"
            "- Naive Bayes assumes conditional independence between predictors given the class.\n"
            "- This small historical dataset may not represent other populations or current practice.\n"
            "- The app trains on the entire dataset for an interactive demonstration, so it must "
            "not be interpreted as a held-out prediction or external validation.\n"
            "- Do not enter real patient data. No patient data is needed or intended."
        )
        st.markdown("[UCI dataset documentation](https://archive.ics.uci.edu/dataset/45/heart+disease)")


if __name__ == "__main__":
    main()
