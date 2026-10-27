"""Explain a held-out prediction as prior odds plus per-feature log evidence.

Run from the repository root:
    python scripts/feature_contributions_demo.py

Contributions describe the fitted model's likelihood ratios, not causal effects
or medical explanations.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bayespulse.naive_bayes import HybridNaiveBayes

DATA_PATH = ROOT / "data" / "cleveland.csv"
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = CONTINUOUS + CATEGORICAL
RANDOM_STATE = 42
TEST_SIZE = 0.25


def main() -> None:
    df = pd.read_csv(DATA_PATH, header=0, na_values=["?"])
    if df.shape != (303, 14) or df["num"].isna().any():
        raise ValueError("Expected the complete 303-row Cleveland dataset with a target.")

    X = df[FEATURES].to_numpy(dtype=float)
    y = (df["num"] > 0).astype(int).to_numpy()
    X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
        X,
        y,
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = HybridNaiveBayes(
        continuous_features=tuple(range(len(CONTINUOUS))),
        categorical_features=tuple(range(len(CONTINUOUS), len(FEATURES))),
        alpha=1.0,
        variance_smoothing=1e-9,
    ).fit(X_train, y_train)

    sample = X_test[[0]]
    explanation = model.explain_log_odds(sample, positive_class=1)
    contributions = explanation["feature_log_likelihood_ratios"][0]
    row_index = int(index_test[0])
    prior_log_odds = float(explanation["prior_log_odds"])
    total_log_odds = float(explanation["total_log_odds"][0])
    probabilities = model.predict_proba(sample)[0]
    positive_position = int(np.flatnonzero(model.classes_ == 1)[0])

    print("BayesPulse: interpretable feature evidence")
    print("Educational demonstration only; not a clinical assessment.")
    print(f"Held-out dataset row index: {row_index}")
    print(f"Training rows: {len(X_train)} | Held-out rows: {len(X_test)}")
    print("Positive class: 1 (original dataset num > 0)")
    print(f"Prior log-odds for class 1: {prior_log_odds:+.6f}")
    print()
    print(f"{'Feature':<12} {'Observed value':>15} {'Log-likelihood ratio':>22}  Interpretation")
    print("-" * 83)

    for feature_index, (feature_name, contribution) in enumerate(zip(FEATURES, contributions)):
        value = sample[0, feature_index]
        value_text = "missing" if np.isnan(value) else f"{value:g}"
        if np.isnan(value):
            interpretation = "missing; no contribution"
        elif contribution > 1e-12:
            interpretation = "raises modelled class-1 odds"
        elif contribution < -1e-12:
            interpretation = "lowers modelled class-1 odds"
        else:
            interpretation = "neutral"
        print(
            f"{feature_name:<12} {value_text:>15} {contribution:>+22.6f}  {interpretation}"
        )

    reconstructed = prior_log_odds + float(np.sum(contributions))
    print()
    print(f"Reconstructed total log-odds: {reconstructed:+.6f}")
    print(f"Posterior P(class 1): {probabilities[positive_position]:.6f}")
    print(f"Predicted class: {model.classes_[int(np.argmax(probabilities))]}")
    print(f"Retrospective dataset label (0/1): {int(y_test[0])}")
    print()
    print("Positive contribution means the observed value is more likely under class 1")
    print("than class 0 according to this fitted model; negative means the reverse.")
    print("These are model-assumption-based associations, not causal or clinical explanations.")


if __name__ == "__main__":
    main()
