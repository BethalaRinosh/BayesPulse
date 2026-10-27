"""Demonstrate feature-by-feature Bayesian updating with BayesPulse.

Run from the repository root:
    python scripts/bayesian_updating_demo.py

This is an educational trace for one deterministic held-out row. The row's label
is printed only for retrospective context; it is never used in model fitting.
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
    progression = model.predict_proba_progression(sample)[0]
    row_index = int(index_test[0])

    print("BayesPulse: feature-by-feature Bayesian updating")
    print("Educational demonstration only; not a clinical assessment.")
    print(f"Dataset row index (held out): {row_index}")
    print(f"Training rows: {len(X_train)} | Held-out rows: {len(X_test)}")
    print(f"Evidence order: {', '.join(FEATURES)}")
    print()
    print(f"{'Evidence included':<20} {'P(class 0)':>12} {'P(class 1)':>12} {'Predicted':>10}")
    print("-" * 58)

    for step, posterior in enumerate(progression):
        if step == 0:
            label = "Prior (none)"
        else:
            label = FEATURES[step - 1]
        predicted_class = model.classes_[int(np.argmax(posterior))]
        print(f"{label:<20} {posterior[0]:>12.6f} {posterior[1]:>12.6f} {predicted_class:>10}")

    final_posterior = progression[-1]
    print()
    print(f"Final posterior: class 0 = {final_posterior[0]:.6f}, "
          f"class 1 = {final_posterior[1]:.6f}")
    print(f"Retrospective dataset label (0/1): {int(y_test[0])}")
    print("Reminder: this single-row trace illustrates how the model updates; it is "
          "not evidence of clinical validity.")


if __name__ == "__main__":
    main()
