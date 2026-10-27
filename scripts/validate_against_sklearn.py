"""Compare the from-scratch Gaussian path with scikit-learn GaussianNB.

Uses only continuous features so both models make the same distributional
assumptions. A stratified held-out split is used, and all parameters are fitted
on the training partition only.

Run from repository root:
    python scripts/validate_against_sklearn.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from bayespulse.naive_bayes import HybridNaiveBayes

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleveland.csv"
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET = "num"
RANDOM_STATE = 42
TEST_SIZE = 0.25
TOLERANCE = 1e-8


def main() -> None:
    df = pd.read_csv(DATA_PATH, header=0, na_values=["?"])
    if len(df) != 303:
        raise ValueError(f"Expected 303 rows; found {len(df)}.")
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")
    y = (df[TARGET] > 0).astype(int).to_numpy()
    X = df[CONTINUOUS].to_numpy(dtype=float)

    # Continuous columns have no missing values in this dataset. Keep the
    # guard here so a changed source file cannot silently alter the comparison.
    if not np.isfinite(X).all():
        raise ValueError("Continuous comparison features must be complete and finite.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected both binary classes 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # var_smoothing=0.0 lets the reference use the raw population variances,
    # matching the from-scratch implementation without a smoothing floor.
    reference = GaussianNB(var_smoothing=0.0)
    reference.fit(X_train, y_train)

    scratch = HybridNaiveBayes(
        continuous_features=tuple(range(len(CONTINUOUS))),
        categorical_features=(),
        variance_smoothing=0.0,
    )
    scratch.fit(X_train, y_train)

    scratch_proba = scratch.predict_proba(X_test)
    reference_proba = reference.predict_proba(X_test)
    scratch_pred = scratch.predict(X_test)
    reference_pred = reference.predict(X_test)

    max_probability_difference = float(np.max(np.abs(scratch_proba - reference_proba)))
    predictions_match = bool(np.array_equal(scratch_pred, reference_pred))
    priors_match = bool(np.allclose(
        np.exp(scratch.class_log_prior_), reference.class_prior_, atol=TOLERANCE, rtol=0
    ))

    # Verify the fitted class-conditional Gaussian means and variances too.
    parameter_differences = []
    for class_position, class_value in enumerate(scratch.classes_):
        means, variances = scratch.gaussian_params_[class_position]
        ref_position = int(np.flatnonzero(reference.classes_ == class_value)[0])
        parameter_differences.extend([
            float(np.max(np.abs(means[:len(CONTINUOUS)] - reference.theta_[ref_position]))),
            float(np.max(np.abs(variances[:len(CONTINUOUS)] - reference.var_[ref_position]))),
        ])
    max_parameter_difference = max(parameter_differences, default=0.0)

    print("BayesPulse vs scikit-learn GaussianNB")
    print(f"Training rows: {len(X_train)}")
    print(f"Held-out rows: {len(X_test)}")
    print(f"Features: {', '.join(CONTINUOUS)}")
    print(f"Class priors match: {priors_match}")
    print(f"Maximum fitted mean/variance difference: {max_parameter_difference:.3e}")
    print(f"Maximum held-out probability difference: {max_probability_difference:.3e}")
    print(f"Held-out class predictions match: {predictions_match}")

    if not priors_match:
        raise AssertionError("Class priors differ from GaussianNB.")
    if max_parameter_difference > TOLERANCE:
        raise AssertionError("Fitted Gaussian parameters differ from GaussianNB.")
    if max_probability_difference > TOLERANCE:
        raise AssertionError("Held-out probabilities differ from GaussianNB.")
    if not predictions_match:
        raise AssertionError("Held-out class predictions differ from GaussianNB.")

    print("PASS: continuous-only implementation matches GaussianNB within tolerance.")
    print("Note: this validates implementation equivalence under Gaussian assumptions; "
          "it is not a claim of clinical validity or generalization performance.")


if __name__ == "__main__":
    main()
