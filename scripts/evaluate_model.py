"""Held-out evaluation for the BayesPulse hybrid Naive Bayes model.

Run from the repository root:
    python scripts/evaluate_model.py

The split is stratified and fixed for reproducibility. The model is fitted only
on the training partition. Metrics and diagnostic plots describe this one split;
they are not clinical validation or a reliable estimate of deployment performance.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from bayespulse.naive_bayes import HybridNaiveBayes

DATA_PATH = ROOT / "data" / "cleveland.csv"
OUTPUT_DIR = ROOT / "reports" / "evaluation"
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = CONTINUOUS + CATEGORICAL
TARGET = "num"
RANDOM_STATE = 42
TEST_SIZE = 0.25
CALIBRATION_BINS = 5


def main() -> None:
    """Fit on train data, evaluate on held-out rows, and save a reproducible report."""
    df = pd.read_csv(DATA_PATH, header=0, na_values=["?"])
    if df.shape != (303, 14):
        raise ValueError(f"Expected 303 rows and 14 columns; found {df.shape}.")
    if df[TARGET].isna().any():
        raise ValueError("The target column 'num' contains missing labels.")

    X = df[FEATURES].to_numpy(dtype=float)
    y = (df[TARGET] > 0).astype(int).to_numpy()
    if not np.isfinite(y).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected a complete binary target containing classes 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = HybridNaiveBayes(
        continuous_features=tuple(range(len(CONTINUOUS))),
        categorical_features=tuple(range(len(CONTINUOUS), len(FEATURES))),
        alpha=1.0,
        variance_smoothing=1e-9,
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    class_one_position = int(np.flatnonzero(model.classes_ == 1)[0])
    positive_probability = probabilities[:, class_one_position]
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (class 1)": precision_score(y_test, predictions, zero_division=0),
        "Recall (class 1)": recall_score(y_test, predictions, zero_division=0),
        "F1 (class 1)": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, positive_probability),
        "Brier score": brier_score_loss(y_test, positive_probability),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        positive_probability,
        n_bins=CALIBRATION_BINS,
        strategy="quantile",
    )
    fpr, tpr, _ = roc_curve(y_test, positive_probability)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reliability diagram: a perfectly calibrated model would lie on the diagonal.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(mean_predicted, fraction_positive, marker="o", label="BayesPulse")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed fraction positive", title="Calibration curve (held-out set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=160)
    plt.close(fig)

    # ROC curve.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, label=f"BayesPulse (AUC = {metrics['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="No-discrimination reference")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate",
           ylabel="True positive rate", title="ROC curve (held-out set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Confusion matrix as a standalone figure.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Predicted 0", "Predicted 1"],
           yticklabels=["Actual 0", "Actual 1"],
           title="Confusion matrix (held-out set)")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    report_lines = [
        "# Step 6: held-out evaluation results",
        "",
        "> Educational research only. These results are not clinical validation,",
        "> medical advice, or evidence of suitability for patient care.",
        "",
        "## Protocol",
        "",
        f"- Dataset rows: {len(df)}",
        f"- Training rows: {len(X_train)}",
        f"- Held-out rows: {len(X_test)}",
        f"- Split: stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} train/test",
        f"- Random seed: {RANDOM_STATE}",
        "- Positive class: original num > 0",
        f"- Predictors: {len(FEATURES)} ({len(CONTINUOUS)} Gaussian, {len(CATEGORICAL)} categorical)",
        "- Missing predictor values are handled by the model's feature-wise skipping.",
        "- Model parameters are fitted on the training partition only.",
        "",
        "## Held-out metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    report_lines.extend(f"| {name} | {value:.4f} |" for name, value in metrics.items())
    report_lines.extend([
        "",
        "## Confusion matrix",
        "",
        "Rows are actual classes; columns are predicted classes. Class order is 0, 1.",
        "",
        "| Actual / Predicted | 0 | 1 |",
        "|---|---:|---:|",
        f"| 0 | {matrix[0, 0]} | {matrix[0, 1]} |",
        f"| 1 | {matrix[1, 0]} | {matrix[1, 1]} |",
        "",
        "## Calibration bins",
        "",
        "| Mean predicted probability | Observed fraction positive |",
        "|---:|---:|",
    ])
    report_lines.extend(
        f"| {predicted:.4f} | {observed:.4f} |"
        for predicted, observed in zip(mean_predicted, fraction_positive)
    )
    report_lines.extend([
        "",
        "## Plots",
        "",
        "- calibration_curve.png: compares predicted probability with observed frequency.",
        "- roc_curve.png: threshold-independent ranking diagnostic.",
        "- confusion_matrix.png: counts at the default maximum-posterior class decision.",
        "",
        "## Interpretation limits",
        "",
        "- This is one 76-row held-out split, so metrics can vary substantially with the split.",
        "- A calibration curve with five quantile bins is descriptive and noisy at this sample size.",
        "- ROC-AUC measures ranking, not probability calibration or clinical utility.",
        "- Naive Bayes assumes conditional independence of features; that assumption is unlikely to hold perfectly here.",
        "- No threshold was selected to optimize clinical decisions. Precision/recall depend on the default maximum-posterior class decision.",
        "- This dataset and evaluation are for an educational project only, not clinical deployment.",
        "",
    ])
    report_path = OUTPUT_DIR / "step6_evaluation_results.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("BayesPulse held-out evaluation")
    print(f"Training rows: {len(X_train)}")
    print(f"Held-out rows: {len(X_test)}")
    print("Features:", ", ".join(FEATURES))
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("Confusion matrix (rows=actual [0,1], columns=predicted [0,1]):")
    print(matrix)
    print("Calibration bins (mean predicted probability -> observed fraction positive):")
    for predicted, observed in zip(mean_predicted, fraction_positive):
        print(f"  {predicted:.4f} -> {observed:.4f}")
    print(f"Saved report: {report_path.relative_to(ROOT)}")
    print(f"Saved plots: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
