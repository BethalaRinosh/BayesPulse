"""Step 9: uncertainty, decision costs, dependence, and model comparison.

Run from repository root:
    python scripts/uncertainty_decision_analysis.py

This is exploratory analysis on a fixed held-out split, not clinical validation.
Bootstrap intervals are conditional on the observed test partition and do not
capture the full uncertainty from collecting a new dataset or retraining.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bayespulse.naive_bayes import HybridNaiveBayes

DATA_PATH = ROOT / "data" / "cleveland.csv"
OUTPUT_DIR = ROOT / "reports" / "evaluation"
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = CONTINUOUS + CATEGORICAL
RANDOM_STATE = 42
TEST_SIZE = 0.25
BOOTSTRAP_REPLICATES = 2000
FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0


def bootstrap_intervals(y_true: np.ndarray, probabilities: np.ndarray,
                        replicates: int = BOOTSTRAP_REPLICATES,
                        seed: int = RANDOM_STATE) -> dict[str, tuple[float, float, float]]:
    """Stratified bootstrap intervals conditional on held-out class counts."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= 0.5).astype(int)
    class_indices = [np.flatnonzero(y_true == label) for label in (0, 1)]
    if any(indices.size == 0 for indices in class_indices):
        raise ValueError("Bootstrap interval calculation requires both classes.")

    values: dict[str, list[float]] = {
        "Accuracy": [], "Recall (class 1)": [], "ROC-AUC": [], "Brier score": []
    }
    for _ in range(replicates):
        sampled = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True)
            for indices in class_indices
        ])
        truth = y_true[sampled]
        prob = probabilities[sampled]
        pred = predicted[sampled]
        values["Accuracy"].append(float(accuracy_score(truth, pred)))
        values["Recall (class 1)"].append(float(recall_score(truth, pred, zero_division=0)))
        values["ROC-AUC"].append(float(roc_auc_score(truth, prob)))
        values["Brier score"].append(float(brier_score_loss(truth, prob)))

    result = {}
    for name, samples in values.items():
        low, median, high = np.percentile(np.asarray(samples), [2.5, 50, 97.5])
        result[name] = (float(median), float(low), float(high))
    return result


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Compute bias-uncorrected Cramér's V for two categorical columns."""
    table = pd.crosstab(left, right).to_numpy(dtype=float)
    n = table.sum()
    if n <= 1 or min(table.shape) < 2:
        return float("nan")
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    valid = expected > 0
    chi_squared = float(np.sum(((table - expected) ** 2 / np.where(valid, expected, 1))[valid]))
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi_squared / denominator)) if denominator > 0 else float("nan")


def main() -> None:
    df = pd.read_csv(DATA_PATH, header=0, na_values=["?"])
    if df.shape != (303, 14):
        raise ValueError(f"Expected 303 rows and 14 columns; found {df.shape}.")
    if df["num"].isna().any() or not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("The target must be complete and contain only original values 0 through 4.")

    X = df[FEATURES].to_numpy(dtype=float)
    y = (df["num"] > 0).astype(int).to_numpy()
    row_indices = df.index.to_numpy()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, row_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = HybridNaiveBayes(
        continuous_features=tuple(range(len(CONTINUOUS))),
        categorical_features=tuple(range(len(CONTINUOUS), len(FEATURES))),
        alpha=1.0,
        variance_smoothing=1e-9,
    ).fit(X_train, y_train)
    nb_prob = model.predict_proba(X_test)[:, int(np.flatnonzero(model.classes_ == 1)[0])]

    # QDA is a comparator, not a replacement for the from-scratch hybrid model.
    # Imputation and one-hot encoding are fitted only on the training partition.
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", SimpleImputer(strategy="median"), CONTINUOUS),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ],
        remainder="drop",
    )
    qda = Pipeline([
        ("preprocess", preprocessor),
        ("model", QuadraticDiscriminantAnalysis(reg_param=0.1)),
    ])
    train_frame = pd.DataFrame(X_train, columns=FEATURES)
    test_frame = pd.DataFrame(X_test, columns=FEATURES)
    qda.fit(train_frame, y_train)
    qda_prob = qda.predict_proba(test_frame)[:, int(np.flatnonzero(qda.classes_ == 1)[0])]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "# Step 9: uncertainty, decision costs, feature dependence, and QDA comparison",
        "",
        "> Educational analysis only. No results here establish clinical validity or suitable patient-care thresholds.",
        "",
        "## Protocol",
        "",
        f"- Dataset: {len(df)} Cleveland-subset records; binary target is original num > 0.",
        f"- Fixed stratified split: {len(X_train)} train / {len(X_test)} test; random seed {RANDOM_STATE}.",
        "- Both models use the same split. QDA preprocessing is fitted on training rows only.",
        f"- Bootstrap: {BOOTSTRAP_REPLICATES} class-stratified resamples of the held-out predictions.",
        "- Intervals are conditional on this test partition and do not include training-sample or dataset-shift uncertainty.",
        "",
        "## Bootstrap intervals for hybrid Naive Bayes",
        "",
        "Intervals are percentile 95% intervals; the median bootstrap metric is also shown.",
        "",
        "| Metric | Test estimate | Bootstrap median | 95% interval |",
        "|---|---:|---:|---:|",
    ]
    point_predictions = (nb_prob >= 0.5).astype(int)
    point_metrics = {
        "Accuracy": float(accuracy_score(y_test, point_predictions)),
        "Recall (class 1)": float(recall_score(y_test, point_predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, nb_prob)),
        "Brier score": float(brier_score_loss(y_test, nb_prob)),
    }
    intervals = bootstrap_intervals(y_test, nb_prob)
    for name, point in point_metrics.items():
        median, low, high = intervals[name]
        report.append(f"| {name} | {point:.4f} | {median:.4f} | [{low:.4f}, {high:.4f}] |")

    report.extend([
        "",
        "## Decision-cost sensitivity",
        "",
        f"Illustrative costs: false negative = {FALSE_NEGATIVE_COST:g} units; false positive = {FALSE_POSITIVE_COST:g} unit. These are hypothetical weights, not clinical or economic estimates.",
        "",
        "The threshold sweep shows the trade-off on this held-out split. It does not select a clinically appropriate threshold.",
        "",
        "| Threshold for class 1 | False negatives | False positives | Recall | Total weighted error cost |",
        "|---:|---:|---:|---:|---:|",
    ])
    for threshold in np.arange(0.1, 1.0, 0.1):
        pred = (nb_prob >= threshold).astype(int)
        fn = int(np.sum((y_test == 1) & (pred == 0)))
        fp = int(np.sum((y_test == 0) & (pred == 1)))
        recall = float(recall_score(y_test, pred, zero_division=0))
        cost = FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp
        report.append(f"| {threshold:.1f} | {fn} | {fp} | {recall:.4f} | {cost:.1f} |")

    report.extend([
        "",
        "## Model comparison: hybrid Naive Bayes vs quadratic discriminant analysis",
        "",
        "QDA is a full-covariance Gaussian discriminant baseline. Because source predictors include discrete-coded variables and missing values, the comparator uses training-fitted median imputation for continuous variables, most-frequent imputation plus one-hot encoding for categorical variables, and covariance regularization (reg_param=0.1). It is a comparison model, not an equally structured implementation of the hybrid Naive Bayes assumptions.",
        "",
        "| Model | Accuracy at 0.5 | Recall (class 1) at 0.5 | ROC-AUC | Brier score |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, probabilities in (("Hybrid Naive Bayes (from scratch)", nb_prob), ("QDA (scikit-learn comparator)", qda_prob)):
        pred = (probabilities >= 0.5).astype(int)
        report.append(
            f"| {name} | {accuracy_score(y_test, pred):.4f} | "
            f"{recall_score(y_test, pred, zero_division=0):.4f} | "
            f"{roc_auc_score(y_test, probabilities):.4f} | "
            f"{brier_score_loss(y_test, probabilities):.4f} |"
        )

    report.extend([
        "",
        "## Feature-dependence diagnostics",
        "",
        "These are exploratory associations in the training partition, not causal relationships. Continuous-continuous pairs use absolute Spearman correlation; categorical-categorical pairs use Cramér's V; individual feature-target associations use mutual information. Mixed-type cross-pairs are not collapsed into one universal correlation coefficient.",
        "",
        "### Absolute Spearman correlation among continuous predictors",
        "",
        "| Feature pair | Absolute Spearman correlation |",
        "|---|---:|",
    ])
    train_df = pd.DataFrame(X_train, columns=FEATURES)
    continuous_filled = train_df[CONTINUOUS].apply(lambda col: col.fillna(col.median()))
    corr = continuous_filled.corr(method="spearman").abs()
    for i, left in enumerate(CONTINUOUS):
        for right in CONTINUOUS[i + 1:]:
            report.append(f"| {left} / {right} | {corr.loc[left, right]:.4f} |")

    report.extend([
        "",
        "### Cramér's V among categorical-coded predictors",
        "",
        "| Feature pair | Cramér's V |",
        "|---|---:|",
    ])
    categorical_filled = train_df[CATEGORICAL].copy()
    for column in CATEGORICAL:
        mode = categorical_filled[column].mode(dropna=True)
        if mode.empty:
            raise ValueError(f"Categorical feature {column} has no observed training values.")
        categorical_filled[column] = categorical_filled[column].fillna(mode.iloc[0])
    for i, left in enumerate(CATEGORICAL):
        for right in CATEGORICAL[i + 1:]:
            value = cramers_v(categorical_filled[left], categorical_filled[right])
            report.append(f"| {left} / {right} | {value:.4f} |" if np.isfinite(value) else f"| {left} / {right} | not defined |")

    report.extend([
        "",
        "### Mutual information with the binary target",
        "",
        "Scores are estimated on the training partition only. Higher values indicate more estimated statistical dependence, not causal importance.",
        "",
        "| Feature | Type used by estimator | Mutual information |",
        "|---|---|---:|",
    ])
    from sklearn.feature_selection import mutual_info_classif
    imputed_train = train_df.copy()
    for column in CONTINUOUS:
        imputed_train[column] = imputed_train[column].fillna(imputed_train[column].median())
    for column in CATEGORICAL:
        imputed_train[column] = imputed_train[column].fillna(imputed_train[column].mode().iloc[0])
    discrete_mask = np.array([False] * len(CONTINUOUS) + [True] * len(CATEGORICAL))
    mi = mutual_info_classif(
        imputed_train[FEATURES].to_numpy(dtype=float),
        y_train,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )
    for feature, score, discrete in zip(FEATURES, mi, discrete_mask):
        kind = "categorical" if discrete else "continuous"
        report.append(f"| {feature} | {kind} | {score:.4f} |")

    report.extend([
        "",
        "## Interpretation limits",
        "",
        "- The held-out set has only 76 rows. Bootstrap intervals can be wide and remain conditional on this one split.",
        "- The threshold table uses hypothetical error costs. It is a sensitivity analysis, not a recommendation for a medical decision threshold.",
        "- QDA's assumptions and performance differ from the hybrid model. The one-split comparison is not proof that either model generalizes better.",
        "- Mutual information and association coefficients are sensitive to sample size, coding, and missing-data handling.",
        "- The Naive Bayes conditional-independence assumption may be violated; dependence diagnostics describe this risk but do not correct it.",
        "- The dataset and all outputs are educational only, not clinical validation or a patient-care tool.",
        "",
    ])
    output_path = OUTPUT_DIR / "step9_uncertainty_decision_analysis.md"
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("BayesPulse Step 9 analysis")
    print("Educational research only; not a clinical assessment.")
    print(f"Training rows: {len(X_train)} | Held-out rows: {len(X_test)}")
    print(f"Bootstrap replicates: {BOOTSTRAP_REPLICATES}")
    print("\nHybrid Naive Bayes bootstrap intervals:")
    for name, (median, low, high) in intervals.items():
        print(f"  {name}: median={median:.4f}, 95% CI=[{low:.4f}, {high:.4f}]")
    print("\nModel comparison (accuracy / recall / ROC-AUC / Brier):")
    for name, probabilities in (("Hybrid Naive Bayes", nb_prob), ("QDA comparator", qda_prob)):
        pred = (probabilities >= 0.5).astype(int)
        print(
            f"  {name}: {accuracy_score(y_test, pred):.4f} / "
            f"{recall_score(y_test, pred, zero_division=0):.4f} / "
            f"{roc_auc_score(y_test, probabilities):.4f} / "
            f"{brier_score_loss(y_test, probabilities):.4f}"
        )
    print(f"Saved report: {output_path.relative_to(ROOT)}")
    print("No threshold is recommended for clinical use.")


if __name__ == "__main__":
    main()
