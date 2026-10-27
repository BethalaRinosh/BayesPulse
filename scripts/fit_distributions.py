"""Fit and diagnose per-class feature distributions for BayesPulse.

Run from the repository root:
    python scripts/fit_distributions.py

This is exploratory modelling, not the final classifier. It writes numeric
summaries, smoothed categorical likelihoods, histograms, and normal Q-Q plots.
"""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleveland.csv"
OUT_DIR = ROOT / "reports" / "distribution_fitting"
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "has_disease"
ALPHA = 0.05


def safe_shapiro(values: np.ndarray) -> tuple[float, float]:
    """Return Shapiro-Wilk W and p-value, using at most 5,000 observations.

    SciPy notes that p-values may be inaccurate above 5,000 observations.
    These class-specific groups are smaller, but the cap keeps this helper safe
    if the dataset is changed later.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.all(values == values[0]):
        return float("nan"), float("nan")
    if len(values) > 5000:
        rng = np.random.default_rng(42)
        values = rng.choice(values, size=5000, replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.shapiro(values)
    return float(result.statistic), float(result.pvalue)


def categorical_likelihoods(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate P(feature=value | class) with Laplace smoothing.

    For a category v and class c:
        P(v | c) = (count(v,c) + alpha) / (count(c) + alpha * K)
    where K is the number of observed categories for that feature.
    Missing values are excluded from the denominator for that feature.
    """
    rows = []
    for feature in CATEGORICAL:
        categories = sorted(df[feature].dropna().unique().tolist())
        k = len(categories)
        for class_value in (0, 1):
            class_values = df.loc[df[TARGET] == class_value, feature].dropna()
            denominator = len(class_values) + k  # alpha = 1
            for category in categories:
                count = int((class_values == category).sum())
                rows.append({
                    "feature": feature,
                    "class": class_value,
                    "category": category,
                    "count": count,
                    "observed_nonmissing_in_class": len(class_values),
                    "category_count_K": k,
                    "laplace_alpha": 1,
                    "smoothed_probability": (count + 1) / denominator,
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, header=0, na_values=["?"])
    expected = {"num", *CONTINUOUS, *CATEGORICAL}
    missing_columns = expected.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    if len(df) != 303:
        raise ValueError(f"Expected 303 records; found {len(df)}.")
    if df["num"].isna().any():
        raise ValueError("The target column 'num' contains missing labels.")
    if not set(df["num"].unique()).issubset({0, 1, 2, 3, 4}):
        raise ValueError("The target column 'num' contains unexpected values.")
    df[TARGET] = (df["num"] > 0).astype(int)

    # Continuous-feature fitting is performed separately within each class.
    fit_rows = []
    for feature in CONTINUOUS:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for class_value, ax_hist, ax_qq in [
            (0, axes[0, 0], axes[1, 0]),
            (1, axes[0, 1], axes[1, 1]),
        ]:
            values = df.loc[df[TARGET] == class_value, feature].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
            w_stat, p_value = safe_shapiro(values)

            # Fit a log-normal candidate only if all observations are strictly positive.
            lognormal_mu = float("nan")
            lognormal_sigma = float("nan")
            if np.all(values > 0):
                log_values = np.log(values)
                lognormal_mu = float(np.mean(log_values))
                lognormal_sigma = float(np.std(log_values, ddof=0))

            fit_rows.append({
                "feature": feature,
                "class": class_value,
                "n_nonmissing": n,
                "gaussian_mean": mean,
                "gaussian_sd_sample": sd,
                "shapiro_W": w_stat,
                "shapiro_p_value": p_value,
                "reject_normality_at_0_05": bool(p_value < ALPHA) if np.isfinite(p_value) else None,
                "lognormal_log_mean": lognormal_mu,
                "lognormal_log_sd": lognormal_sigma,
                "lognormal_candidate_valid": bool(np.all(values > 0)),
                "kde_candidate": "available" if n >= 2 and np.std(values) > 0 else "not available",
            })

            ax_hist.hist(values, bins="auto", density=True, alpha=0.55, edgecolor="black")
            x_grid = np.linspace(float(values.min()), float(values.max()), 300)
            if np.isfinite(sd) and sd > 0:
                ax_hist.plot(x_grid, stats.norm.pdf(x_grid, loc=mean, scale=sd), label="Gaussian PDF")
            # KDE is a visual non-parametric candidate, not an automatic winner.
            if n >= 2 and np.std(values) > 0:
                try:
                    kde = gaussian_kde(values)
                    ax_hist.plot(x_grid, kde(x_grid), label="KDE")
                except (ValueError, np.linalg.LinAlgError):
                    pass
            ax_hist.set_title(f"{feature}: class {class_value} (n={n})")
            ax_hist.set_xlabel(feature)
            ax_hist.set_ylabel("Density")
            ax_hist.legend()

            stats.probplot(values, dist="norm", plot=ax_qq)
            ax_qq.set_title(f"Normal Q-Q: {feature}, class {class_value}")

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{feature}_distribution_diagnostics.png", dpi=160)
        plt.close(fig)

    fit_table = pd.DataFrame(fit_rows)
    fit_table.to_csv(OUT_DIR / "continuous_fit_diagnostics.csv", index=False)
    cat_table = categorical_likelihoods(df)
    cat_table.to_csv(OUT_DIR / "categorical_laplace_probabilities.csv", index=False)

    print("\nContinuous distribution diagnostics:")
    print(fit_table[["feature", "class", "n_nonmissing", "gaussian_mean",
                    "gaussian_sd_sample", "shapiro_W", "shapiro_p_value",
                    "reject_normality_at_0_05", "lognormal_candidate_valid"]].to_string(index=False))
    print("\nCategorical likelihood table rows:", len(cat_table))
    print("Output directory:", OUT_DIR.relative_to(ROOT))
    print("\nInterpretation reminder: p < 0.05 is evidence against normality, not proof "
          "that a different distribution is best. Use Q-Q plots, histograms, support, "
          "and the model's validation results together.")


if __name__ == "__main__":
    main()
