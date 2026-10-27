"""Exploratory analysis for the UCI Cleveland Heart Disease dataset.

Run from the repository root:
    python scripts/explore_dataset.py

Descriptive analysis only; this script does not train a predictive model.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleveland.csv"
OUT_DIR = ROOT / "reports" / "eda"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def main() -> None:
    """Load the data, audit it, and generate descriptive summaries and plots."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, names=COLUMNS, na_values=["?"])

    if df.shape != (303, 14):
        raise ValueError(f"Expected 303 rows and 14 columns; found {df.shape}.")
    if df["num"].isna().any():
        raise ValueError("The target column 'num' must not contain missing values.")

    df["has_disease"] = (df["num"] > 0).astype(int)

    print("\nDataset shape:", df.shape)
    print("\nMissing values by column:\n", df.isna().sum().to_string())
    print("\nOriginal target counts:\n", df["num"].value_counts().sort_index().to_string())
    print("\nBinary target counts:\n", df["has_disease"].value_counts().sort_index().to_string())

    df[CONTINUOUS].describe().T.to_csv(OUT_DIR / "continuous_summary.csv")
    df.isna().sum().rename("missing_count").to_csv(OUT_DIR / "missingness.csv")
    df["num"].value_counts().sort_index().rename_axis("num").rename("count").to_csv(
        OUT_DIR / "target_counts_original.csv"
    )
    df["has_disease"].value_counts().sort_index().rename_axis("has_disease").rename(
        "count"
    ).to_csv(OUT_DIR / "target_counts_binary.csv")

    # Histograms are descriptive; formal distribution diagnostics come in Step 3.
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, feature in zip(axes.flat, CONTINUOUS):
        ax.hist(df[feature].dropna(), bins="auto", edgecolor="black")
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
    axes.flat[-1].axis("off")
    fig.suptitle("Cleveland dataset: continuous feature distributions")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "continuous_histograms.png", dpi=160)
    plt.close(fig)

    # Keep missing category values visible rather than treating them as zero.
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    target_counts = df["num"].value_counts().sort_index()
    axes.flat[0].bar(target_counts.index.astype(str), target_counts.values)
    axes.flat[0].set_title("Original target num")
    axes.flat[0].set_xlabel("num")
    axes.flat[0].set_ylabel("Count")
    for ax, feature in zip(list(axes.flat)[1:], CATEGORICAL):
        counts = df[feature].value_counts(dropna=False).sort_index()
        labels = ["Missing" if pd.isna(v) else str(v) for v in counts.index]
        ax.bar(labels, counts.values, edgecolor="black")
        ax.set_title(feature)
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle("Cleveland dataset: target and discrete feature counts")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "categorical_counts.png", dpi=160)
    plt.close(fig)

    print(f"\nSaved summaries and plots to: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
