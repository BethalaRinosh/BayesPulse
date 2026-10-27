# Step 2: Dataset acquisition and exploratory analysis

## Dataset provenance

- **Original source:** UCI Machine Learning Repository, Heart Disease (Cleveland subset): https://archive.ics.uci.edu/dataset/45/heart+disease
- **Dataset citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- **License:** CC BY 4.0, as stated by UCI.
- **File:** `data/cleveland.csv`. The original target column `num` is retained with values 0–4. Missing values are represented by `?`.
- **Retrieval note:** The UCI-hosted file endpoint was not directly retrievable in this session. The values were retrieved from the Data Professor public mirror (https://github.com/dataprofessor/data/blob/master/heart-disease-cleveland.csv), which identifies UCI as the source, then retained with UCI attribution. The mirror's separate license was not relied upon; the original UCI dataset's stated CC BY 4.0 license is the provenance basis. The official UCI page remains the canonical source.

## Dataset structure

- **Rows:** 303
- **Columns:** 14 total, comprising 13 predictors and target `num`
- **Target values observed:** 0, 1, 2, 3, 4
- **Binary mapping planned:** `num == 0` maps to 0; `num > 0` maps to 1. The original `num` column is preserved.

## Target distribution

| Original `num` | Count | Binary interpretation |
|---:|---:|---|
| 0 | 164 | No recorded heart disease |
| 1 | 55 | Recorded heart disease |
| 2 | 36 | Recorded heart disease |
| 3 | 35 | Recorded heart disease |
| 4 | 13 | Recorded heart disease |
| **Total** | **303** | |

After binary mapping: class 0 = 164 records (54.1%); class 1 = 139 records (45.9%). The binary classes are reasonably balanced for a small teaching dataset, but the original five target levels are unevenly represented. The project models the binary task, not disease severity.

## Missingness audit

| Feature | Missing values | Share of 303 rows |
|---|---:|---:|
| `ca` | 4 | 1.32% |
| `thal` | 2 | 0.66% |
| All other predictors and `num` | 0 | 0% |
| **Total missing cells** | **6** | **0.14% of all 4,242 cells** |

Only 6 rows contain a missing predictor value, so 297 rows are complete. The missing values are encoded as `?` in the stored file and must not be interpreted as zero.

**Decision for modelling:** preserve the original records. When training/testing is introduced, any imputation values must be learned from the training partition only. We will consider a simple training-only mode imputation for the discrete `ca` and `thal` fields, and keep missing-feature handling explicit. No test-set information may be used to fit imputation values.

## Continuous-feature summary

These statistics were calculated from the 303 available records; standard deviation is the sample standard deviation.

| Feature | Minimum | Maximum | Mean | Std. dev. | Median |
|---|---:|---:|---:|---:|---:|
| age | 29 | 77 | 54.44 | 9.04 | 56 |
| trestbps | 94 | 200 | 131.69 | 17.60 | 130 |
| chol | 126 | 564 | 246.69 | 51.78 | 241 |
| thalach | 71 | 202 | 149.61 | 22.88 | 153 |
| oldpeak | 0.0 | 6.2 | 1.04 | 1.16 | 0.8 |

Observed ranges make it important to inspect distribution shapes and possible extreme values rather than assume every feature is Gaussian. `oldpeak` is non-negative and may be right-skewed; formal per-class distribution diagnostics belong in Step 3.

## Feature-type plan

| Type | Features | Initial modelling treatment |
|---|---|---|
| Continuous / measured numeric | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | Compare Gaussian fit per class; consider log-normal or KDE if diagnostics justify it |
| Discrete / categorical | `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal` | Bernoulli for genuinely binary fields; categorical likelihoods with Laplace smoothing for multi-category fields |
| Target | `num` | Preserve original 0–4 values; derive binary target using `num > 0` for the model task |

Several discrete fields are stored as numbers, but their codes are category labels, not measurements whose numeric distances should automatically be treated as meaningful.

## Leakage and limitations

- Keep `num` separate from predictor columns.
- No patient names or social-security identifiers are present in this processed subset.
- Imputation, distribution fitting, feature selection, and threshold selection must use training data only.
- This is a small historical dataset of 303 records and is not representative of all current clinical populations.
- The target reflects recorded diagnosis in these records. It is not an individual's future clinical risk.

## Reproducible exploratory analysis

Run from the repository root:

```bash
python scripts/explore_dataset.py
```

The script prints the data-quality and class-count summaries, and writes CSV summaries and exploratory plots under `reports/eda/`. The script has been added but was not executed in GitHub's remote environment during this step; its generated plots are created when run locally.


---

# Step 3: Per-class distribution fitting and diagnostics

## Purpose

This step examines candidate likelihood distributions for the future from-scratch Naive Bayes model. It is exploratory: it does not train or evaluate the final classifier and does not automatically choose a distribution based only on a statistical test.

## Implementation added

- `scripts/fit_distributions.py` loads `data/cleveland.csv` and creates separate continuous-feature diagnostics for binary classes 0 and 1.
- For each continuous feature and class, it records sample size, Gaussian mean and sample standard deviation, Shapiro-Wilk W statistic and p-value, and whether a log-normal candidate is mathematically valid (all observed values must be strictly positive).
- It writes a histogram with fitted Gaussian density and a KDE curve when fitting is numerically possible, alongside a normal Q-Q plot.
- It writes categorical likelihoods using Laplace smoothing:
  `P(feature=v | class=c) = (count(v,c) + 1) / (nonmissing_count(c) + K)`,
  where K is the number of observed categories for that feature. Missing feature values are excluded from that feature's count denominator rather than treated as an extra category.
- Outputs are written to `reports/distribution_fitting/`.

## Run

From the repository root:

```bash
pip install numpy pandas scipy matplotlib
python scripts/fit_distributions.py
```

Expected outputs:

- `continuous_fit_diagnostics.csv`
- `categorical_laplace_probabilities.csv`
- One `*_distribution_diagnostics.png` image per continuous feature

## How to interpret the tests

- The Shapiro-Wilk null hypothesis is that a sample was drawn from a normal distribution. A small p-value (for example, below 0.05) is evidence against normality, not proof that a particular alternative is better. [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
- Q-Q plots and histograms must be considered alongside the p-value. Small deviations may be statistically detectable in larger samples, while tests may have limited power in smaller class-specific samples.
- A log-normal candidate is not valid for features containing zero or negative observations. In this dataset, `oldpeak` includes zero, so the script marks log-normal as unavailable for that feature.
- KDE is a flexible exploratory density estimate, but bandwidth affects its shape. It is not automatically superior to a Gaussian fit and should be assessed alongside downstream validation.
- These diagnostics are descriptive on the full dataset. They must not be used to claim held-out model performance. In model evaluation, every distribution parameter must be fitted using the training partition only.

## Current verification status

The script was statically reviewed against the repository's dataset schema. It has **not yet been executed end-to-end in this environment**, so no Shapiro-Wilk p-values, fitted plots, or categorical probability tables are claimed as generated results. Run it locally to produce measured diagnostics before deciding which continuous likelihoods to use in the classifier.

## Sources & credits

- SciPy `shapiro`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html. Official API reference for the normality test; no implementation code copied.
- SciPy `probplot`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html. API reference for normal Q-Q plots; no example code copied.
- pandas KDE documentation: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.kde.html. Reference for KDE interpretation and bandwidth dependence; script uses SciPy's `gaussian_kde` API.
