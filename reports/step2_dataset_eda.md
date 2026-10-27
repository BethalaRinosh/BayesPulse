# Step 2: Dataset acquisition and exploratory analysis

## Dataset provenance

- **Original source:** UCI Machine Learning Repository, Heart Disease (Cleveland subset): https://archive.ics.uci.edu/dataset/45/heart+disease
- **Dataset citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- **License:** CC BY 4.0, as stated by UCI.
- **File:** `data/cleveland.csv`. The original target column `num` is retained with values 0–4. Missing values are represented by `?`.
- **Retrieval note:** The UCI-hosted file endpoint was not directly retrievable in this session. The values were retrieved from a public mirror that identifies UCI as the source, then retained with UCI attribution. The official UCI page remains the canonical provenance source.

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
