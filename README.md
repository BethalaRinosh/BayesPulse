# BayesPulse
## Medical Risk Prediction Using Probabilistic Feature Modelling

BayesPulse is an educational micro-project for Probability and Statistics for AI & ML. It will explore probabilistic feature modelling and a transparent Naive Bayes implementation for a binary heart-disease classification task.

> **Educational disclaimer:** BayesPulse is a student project. Its output is not a medical diagnosis, treatment recommendation, or clinically validated estimate of an individual's health risk.

## Project status

- **Current stage:** Step 4 complete — NumPy-only hybrid Naive Bayes implementation and unit tests added; runtime verification status is recorded below.
- **Dataset selected:** UCI Heart Disease, Cleveland subset.
- **Implementation status:** Dataset EDA, distribution diagnostics, and a from-scratch hybrid Naive Bayes classifier are present. The classifier uses Gaussian likelihoods for continuous features and Laplace-smoothed categorical likelihoods for discrete-coded features.
- **Repository audit:** The repository was empty when inspected. The README, dataset, EDA script, and Step 2 report now establish the project baseline; no existing project files were overwritten or removed.

## Planned approach

1. Acquire and document the UCI Cleveland heart-disease dataset and target definition.
2. Explore data quality, class balance, feature types, and summary statistics.
3. Fit and diagnose per-class feature distributions. Assess Gaussian assumptions with histograms, Q-Q plots, and Shapiro-Wilk tests where appropriate; consider log-normal or KDE alternatives when justified. Use Bernoulli/categorical probabilities with Laplace smoothing for discrete features.
4. Implement Naive Bayes from scratch in NumPy using class priors, Gaussian continuous-feature likelihoods, Laplace-smoothed categorical likelihoods, and normalized log-space posteriors. (Implemented; see `bayespulse/naive_bayes.py`.)
5. Validate the implementation against scikit-learn's GaussianNB where assumptions align, then audit code and dependencies.
6. Evaluate on a proper held-out split or cross-validation using confusion matrix, precision, recall, F1, ROC-AUC, and a calibration curve.
7. Add a feature-by-feature Bayesian updating demonstration.
8. Add interpretable per-feature log-likelihood ratios.
9. Add bootstrap uncertainty intervals, cost-sensitive thresholds, and feature-dependence/model comparison analysis where feasible.
10. Build a Streamlit demo with input validation, risk estimates, contribution breakdowns, distribution context, and a visible disclaimer.
11. Run end-to-end checks and prepare reproducible setup instructions and viva notes.

The project will be implemented one step at a time. Core probability calculations, distribution fitting logic, and Bayesian updating will be written in readable, commented code rather than copied from an existing implementation.

## Dataset

**UCI Machine Learning Repository: Heart Disease**  
https://archive.ics.uci.edu/dataset/45/heart+disease

The UCI listing describes 303 Cleveland instances and 13 commonly used predictors. The original target `num` has values 0–4; this project plans to define the binary target as 0 = no recorded heart disease and 1 = recorded heart disease (original values 1–4). The selected file is `data/cleveland.csv`. The target mapping, missing-value counts, class counts, descriptive statistics, feature-type plan, and limitations are recorded in `reports/step2_dataset_eda.md`.

Dataset license: **CC BY 4.0**. Attribution will be retained in the project documentation.

## Sources & Credits

These are planned reference sources for the implementation. No third-party project code has been copied into this repository at Step 1.

| Source | Planned use | License / attribution notes |
|---|---|---|
| UCI Heart Disease dataset — https://archive.ics.uci.edu/dataset/45/heart+disease | Canonical dataset source and documentation | CC BY 4.0; cite the UCI dataset and its creators. |
| Data Professor public mirror — https://github.com/dataprofessor/data/blob/master/heart-disease-cleveland.csv | Temporary retrieval fallback when the UCI file endpoint could not be fetched directly; only dataset values were used | The mirror's separate license was not relied upon; attribution and license are taken from the original UCI dataset. No code was copied. |
| SciPy Shapiro documentation — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html | Reference for normality-test behavior | Official SciPy documentation; SciPy is distributed under a modified BSD license. We call the library API, not copy implementation code. |
| SciPy probability plot documentation — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html | Reference for normal Q-Q plots | Official SciPy API documentation; no example code copied. |
| pandas KDE documentation — https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.kde.html | Reference for kernel density estimation and bandwidth interpretation | Documentation reference only; the script calls SciPy's `gaussian_kde` API. |
| scikit-learn Naive Bayes guide — https://scikit-learn.org/stable/modules/naive_bayes.html | Independent sanity-check reference for GaussianNB and Naive Bayes assumptions | scikit-learn is BSD-3-Clause licensed. Used as documentation/reference only; the core model will be our own NumPy implementation. |
| scikit-learn calibration guide — https://scikit-learn.org/stable/modules/calibration.html | Understand and validate reliability diagrams | scikit-learn is BSD-3-Clause licensed. Prefer library APIs for evaluation plumbing rather than copying example code. |
| Streamlit app tutorial — https://docs.streamlit.io/get-started/tutorials/create-an-app | Reference for basic app layout and interaction | Official documentation; use the documented API, not copied tutorial source. |
| NumPy license — https://github.com/numpy/numpy/blob/main/LICENSE.txt | Reference for numerical-computing dependency | BSD-3-Clause; dependencies retain their own licenses. |
| Matplotlib license — https://matplotlib.org/stable/project/license.html | Reference for plotting dependency | PSF/BSD-compatible licensing as described by the project. |

### Reviewed existing implementation

- Example repository: https://github.com/MastersAbh/Heart-Disease-Prediction-using-Naive-Bayes-Classifier
- Its README describes a from-scratch Naive Bayes implementation and a comparison with scikit-learn GaussianNB.
- A LICENSE file was not found at the repository's root during this review, so its reuse permissions are **unclear**. **No code from this repository will be copied.** It may only serve as a high-level topic reference unless a compatible license is confirmed.

## Dataset exploration (Step 2)

- Dataset: 303 rows, 13 predictors, and the original `num` target.
- Original target counts: 0=164, 1=55, 2=36, 3=35, 4=13. Binary mapping (`num > 0`): class 0=164 and class 1=139.
- Missing predictors: `ca` has 4 missing values; `thal` has 2. The target has no missing values. The data file preserves `?` markers.
- EDA command: `python scripts/explore_dataset.py`. It writes descriptive CSVs and histograms under `reports/eda/` when run locally.
- Measured statistics and handling decisions are documented in `reports/step2_dataset_eda.md`.

## Step 3: distribution diagnostics

Install the relevant packages with `pip install numpy pandas scipy matplotlib`, then run `python scripts/fit_distributions.py`. Shapiro-Wilk p-values are evidence about normality, not automatic distribution selectors. The script is exploratory and has not yet been executed end-to-end in this environment; it does not claim measured diagnostic outputs until you run it. See `reports/step2_dataset_eda.md` for interpretation, missing-value handling, and sources.

## Environment

Python with NumPy, pandas, SciPy, matplotlib, scikit-learn (validation and evaluation only), and Streamlit. Install the declared minimum versions with `pip install -r requirements.txt`. These are minimum-version ranges, not a fully locked reproducible environment.

## Run instructions

From the repository root:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/explore_dataset.py
python scripts/fit_distributions.py
```

The model can be imported with `from bayespulse.naive_bayes import HybridNaiveBayes`. It accepts a numeric 2D NumPy array; represent missing feature values as `numpy.nan`, and pass continuous and categorical column indices explicitly. Fit all model parameters on training data only. The Step 4 unit tests cover priors, posterior normalization, separated synthetic classes, missing features, unseen categories, constant features, invalid inputs, single-class/missing-label rejection, and prediction-before-fit. **The tests have been added but have not yet been executed in a clean local clone**, so passing runtime status is not claimed.

## Decisions and open questions

- Dataset choice confirmed: UCI Heart Disease, Cleveland subset.
- Existing repository files: none were present at inspection time. Step 2 added `data/cleveland.csv`, `scripts/explore_dataset.py`, and the dataset audit. Step 3 added `scripts/fit_distributions.py` and the distribution-diagnostic notes.
- The initial classifier uses Gaussian likelihoods for continuous features. Step 3 diagnostics are exploratory and have not been run end-to-end here; distribution alternatives are not automatically selected. No clinical validity or predictive performance claim is made.


## Step 4: from-scratch classifier

- Core implementation: `bayespulse/naive_bayes.py`, using NumPy only for model calculations. It estimates empirical class priors, Gaussian likelihood parameters for designated continuous columns, and Laplace-smoothed categorical probabilities for designated category-code columns.
- Posterior calculations are performed in log space and normalized with a stable log-sum-exp calculation. A variance floor guards against zero-variance features. Missing feature values are skipped per feature, while missing labels, infinities, invalid feature indices, and malformed shapes are rejected.
- Unit tests: `tests/test_naive_bayes.py`. Run `python -m unittest discover -s tests -v`.
- **Verification note:** the files have been statically reviewed; tests have not been run against a clean clone in this environment. Do not describe the tests as passing until the command has been run and its output checked.


## Step 4 repository health audit

The file inventory, known issues, fixes, and remaining verification limits are recorded in [`reports/step4_repo_audit.md`](reports/step4_repo_audit.md). A local logic-level smoke test passed for core synthetic cases, but the committed test suite and data scripts have not been executed in a clean clone because direct GitHub cloning is unavailable in this environment. Run the commands in the audit report locally before treating the project as runtime-verified.
