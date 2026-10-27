# BayesPulse

**Medical Risk Prediction Using Probabilistic Feature Modelling**

BayesPulse is an educational Probability & Statistics project that implements a transparent hybrid Naive Bayes classifier for a binary classification task using the Cleveland subset of the UCI Heart Disease dataset.

> **Important:** This is a student research project, not a medical device. Its output is not a diagnosis, treatment recommendation, or clinically validated estimate of an individual's health risk.

## Current status

- The NumPy hybrid Naive Bayes classifier, dataset diagnostics, GaussianNB equivalence check, held-out evaluation, and sequential Bayesian updating demo are implemented.
- **Repository health verification passed locally:** 16 unit tests passed, the GaussianNB equivalence check passed, and all analysis/evaluation/demo scripts completed successfully after the audit fixes. See `reports/repository_health_audit.md`.
- Step 8 is in progress: per-feature log-likelihood ratios are implemented with tests and a demo script, but the new Step 8 tests/demo have not yet been run locally.
- There is no Streamlit app yet. A GitHub Actions verification workflow has been committed, but its first successful run remains unconfirmed. The repository has no declared license for the project source code; the dataset's license does not automatically license the code.

## What it does

- Estimates empirical class priors from training labels.
- Models continuous predictors with class-conditional Gaussian likelihoods.
- Models discrete-coded predictors with Laplace-smoothed categorical likelihoods.
- Handles missing predictors by skipping their likelihood contribution rather than silently converting missing values to zero.
- Normalizes posterior calculations in log space.
- Exposes feature-by-feature posterior updates for an educational demonstration.
- Decomposes binary posterior log-odds into prior odds and per-feature log-likelihood ratios.
- Compares the continuous-only implementation with scikit-learn's `GaussianNB`.
- Evaluates a fixed stratified held-out split and produces metrics, calibration data, and diagnostic plots.

## Dataset and target

- **Source:** [UCI Machine Learning Repository: Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Local file:** `data/cleveland.csv`
- **Rows / columns:** 303 records, 13 predictors, and original target `num`
- **Binary target:** `0` when `num == 0`; `1` when `num > 0`
- **Missing predictors:** four missing `ca` values and two missing `thal` values in the checked dataset.
- **Dataset license:** CC BY 4.0, as documented by UCI. Retain dataset attribution.

This is a small historical dataset. It does not represent all current populations, and the target is recorded disease status in the dataset, not future individual risk.

## Repository layout

```text
BayesPulse/
├── .github/
│   └── workflows/
│       └── verify.yml               # Automated tests and script checks
├── bayespulse/
│   └── naive_bayes.py               # NumPy-only hybrid Naive Bayes estimator
├── data/
│   └── cleveland.csv                # Cleveland subset
├── reports/
│   ├── step2_dataset_eda.md
│   ├── step4_model.md
│   ├── step4_repo_audit.md
│   ├── step5_validation.md
│   ├── step5_repo_audit.md
│   ├── step6_evaluation.md
│   ├── step7_bayesian_updating.md
│   ├── step8_feature_contributions.md
│   └── repository_health_audit.md
├── scripts/
│   ├── explore_dataset.py
│   ├── fit_distributions.py
│   ├── validate_against_sklearn.py
│   ├── evaluate_model.py
│   ├── bayesian_updating_demo.py
│   └── feature_contributions_demo.py
├── tests/
│   └── test_naive_bayes.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Recommended: Python 3.10 or newer in a virtual environment.

```powershell
git clone https://github.com/BethalaRinosh/BayesPulse.git
cd BayesPulse
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the Python executable directly at `.venv\Scripts\python.exe` or follow your local Python environment policy.

## Verification and run commands

Run these from the repository root after pulling the latest `main`:

```powershell
git pull origin main
python -m unittest discover -s tests -v
python scripts/explore_dataset.py
python scripts/fit_distributions.py
python scripts/validate_against_sklearn.py
python scripts/evaluate_model.py
python scripts/bayesian_updating_demo.py
python scripts/feature_contributions_demo.py
```

The latest locally confirmed pre-Step-8 verification run passed all **16 tests** and all six analysis/validation/evaluation/demo commands. Step 8 adds four more unit tests; its expanded suite and new contribution demo must be run after pulling the latest changes.

### What the scripts generate

- `explore_dataset.py`: summary CSVs and exploratory plots in `reports/eda/`.
- `fit_distributions.py`: per-class distribution diagnostics and categorical likelihood tables in `reports/distribution_fitting/`.
- `validate_against_sklearn.py`: checks continuous-only parameter, prior, probability, and prediction equivalence with `GaussianNB`.
- `evaluate_model.py`: held-out metrics, calibration table, and ROC/confusion-matrix/calibration plots in `reports/evaluation/`.
- `bayesian_updating_demo.py`: a deterministic held-out sample's posterior after each feature is incorporated.
- `feature_contributions_demo.py`: the prior log-odds and each held-out feature's log-likelihood ratio, with positive/negative model-evidence direction.

Generated diagnostic outputs are ignored by Git and can be recreated by rerunning the scripts.

## Using the estimator

```python
import numpy as np
from bayespulse.naive_bayes import HybridNaiveBayes

# X is a numeric 2D NumPy array; use np.nan for missing predictor values.
model = HybridNaiveBayes(
    continuous_features=(0, 1),
    categorical_features=(2, 3),
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)
```

The feature indices are examples only. Supply the correct indices for your dataset and decide deliberately which variables are continuous versus categorical. Fit preprocessing and model parameters on training data only.

## Model assumptions and limitations

- Naive Bayes assumes features are conditionally independent given the class; real-world predictors may be correlated.
- Continuous features are currently modelled with Gaussian likelihoods regardless of exploratory diagnostics. The diagnostic script does not automatically select a better distribution.
- An unseen categorical code is mapped to a shared smoothed `OTHER` bucket.
- Feature contributions are log-likelihood ratios under the fitted model, not causal effects, medical explanations, or evidence that a feature changes an individual's health.
- Missing continuous observations must be available in both classes during fitting; otherwise the model rejects the training data rather than silently using different feature evidence for different classes.
- The current evaluation is a single split of a small dataset. Metrics and calibration bins can vary substantially with the split.
- No clinical validation, external validation, clinical threshold selection, or deployment safety evaluation has been performed.

## References and licensing

- [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [scikit-learn Naive Bayes documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- [SciPy Shapiro-Wilk documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
- [SciPy probability plot documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html)

The dataset is attributed to UCI and its listed creators. **Project source-code licensing has not yet been selected.** Add a license file only after deciding what reuse permissions you want to grant.
