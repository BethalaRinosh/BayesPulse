# BayesPulse

**Medical Risk Prediction Using Probabilistic Feature Modelling**

BayesPulse is an educational Probability & Statistics project that implements a transparent hybrid Naive Bayes classifier for a binary classification task using the Cleveland subset of the UCI Heart Disease dataset.

> **Important:** This is a student research project, not a medical device. Its output is not a diagnosis, treatment recommendation, or clinically validated estimate of an individual's health risk.

## Current status

- The NumPy hybrid Naive Bayes classifier, dataset diagnostics, GaussianNB equivalence check, held-out evaluation, sequential Bayesian updating, per-feature log-odds explanation, and uncertainty/dependence/model-comparison analysis are implemented.
- **Step 8 verified:** 20 unit tests passed; the contribution sum reconstructed posterior log-odds.
- **Step 9 verified:** 24 unit tests passed; GaussianNB equivalence validation passed; uncertainty and QDA analysis completed; GitHub Actions passed. Results are documented in `reports/step9_uncertainty_decision_analysis.md`.
- **Step 10 app implementation is pushed and awaiting local UI verification:** `app.py` provides an interactive Streamlit demonstration with feature inputs, modelled posterior probabilities, per-feature log-likelihood ratios, and a Bayesian updating trace.
- The app fits on the full dataset for an educational interactive demonstration. Its displayed probabilities are not held-out test predictions, clinical risk estimates, or evidence of generalization.
- The project source-code license has not been selected. The dataset's license does not automatically license this repository's source code.

## What it does

- Estimates empirical class priors from training labels.
- Models continuous predictors with class-conditional Gaussian likelihoods.
- Models discrete-coded predictors with Laplace-smoothed categorical likelihoods.
- Handles missing predictors by skipping their likelihood contribution rather than silently converting missing values to zero.
- Normalizes posterior calculations in log space.
- Exposes feature-by-feature posterior updates.
- Decomposes binary posterior log-odds into prior odds and per-feature log-likelihood ratios.
- Compares the continuous-only implementation with scikit-learn's `GaussianNB`.
- Reports held-out metrics, calibration diagnostics, bootstrap intervals, decision-cost sensitivity, feature-dependence measures, and a QDA comparator.
- Provides an interactive educational interface through Streamlit.

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
├── app.py                           # Streamlit educational interface
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
│   ├── step9_uncertainty_decision_analysis.md
│   ├── step10_streamlit_app.md
│   └── repository_health_audit.md
├── scripts/
│   ├── explore_dataset.py
│   ├── fit_distributions.py
│   ├── validate_against_sklearn.py
│   ├── evaluate_model.py
│   ├── bayesian_updating_demo.py
│   ├── feature_contributions_demo.py
│   └── uncertainty_decision_analysis.py
├── tests/
│   ├── test_naive_bayes.py
│   └── test_step9_analysis.py
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

## Launch the interactive app

From the repository root:

```powershell
streamlit run app.py
```

The app lets you enter a hypothetical record, inspect the modelled class probabilities, review feature log-likelihood ratios, and see how the posterior changes as features are incorporated. It uses original coded categorical values and labels them where practical.

**Privacy and safety:** Do not enter real patient information. The app is for coursework and model interpretation only. It trains the demonstration model on the full 303-row dataset, so its interactive output is not an independent held-out prediction.

## Verification and analysis commands

Run these from the repository root after pulling the latest `main`:

```powershell
git pull origin main
python -m py_compile app.py
python -m unittest discover -s tests -v
python scripts/explore_dataset.py
python scripts/fit_distributions.py
python scripts/validate_against_sklearn.py
python scripts/evaluate_model.py
python scripts/bayesian_updating_demo.py
python scripts/feature_contributions_demo.py
python scripts/uncertainty_decision_analysis.py
```

### Step 9 confirmed results

The user ran the 24-test suite successfully in 0.464 seconds, the GaussianNB comparison printed PASS, and the uncertainty analysis completed on 227 training rows and 76 held-out rows using 2,000 bootstrap replicates. GitHub Actions also passed for the Step 9 commit.

| Metric | Hybrid Naive Bayes | QDA comparator |
|---|---:|---:|
| Accuracy at threshold 0.5 | 0.8816 | 0.8947 |
| Class-1 recall | 0.9429 | 0.9143 |
| ROC-AUC | 0.9742 | 0.9519 |
| Brier score | 0.0796 | 0.0949 |

Hybrid Naive Bayes bootstrap percentile 95% intervals, conditional on the fixed test partition:

| Metric | Bootstrap median | 95% interval |
|---|---:|---:|
| Accuracy | 0.8816 | [0.8026, 0.9474] |
| Class-1 recall | 0.9429 | [0.8571, 1.0000] |
| ROC-AUC | 0.9763 | [0.9394, 0.9951] |
| Brier score | 0.0776 | [0.0393, 0.1268] |

These metrics do not establish clinical validity. QDA has slightly higher accuracy at the default threshold, while hybrid Naive Bayes has higher recall and ROC-AUC and lower Brier score on this one split.

### What the scripts generate

- `explore_dataset.py`: summary CSVs and exploratory plots in `reports/eda/`.
- `fit_distributions.py`: per-class distribution diagnostics and categorical likelihood tables in `reports/distribution_fitting/`.
- `validate_against_sklearn.py`: checks continuous-only parameter, prior, probability, and prediction equivalence with `GaussianNB`.
- `evaluate_model.py`: held-out metrics, calibration table, and ROC/confusion-matrix/calibration plots in `reports/evaluation/`.
- `bayesian_updating_demo.py`: a deterministic held-out sample's posterior after each feature is incorporated.
- `feature_contributions_demo.py`: prior log-odds and each held-out feature's log-likelihood ratio.
- `uncertainty_decision_analysis.py`: bootstrap intervals, illustrative error-cost threshold sweep, dependence diagnostics, mutual information, and QDA comparison. It writes a generated report under `reports/evaluation/`.

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
- Bootstrap intervals in Step 9 are conditional on the fixed held-out partition. They do not include uncertainty from retraining or a new sample.
- Step 9's false-negative and false-positive costs are illustrative placeholders, not clinically or economically validated values. No threshold is recommended for medical use.
- QDA is a comparator with different assumptions and preprocessing; a single split does not establish which model generalizes better.
- The Streamlit app fits on the full dataset for demonstration. Its output is not a held-out prediction and must not be used for patient-care decisions.
- No clinical validation, external validation, clinical threshold selection, or deployment safety evaluation has been performed.

## References and licensing

- [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [scikit-learn Naive Bayes documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn discriminant analysis documentation](https://scikit-learn.org/stable/modules/lda_qda.html)
- [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- [SciPy Shapiro-Wilk documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
- [SciPy probability plot documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html)

The dataset is attributed to UCI and its listed creators. **Project source-code licensing has not yet been selected.** Add a license file only after deciding what reuse permissions you want to grant.
