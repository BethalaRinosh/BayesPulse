# Step 5 repository, dependency, and bug audit

## Current tracked repository inventory

Verified the GitHub tree after Step 5 updates. GitHub reported `truncated: false`.

| Path | Role | Audit result |
|---|---|---|
| `.gitignore` | Ignore Python caches, virtual environments, generated exploratory outputs | Added |
| `README.md` | Plan, provenance, commands, current status | Updated for Step 5 |
| `requirements.txt` | Minimum versions for NumPy, pandas, SciPy, matplotlib, scikit-learn, Streamlit | Present; install succeeded in user's environment |
| `data/cleveland.csv` | 303-record Cleveland dataset | User's EDA script loaded it successfully |
| `scripts/explore_dataset.py` | Step 2 EDA | User's run completed and wrote reports |
| `scripts/fit_distributions.py` | Step 3 distribution diagnostics | User's run completed and wrote diagnostics |
| `scripts/validate_against_sklearn.py` | Step 5 continuous Gaussian reference comparison | Added; execution result pending |
| `bayespulse/naive_bayes.py` | From-scratch hybrid classifier | Fixed prediction-before-fit guard |
| `tests/test_naive_bayes.py` | Classifier unit tests | Assertion strengthened; rerun pending |
| `reports/step2_dataset_eda.md` | Dataset and Step 3 audit | Preserved |
| `reports/step4_model.md` | Model design | Present |
| `reports/step4_repo_audit.md` | Step 4 audit and follow-up | Updated with actual user output |
| `reports/step5_validation.md` | Step 5 design and evidence | Added |

No unrelated files were removed. The repo does not yet include a Streamlit application, a locked dependency file, or a CI workflow.

## User-run evidence

The supplied terminal log establishes that:

- `pip install -r requirements.txt` completed with all declared requirements already satisfied in Python 3.11.
- `scripts/explore_dataset.py` completed successfully. It printed shape `(303, 15)` because it adds the derived `has_disease` column to the 14-column source, and reported 164 class-0 and 139 class-1 records.
- `scripts/fit_distributions.py` completed successfully and printed per-class diagnostics for five continuous features plus 46 categorical likelihood rows.
- The original unit suite ran 9 tests and had one error in prediction-before-fit handling.

The 303-row and class-count output is consistent with the documented source data. The 15-column EDA output is expected after adding `has_disease`, not a dataset schema bug.

## Bug fixed

Before the fix, `predict()` evaluated `self.classes_` before calling `predict_log_proba()`, so a fresh estimator raised `AttributeError` rather than a deliberate fitted-state error. `predict()` now checks `self._fitted` first and raises `RuntimeError("Call fit before predict.")`. The unit test now asserts this message.

The fix has been fetched back from GitHub and verified in the current source. However, the user's original log predates this change. A new test run is required to establish the fix works in their local checkout.

## Dependency audit

The user environment reported Python 3.11 and the following installed versions: NumPy 1.26.4, pandas 2.3.3, SciPy 1.17.1, matplotlib 3.10.8, scikit-learn 1.9.0, and Streamlit 1.65.0. The dependency installation command reported these requirements already satisfied. The repository uses lower bounds rather than exact pins, so it is installable but not strictly environment-reproducible.

## Comparison design audit

The new validation script uses the same stratified train/test split for both models, fits only on the training partition, and compares class priors, Gaussian means/variances, held-out probabilities, and predicted classes. It uses only continuous features and `GaussianNB(var_smoothing=0.0)` to align the reference's raw population variance with the from-scratch model. It must be run before numerical agreement can be claimed.

## Hygiene changes

Added `.gitignore` rules for Python bytecode/caches, local virtual environments, editor files, and generated CSV/PNG files under `reports/eda/` and `reports/distribution_fitting/`. Generated outputs remain reproducible through the scripts and should not accidentally clutter commits.

## Required local commands after pulling latest main

```bash
git pull origin main
python -m unittest discover -s tests -v
python scripts/validate_against_sklearn.py
```

Expected next evidence: all unit tests pass, then the comparison script prints maximum parameter/probability differences and matching predictions. If either command fails, preserve and share the complete traceback/output before proceeding to Step 6.

## Gate

Step 5 implementation files and documentation are committed, but runtime validation is **not yet complete** because the prediction fix has not been rerun locally and the GaussianNB comparison has not yet produced results. The project must pause here until those two commands are run and reviewed. Step 6 has not been started.
