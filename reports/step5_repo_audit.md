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
| `scripts/validate_against_sklearn.py` | Step 5 continuous Gaussian reference comparison | User confirmed numerical parity within tolerance |
| `bayespulse/naive_bayes.py` | From-scratch hybrid classifier | Fixed prediction-before-fit guard |
| `tests/test_naive_bayes.py` | Classifier unit tests | Assertion strengthened; rerun pending |
| `reports/step2_dataset_eda.md` | Dataset and Step 3 audit | Preserved |
| `reports/step4_model.md` | Model design | Present |
| `reports/step4_repo_audit.md` | Step 4 audit and follow-up | Updated with actual user output |
| `reports/step5_validation.md` | Step 5 design and evidence | Updated with user-confirmed successful run |
| `scripts/evaluate_model.py` | Step 6 held-out hybrid-model evaluation | Added; local execution pending |
| `reports/step6_evaluation.md` | Step 6 protocol and completion gate | Added |

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

The user ran the comparison successfully. Class priors and held-out predictions matched; maximum fitted mean/variance difference was 2.842e-13 and maximum held-out probability difference was 3.109e-15. The script uses only continuous features and `GaussianNB(var_smoothing=0.0)` to align the reference's raw population variance with the from-scratch model.

## Hygiene changes

Added `.gitignore` rules for Python bytecode/caches, local virtual environments, editor files, generated EDA/distribution outputs, and generated Step 6 evaluation plots/results under `reports/evaluation/`. Generated outputs remain reproducible through the scripts and should not accidentally clutter commits.

## Required local commands after pulling latest main

```bash
git pull origin main
python -m unittest discover -s tests -v
python scripts/validate_against_sklearn.py
```

Step 5 completion evidence is confirmed in the user's latest output: all 11 tests pass and the GaussianNB comparison reports PASS. Step 6 has started; run `python scripts/evaluate_model.py` and inspect the metrics, calibration bins, report, and plots.

## Gate

Step 5 runtime validation is complete based on the user's supplied terminal output. Step 6's held-out evaluation script is committed, but its runtime output has not yet been reviewed.
