# Step 5: Implementation validation against scikit-learn

## Purpose

Validate the continuous-only Gaussian path of the from-scratch classifier against scikit-learn's `GaussianNB` implementation under matching assumptions. This is an implementation-equivalence check, not a model-selection result, clinical claim, or substitute for held-out evaluation in Step 6.

## Reference comparison

Added `scripts/validate_against_sklearn.py`. It:

1. Loads the Cleveland dataset and maps `num > 0` to the binary target.
2. Uses only `age`, `trestbps`, `chol`, `thalach`, and `oldpeak`, so both models use Gaussian feature likelihoods.
3. Makes a stratified 75/25 train/test split with random seed 42.
4. Fits both models on the training partition only.
5. Compares class priors, fitted means and variances, held-out probabilities, and predicted classes.
6. Fails with an assertion if values differ beyond a tolerance of `1e-8`.

The reference uses `GaussianNB(var_smoothing=0.0)` so its raw population variances match the from-scratch implementation's no-smoothing comparison mode. The Cleveland continuous features have no missing values; the script explicitly checks this assumption before splitting.

## Run

```bash
python scripts/validate_against_sklearn.py
```

## Runtime findings supplied by the user

The user's Step 4 terminal output shows that dependency installation found all declared packages already installed. The EDA and distribution-fitting scripts completed successfully, produced the expected 303-record summaries, and wrote their outputs. The output reports 164 class-0 and 139 class-1 records, 4 missing `ca` values and 2 missing `thal` values, and 46 rows in the categorical likelihood table.

The same output reports **9 unit tests run, 1 error**: `test_predict_before_fit_is_rejected` raised `AttributeError` because `predict()` accessed `classes_` before the fitted-state check in `predict_log_proba()`. This was a genuine bug. `predict()` now checks the fitted state first and raises `RuntimeError("Call fit before predict.")`; the test now checks for that clear error.

## Windows import-path fix

The user ran the unit suite from the repository root on Windows: all 11 tests passed. The subsequent comparison command failed before fitting either model because executing a script inside `scripts/` did not put the repository root on Python's import path (`ModuleNotFoundError: No module named 'bayespulse'`). The script now inserts its own repository root into `sys.path` before importing the project package, making the documented root-level command self-contained. This is a path/bootstrap fix, not a model result.

## Verification status and remaining checks

- User-confirmed: package installation succeeded; EDA and distribution diagnostics ran successfully.
- User-confirmed failure: 1 of 9 Step 4 tests errored before the fix.
- Fix committed: explicit fitted-state guard in `predict()`; matching test assertion updated.
- Confirmed by the user's later run: the post-fix unit suite passes all 11 tests.
- Confirmed: all 11 unit tests pass after the fitted-state guard fix.
- Confirmed by the user's later run: class priors match; maximum fitted mean/variance difference is 2.842e-13; maximum held-out probability difference is 3.109e-15; held-out predictions match; the script reports PASS.
- A clean clone was not independently run by the assistant environment.

## Interpretation guardrails

- Agreement with GaussianNB on continuous features checks that the NumPy Gaussian calculations align with a trusted implementation when assumptions match.
- It does not validate the hybrid categorical component against GaussianNB, establish that Gaussian likelihoods are the best fit for every feature, or demonstrate clinical utility.
- Feature distributions and all model parameters must be fitted on training data only for any evaluation.
- Step 6's earlier local evaluation results are recorded in `reports/step6_evaluation.md`. Those metrics predate the later categorical-smoothing audit fix and must be refreshed for the current implementation. See `reports/repository_health_audit.md`.
