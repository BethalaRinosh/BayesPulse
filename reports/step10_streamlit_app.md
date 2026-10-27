# Step 10: Streamlit educational interface

## Goal

Provide a simple, inspectable UI for demonstrating the existing hybrid Naive Bayes model, without presenting it as a medical product.

## Implementation

Added `app.py`, launched from the repository root with `streamlit run app.py`.

The interface includes:
- Continuous-feature number inputs using observed dataset ranges and median defaults.
- Categorical-coded dropdowns with readable labels where the source code meanings are known.
- An optional control to mark selected predictors missing, which exercises the model's documented missing-feature behavior.
- Posterior probabilities for the dataset's binary class labels.
- A feature-evidence table and bar chart using the model's per-feature log-likelihood ratios.
- A Bayesian updating trace showing posterior probabilities as each feature is incorporated.
- An expandable explanation of the dataset, assumptions, and limitations.
- Prominent educational-only warnings. The interface tells users not to enter real patient information.

The model is trained on the full 303-record dataset for this interactive demonstration. This is intentionally described in the UI so users do not confuse its output with a held-out prediction or external validation result.

## Repository changes

- Added `app.py`.
- Updated README with setup, launch instructions, current Step 9 results, and the Step 10 status.
- Added a generated Step 9 analysis report to `.gitignore`.
- Updated GitHub Actions to run `python -m py_compile app.py` before the existing tests and analysis scripts.
- Added `tests/test_streamlit_app.py` to smoke-test dataset loading, model predictions, feature evidence, and posterior reconstruction. The expected test count is now 25.
- Fixed the app to use the estimator's actual `feature_log_likelihood_ratios` result key.

## Commands

From the repository root:

```powershell
git pull origin main
python -m py_compile app.py
streamlit run app.py
```

Then verify the UI manually:
1. Confirm the page loads without a traceback.
2. Confirm continuous and categorical controls display and can be changed.
3. Click **Run Bayesian demonstration**.
4. Confirm the posterior metrics, feature-evidence table/chart, and Bayesian updating trace appear.
5. Mark at least one feature missing and rerun; its evidence should be skipped and the app should remain functional.
6. Check that the educational warning remains visible.

## Verification status

**Verification passed on 2026-10-28.** In the original environment, `python -m py_compile app.py` passed, all **25 tests passed**, Streamlit started at `http://localhost:8501`, and the user reported that the UI looked fine. A clean clone with newer dependencies initially exposed a read-only NumPy array in test setup. The row now calls `.copy()` before categorical-mode assignment. After pulling the fix, the user reran `python -m py_compile app.py` and `python -m unittest discover -s tests -v`: **all 25 tests passed in 0.553 seconds**. Streamlit cache/script-context warnings in bare test mode were non-fatal. This is now verified in both the original environment and a fresh clone.

## Safety and interpretation limits

- The app is a coursework demonstration, not a diagnosis or clinical decision tool.
- Do not enter real patient information.
- The app trains on the entire historical dataset for interactive exploration, so its displayed posterior is not an independent held-out prediction.
- The binary target represents recorded disease status in this dataset, not future individual risk.
- Model feature contributions are likelihood ratios under the model's assumptions, not causal or medical explanations.
- The dataset is small and may not represent other populations or current clinical practice.
