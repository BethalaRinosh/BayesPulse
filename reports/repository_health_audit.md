# Repository health, bug, and code audit

**Audit date:** 2026-10-28  
**Repository:** [BethalaRinosh/BayesPulse](https://github.com/BethalaRinosh/BayesPulse)  
**Scope:** Model implementation, tests, analysis scripts, Streamlit app, dependencies, documentation, dataset provenance, CI, and repository hygiene.

## Executive result

- **Step 8 passed:** the user ran 20 tests and the feature-contribution demo; the contribution sum reconstructed posterior log-odds.
- **Step 9 passed:** the user ran 24 tests, the GaussianNB equivalence comparison printed PASS, and the uncertainty/QDA analysis completed. GitHub Actions passed for the Step 9 commit series.
- **Step 10 and clean-clone verification passed:** `app.py` compiled, all 25 tests passed in the original environment, and Streamlit launched with the user confirming the UI looked fine. A fresh clone with newer dependency versions initially exposed a read-only NumPy array in test setup. The test row now uses `.copy()` before mutation, and the user reran the clean-clone checks successfully: `py_compile` passed and all 25 tests passed in 0.553 seconds. Non-fatal Streamlit cache/script-context warnings appeared in bare test mode; the user's app run also emitted a `use_container_width` deprecation warning.

## Previously identified defects and fixes

1. Categorical smoothing now includes a shared OTHER bucket in the denominator so known categories plus OTHER form a normalized distribution.
2. `alpha` and `variance_smoothing` must be finite and satisfy their sign constraints.
3. Non-integer and boolean feature indices are rejected.
4. Fitting rejects continuous features with no observed values in one class.
5. A failed refit cannot leave the model marked fitted.
6. Distribution diagnostics reject missing/invalid targets, entirely missing predictor columns, and empty per-class groups.
7. README and stage reports distinguish verified results from pending work and state the educational-only scope.

## Step 9 verification record

- 24 tests passed in 0.464 seconds.
- GaussianNB comparison passed: priors matched; maximum fitted mean/variance difference was 2.842e-13; maximum held-out probability difference was 3.109e-15; predictions matched.
- The uncertainty analysis completed on 227 training rows and 76 held-out rows with 2,000 bootstrap replicates.
- Hybrid Naive Bayes point metrics: accuracy 0.8816, class-1 recall 0.9429, ROC-AUC 0.9742, Brier score 0.0796.
- QDA comparator point metrics: accuracy 0.8947, class-1 recall 0.9143, ROC-AUC 0.9519, Brier score 0.0949.
- Hybrid Naive Bayes bootstrap 95% intervals: accuracy [0.8026, 0.9474], class-1 recall [0.8571, 1.0000], ROC-AUC [0.9394, 0.9951], Brier score [0.0393, 0.1268].
- GitHub Actions passed for the Step 9 commit `9d55ca5fd16f6ff77a56fc5e5c90a85165ae7b58`.

These results are conditional on one small fixed split and are not clinical validation.

## Step 10 implementation review

- `app.py` exposes continuous inputs, categorical-coded dropdowns, optional missing predictors, posterior probabilities, per-feature log-likelihood ratios, and a Bayesian updating trace.
- The app trains on the full 303-row dataset for interactive demonstration. Its output is not a held-out prediction; this is stated in the UI and README.
- The UI warns users not to enter real patient information and not to use results for diagnosis or patient care.
- The workflow checks `python -m py_compile app.py`; the new integration smoke test exercises the data/model interface and checks that feature contributions reconstruct posterior log-odds.
- The user confirmed that the browser UI looked fine in the original environment. The integration smoke test covers model/data output, but it does not automate full browser interaction. The clean-clone test issue was fixed by copying the NumPy row before mutation; a fresh clone then passed `py_compile` and all 25 tests.

## Remaining repository gaps

- `requirements.txt` specifies minimum versions rather than a fully locked environment.
- No `LICENSE` file declares reuse permissions for project source code; the dataset license does not automatically license the code.
- Fresh-clone verification is now complete: after pulling the `.copy()` fix, `py_compile` passed and all 25 tests passed in 0.553 seconds under newer dependencies.
- Streamlit browser behavior and missing-feature interaction need manual verification.

## Step 10 verification record\n\n```powershell\npython -m py_compile app.py\npython -m unittest discover -s tests -v\nstreamlit run app.py\n```\n\n- Original environment: syntax check passed, 25 tests passed, Streamlit started at `http://localhost:8501`, and the user reported that the UI looked fine.\n- Fresh clone with newer dependencies: syntax check passed and all 25 tests passed in 0.553 seconds after fixing the read-only test array.\n- Bare test mode produced non-fatal Streamlit cache/script-context warnings. The app run also emitted a deprecation warning for `use_container_width`.\n\n## Current conclusion\n\nSteps 8, 9, 10, and clean-clone verification are complete. Remaining maintenance items include dependency pinning, choosing a source-code license, checking the latest GitHub Actions run, and replacing deprecated Streamlit width arguments.\n