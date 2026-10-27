# Repository health audit

**Audit date:** 2026-10-28  
**Scope:** project requirements, repository structure, model implementation, API, React frontend, tests, dependency management, CI, documentation, and licensing.

## Summary

BayesPulse implements the core probability/statistics requirements and the user has confirmed the API checks, 33 Python unit tests, and the initial React production build passed. The React UI is connected to the FastAPI adapter, and the original Streamlit interface and NumPy classifier are retained.

The repository is **not yet at a fully verified release state**. The frontend build was added to GitHub Actions during this audit, but that workflow run has not completed at the time this report was written. The following maintenance gaps remain open.

## Requirements review

- Cleveland UCI dataset with 303 records and a binary target `num > 0`: implemented.
- From-scratch hybrid Naive Bayes with Gaussian continuous likelihoods and smoothed categorical likelihoods: implemented in `bayespulse/naive_bayes.py`.
- Log-space posterior normalization and missing-predictor support: implemented and unit-tested.
- GaussianNB equivalence validation: implemented by `scripts/validate_against_sklearn.py`; the user previously confirmed it passed.
- Held-out evaluation, calibration diagnostics, bootstrap intervals, decision-cost sensitivity, dependence diagnostics, and QDA comparison: implemented as analysis scripts.
- Feature evidence and sequential posterior progression: exposed through the model, Streamlit app, FastAPI API, and React UI.
- Educational-only health disclaimer and full-dataset interactive-model limitation: documented in the interfaces and README.
- React + TypeScript frontend and FastAPI API: implemented. User confirmed the initial frontend production build and the API prediction endpoint worked.
- Automated frontend build in CI: added in commit `61ea0b4583410717ecff96ec77cd12c55aa5ffef`; the run is pending/in progress and is not yet recorded as passing.

## Findings and remaining work

1. **Frontend lockfile missing.** `frontend/package-lock.json` is not committed. `npm install` works, but a fresh install can resolve newer transitive versions over time. Generate and commit the lockfile from a network-enabled local checkout, then use `npm ci` in CI.
2. **Source-code license undecided.** No root `LICENSE` exists. The README now states this clearly. Do not assume the dataset's license grants reuse rights to this project's code. The user must choose the intended project-code license before one is added.
3. **Frontend automated tests are absent.** The frontend currently has a TypeScript/production-build check in CI, but no React component or browser end-to-end tests. The existing Python tests cover model, API, analysis helpers, and Streamlit model/data integration.
4. **Python dependencies are minimum-version ranges, not a lockfile.** The build has emitted a Starlette/TestClient deprecation warning around HTTPX in the local environment. It did not fail the previously reported test run, but the environment should be pinned and the warning resolved after checking compatible FastAPI/Starlette/http client versions.
5. **Step-named reports remain.** They contain scientific methodology and historical verification records, but some also contain dated workflow notes. Because the full repository verification is not yet green and the source license is unresolved, they have not been deleted during this audit. Once the open checks are resolved, consolidate useful scientific content into durable methodology/evaluation documentation and remove process logs.
6. **Latest frontend style revision needs a fresh production build.** The user previously confirmed the initial React build. Subsequent styling-only changes should be covered by the newly added CI frontend build.

## Scientific evaluation snapshot

The recorded fixed stratified split uses 227 training records and 76 held-out records (seed 42). Hybrid Naive Bayes metrics at threshold 0.5 were:

| Metric | Result |
|---|---:|
| Accuracy | 0.8816 |
| Class-1 precision | 0.8250 |
| Class-1 recall | 0.9429 |
| Class-1 F1 | 0.8800 |
| ROC-AUC | 0.9742 |
| Brier score | 0.0796 |

The QDA comparator had slightly higher accuracy on that split (0.8947), while hybrid Naive Bayes had higher class-1 recall (0.9429 vs 0.9143), higher ROC-AUC (0.9742 vs 0.9519), and lower Brier score (0.0796 vs 0.0949). These are single-split results on a small historical dataset and do not establish clinical validity or real-world generalization.

## Repository hygiene updates made during this audit

- Rewrote the root README as project documentation, removing the step-by-step development narrative and adding setup, API, tests, evaluation, limitations, and licensing status.
- Added ignores for `frontend/node_modules/`, `frontend/dist/`, and local frontend environment files.
- Added Node.js setup, dependency installation, and `npm run build` to the GitHub Actions workflow.
- Kept the Streamlit interface and classifier implementation unchanged.

## Release gate

Do not describe the repository as fully verified until the latest GitHub Actions run passes, the frontend lockfile is committed, the local frontend build is repeated after the latest style update, and the project-code license decision is made. No clinical deployment or medical use is appropriate.
