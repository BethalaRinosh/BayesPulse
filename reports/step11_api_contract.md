# Frontend migration, Step 1: API contract and safety baseline

## Goal

Create a stable HTTP boundary for a future React + TypeScript frontend without changing the existing classifier or removing the working Streamlit interface.

## Implementation on `ui/react-fastapi-migration`

- Added `backend/main.py` as a thin FastAPI adapter.
- Added health, metadata, and prediction endpoints under `/api/v1`.
- Kept the canonical feature order identical to the existing Streamlit app: five continuous predictors followed by eight categorical-coded predictors.
- Reused `HybridNaiveBayes` directly for probabilities, feature log-likelihood ratios, and Bayesian updating. No model math is reimplemented in the API.
- Added request validation for unknown feature names and non-finite numeric values. Omitted/null predictors are represented as `NaN` for the existing model's missing-feature behavior.
- Added localhost-only CORS for the planned Vite development server.
- Added six API contract tests covering health, metadata ordering, exact prediction parity with direct model calls, missing inputs, unknown feature rejection, and non-finite input rejection.
- Added FastAPI, Uvicorn, and HTTPX dependencies. The Streamlit dependency and `app.py` are retained.
- Updated CI to compile both app entry points and run tests on the migration branch.

## Safety invariants

1. `bayespulse/naive_bayes.py` is not modified by this step.
2. The API must preserve posterior probabilities from direct model calls to floating-point tolerance.
3. Feature evidence remains model log-likelihood ratios, not causal explanations.
4. The API labels outputs as recorded dataset class probabilities, not future individual disease risk.
5. No real patient information should be entered or stored.
6. The React frontend must not be started until this API contract passes tests in a clean checkout.

## Verification status

Implementation and tests have been committed to the migration branch. Local execution cannot be claimed from the repository editing tool alone; the user should run the exact commands in the next message and share the output before we proceed. GitHub Actions status is not yet confirmed for the latest branch commit.
