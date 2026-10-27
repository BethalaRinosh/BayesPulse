# BayesPulse

**An explainable hybrid Naive Bayes learning lab**

BayesPulse demonstrates how class priors and feature likelihoods combine into posterior probabilities. It implements a hybrid Naive Bayes classifier from scratch with NumPy and exposes its predictions and evidence through a React + TypeScript interface backed by FastAPI. The original Streamlit interface remains available as a fallback.

> **Educational research only.** BayesPulse is not clinically validated, is not a medical device, and must not be used for diagnosis, treatment, or patient-care decisions. Do not enter real patient information.

## Features

- From-scratch classifier: empirical class priors, Gaussian likelihoods for continuous predictors, and Laplace-smoothed categorical likelihoods.
- Numerically stable inference: posterior probabilities are normalized in log space.
- Missing-value support: missing predictors are skipped rather than silently converted to zero.
- Inspectable evidence: prior log-odds, per-feature log-likelihood ratios, total log-odds, and sequential posterior updates.
- Model validation against scikit-learn `GaussianNB` under a controlled split.
- Evaluation tools for confusion matrix, precision, recall, F1, ROC-AUC, Brier score, calibration, bootstrap intervals, threshold-cost sensitivity, feature dependence, and a QDA comparator.
- React + TypeScript frontend with a FastAPI JSON API; Streamlit fallback retained.
- GitHub Actions checks for Python tests, statistical scripts, and the frontend production build.

## Dataset

Uses the **Cleveland subset of the UCI Heart Disease dataset**.

- Source: [UCI Machine Learning Repository: Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Local path: `data/cleveland.csv`
- 303 records, 13 predictors, and original target `num`
- Binary target: class 0 when `num == 0`; class 1 when `num > 0`
- The checked data contains four missing `ca` values and two missing `thal` values.

This is a small historical dataset. Its recorded target is dataset disease status, not future individual risk, and it does not represent all populations. Dataset attribution and licensing are separate from the license for this repository's source code.

## Tech stack

| Layer | Technology |
|---|---|
| Model | Python, NumPy |
| Data and evaluation | pandas, SciPy, scikit-learn, Matplotlib |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | React, TypeScript, Vite, Lucide |
| Legacy interface | Streamlit |
| Tests and CI | Python `unittest`, GitHub Actions |

## Repository structure

```text
BayesPulse/
├── app.py                         # Streamlit fallback interface
├── backend/main.py                # FastAPI API and model adapter
├── bayespulse/naive_bayes.py      # From-scratch hybrid classifier
├── data/cleveland.csv             # UCI Cleveland subset
├── frontend/
│   ├── src/App.tsx                # React workspace
│   ├── src/main.tsx               # Frontend entry point
│   ├── src/styles.css             # Responsive dark theme
│   ├── package.json
│   └── vite.config.ts
├── scripts/                       # EDA, validation, evaluation, explanations
├── tests/                         # Model, API, analysis, and Streamlit tests
├── reports/                       # Analysis and repository audit records
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- Node.js 20+ and npm
- Windows PowerShell, macOS, or Linux

## Quick start

### 1. Clone and install Python dependencies

```powershell
git clone https://github.com/BethalaRinosh/BayesPulse.git
cd BayesPulse
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

### 2. Start the API

From the repository root:

```powershell
uvicorn backend.main:app --reload
```

API base URL: `http://127.0.0.1:8000`. Interactive documentation: `http://127.0.0.1:8000/docs`.

### 3. Start the React frontend

In a second terminal, from the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

The frontend defaults to `http://127.0.0.1:8000` for its API. To change it, copy `frontend/.env.example` to `frontend/.env`, set `VITE_API_URL`, and restart Vite. The development API allows browser origins on localhost port 5173.

### Optional: Streamlit interface

From the repository root:

```powershell
streamlit run app.py
```

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Reports API health |
| `GET` | `/api/v1/metadata` | Feature order, types, descriptions, options, defaults, dataset metadata, limitations |
| `POST` | `/api/v1/predict` | Predicted dataset class, class probabilities, log-odds, feature evidence, posterior progression |

Example request:

```json
{
  "features": {
    "age": 55, "trestbps": 130, "chol": 240, "thalach": 150, "oldpeak": 1.2,
    "sex": 1, "cp": 2, "fbs": 0, "restecg": 1, "exang": 0, "slope": 2, "ca": 0, "thal": 3
  }
}
```

Feature values may be omitted or set to `null`; these are treated as missing. Unknown feature names and non-finite values are rejected. Feature evidence represents model log-likelihood ratios, not causal effects.

## Tests and analysis

Run from the repository root:

```powershell
python -m py_compile app.py backend/main.py
python -m unittest discover -s tests -v
python scripts/validate_against_sklearn.py
python scripts/explore_dataset.py
python scripts/fit_distributions.py
python scripts/evaluate_model.py
python scripts/bayesian_updating_demo.py
python scripts/feature_contributions_demo.py
python scripts/uncertainty_decision_analysis.py
```

Check the frontend TypeScript and production bundle:

```powershell
cd frontend
npm install
npm run build
```

## Evaluation snapshot

On the fixed stratified 75/25 split (seed 42; 227 training rows and 76 held-out rows), the recorded hybrid Naive Bayes results were:

| Metric | Result |
|---|---:|
| Accuracy | 0.8816 |
| Class-1 precision | 0.8250 |
| Class-1 recall | 0.9429 |
| Class-1 F1 | 0.8800 |
| ROC-AUC | 0.9742 |
| Brier score | 0.0796 |

These results are from one small held-out split, not a reliable estimate of real-world performance. The QDA comparator had slightly higher accuracy on that split, while hybrid Naive Bayes had higher recall and ROC-AUC and a lower Brier score. See the analysis reports for evaluation design, uncertainty intervals, and limitations.

## Assumptions and limitations

- Naive Bayes assumes predictors are conditionally independent given the class.
- Continuous predictors use Gaussian likelihoods even when exploratory diagnostics suggest their distributions may not be perfectly Gaussian.
- Unseen categorical values use a shared smoothed `OTHER` bucket.
- The interactive demo fits the model on all 303 records. Its probabilities are not held-out estimates and must not be interpreted as individual clinical risk.
- Bootstrap intervals are conditional on the fixed test split. Threshold costs are illustrative placeholders.
- No external validation, prospective validation, clinical threshold selection, or deployment safety evaluation has been performed.

## References

- [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn discriminant analysis](https://scikit-learn.org/stable/modules/lda_qda.html)
- [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- [SciPy Shapiro-Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
- [SciPy probability plots](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html)

## License

A source-code license has not been selected for this repository yet. Until a license is added, do not assume the project code is available for reuse or redistribution. The dataset's license is separate from the project's code license.
