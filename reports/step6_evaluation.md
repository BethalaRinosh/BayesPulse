# Step 6: held-out model evaluation

## Goal

Measure how the from-scratch hybrid Naive Bayes implementation performs on a held-out partition, including discrimination and probability calibration. This follows the Step 5 implementation-equivalence check.

## Evaluation protocol

- Source: the 303-row Cleveland subset in data/cleveland.csv.
- Target: binary class 1 when original num > 0, otherwise class 0.
- Features: all 13 predictors. Five continuous features use Gaussian likelihoods; eight discrete-coded features use categorical likelihoods with Laplace smoothing.
- Split: stratified 75/25 train/test, random seed 42. Expected partition sizes are 227 training rows and 76 held-out rows.
- Fit: the estimator is fitted only on the training partition. Held-out rows are used only for scoring.
- Metrics: accuracy, class-1 precision, class-1 recall, class-1 F1, ROC-AUC, and Brier score.
- Diagnostics: confusion matrix, ROC curve, and five-bin quantile calibration curve.

## Implementation

Added scripts/evaluate_model.py. It prints the metrics and confusion matrix, writes reports/evaluation/step6_evaluation_results.md, and generates three PNG plots in reports/evaluation/. These are generated outputs and are excluded by .gitignore.

Run from the repository root:

```powershell
git pull origin main
python scripts/evaluate_model.py
```

## User-confirmed results

The user ran `python scripts/evaluate_model.py` locally after the 11-test suite passed.

| Metric | Held-out result |
|---|---:|
| Accuracy | 0.8816 |
| Precision (class 1) | 0.8250 |
| Recall (class 1) | 0.9429 |
| F1 (class 1) | 0.8800 |
| ROC-AUC | 0.9742 |
| Brier score | 0.0798 |

Confusion matrix (rows = actual classes 0/1, columns = predicted classes 0/1):

| Actual / Predicted | 0 | 1 |
|---|---:|---:|
| 0 | 34 | 7 |
| 1 | 2 | 33 |

The five calibration bins printed by the user's run were:

| Mean predicted probability | Observed fraction positive |
|---:|---:|
| 0.0025 | 0.0000 |
| 0.0761 | 0.0000 |
| 0.6273 | 0.4667 |
| 0.9834 | 0.8667 |
| 0.9995 | 1.0000 |

## Interpretation

On this fixed split, the classifier identified 33 of 35 positive held-out rows and 34 of 41 negative rows. Its ROC-AUC indicates strong ranking on this split, but the sample is small and only one split was evaluated. In the 0.9834 mean-probability bin, the observed positive fraction was 0.8667, suggesting some overconfidence in that region. Calibration bins are noisy with 76 test cases; do not overinterpret individual bins.

The reported metrics are one-split exploratory results, not a clinical claim. Before stronger conclusions, use repeated stratified cross-validation or bootstrap intervals, compare against a simple baseline, and assess calibration more robustly. The Brier score is recorded as a probability-error metric; it has not yet been compared with a prevalence-only baseline.

## Interpretation limits

- A single held-out split is a useful first check, not a robust estimate of future performance. With only 76 held-out records, individual predictions can noticeably affect metrics.
- The calibration curve is descriptive and noisy at this sample size. The Brier score is a probability error metric, but it should be compared with a baseline before making strong claims.
- ROC-AUC describes ranking quality, not calibration or clinical usefulness.
- Naive Bayes assumes conditional independence given the class; correlated predictors can violate this assumption.
- This dataset and evaluation are educational only. They do not establish clinical validity, safe thresholds, or suitability for patient care.

## Step 6 completion gate

Step 6 runtime checks passed: the user's 11-test suite passed and the evaluation script completed. The results above are recorded. Step 7 can proceed with the sequential Bayesian updating demonstration.
