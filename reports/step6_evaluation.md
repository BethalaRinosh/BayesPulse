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

## Results status

The evaluation script has been committed but has not yet been run in the user's environment. No performance numbers are claimed in this report until the command output is reviewed.

## Interpretation limits

- A single held-out split is a useful first check, not a robust estimate of future performance. With only 76 held-out records, individual predictions can noticeably affect metrics.
- The calibration curve is descriptive and noisy at this sample size. The Brier score is a probability error metric, but it should be compared with a baseline before making strong claims.
- ROC-AUC describes ranking quality, not calibration or clinical usefulness.
- Naive Bayes assumes conditional independence given the class; correlated predictors can violate this assumption.
- This dataset and evaluation are educational only. They do not establish clinical validity, safe thresholds, or suitability for patient care.

## Step 6 completion gate

Run the script locally, inspect the complete output and generated report/plots, and share the results. If any errors occur, fix them before proceeding to Step 7.
