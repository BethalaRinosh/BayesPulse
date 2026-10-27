# Step 9: uncertainty, decision costs, dependence, and model comparison

## Goal

Extend the single-split evaluation with uncertainty intervals, an explicit error-cost sensitivity analysis, diagnostics for dependence that challenges Naive Bayes' conditional-independence assumption, and a full-covariance Gaussian discriminant comparator.

## Implementation

Added `scripts/uncertainty_decision_analysis.py`. It reuses the deterministic stratified 75/25 split (seed 42) and fits models/preprocessing on the training partition only.

- **Bootstrap intervals:** 2,000 class-stratified bootstrap resamples of held-out predictions. Percentile 95% intervals and medians are reported for accuracy, class-1 recall, ROC-AUC, and Brier score.
- **Decision-cost sensitivity:** compares thresholds from 0.1 through 0.9 with illustrative false-negative cost 5 and false-positive cost 1. These are explicit placeholders, not medical or economic recommendations.
- **Dependence diagnostics:** absolute Spearman correlation for continuous-continuous pairs, Cramér's V for categorical-categorical pairs, and feature-target mutual information estimated on the training partition.
- **Model comparison:** compares the from-scratch hybrid Naive Bayes model against scikit-learn Quadratic Discriminant Analysis (QDA), a full-covariance Gaussian discriminant comparator. QDA uses training-fitted median imputation for continuous variables, most-frequent imputation plus one-hot encoding for categorical variables, and covariance regularization (`reg_param=0.1`).
- **Report:** writes generated results to `reports/evaluation/step9_uncertainty_decision_analysis.md`; generated outputs remain ignored by Git.
- **Tests:** added four helper tests covering deterministic finite bootstrap intervals, missing-class input rejection, Cramér's V for identical nonconstant columns, and undefined Cramér's V for a constant column.
- **CI:** added the Step 9 analysis script to `.github/workflows/verify.yml`.

## Commands

Run from the repository root:

```powershell
git pull origin main
python -m unittest discover -s tests -v
python scripts/uncertainty_decision_analysis.py
```

The full existing verification gate should also be run before accepting this step.

## Verification status

**Pending local verification.** The script and tests have been pushed, but no local result for the Step 9 additions has been confirmed yet. The expected test count is 24 (20 previous model tests plus four Step 9 helper tests). Do not mark this step complete until the user runs the updated test suite and analysis script, shares the output, and any failures or numerical warnings are reviewed. Check the GitHub Actions result for the exact latest commit as a separate verification signal.

## Interpretation limits

- Bootstrap intervals resample the 76 held-out rows while preserving their class counts. They describe sampling variation conditional on this particular split, not the full uncertainty from refitting on new data, collecting a new dataset, or population shift.
- A threshold's error-cost score depends entirely on the hypothetical weights and this held-out sample. No threshold is recommended for clinical use.
- QDA and hybrid Naive Bayes make different assumptions; a single split cannot establish superiority or reliable generalization.
- Cramér's V, Spearman correlation, and mutual information describe statistical dependence, not causation. Estimates are sensitive to sample size, encoding, and imputation.
- Dependence diagnostics describe a potential weakness of the Naive Bayes independence assumption; they do not remove it.
- BayesPulse remains an educational project, not a validated medical device or patient-care tool.
