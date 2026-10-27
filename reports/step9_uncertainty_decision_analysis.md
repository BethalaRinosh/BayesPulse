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
- **CI:** the GitHub Actions workflow runs the Step 9 analysis script.

## Verification

**Passed locally on 2026-10-28.**

Commands run:

```powershell
python -m unittest discover -s tests -v
python scripts/validate_against_sklearn.py
python scripts/uncertainty_decision_analysis.py
```

Results:
- Unit tests: **24 passed**, zero failures/errors, in 0.464 seconds.
- GaussianNB implementation comparison: **PASS**. Class priors match; maximum fitted mean/variance difference is `2.842e-13`; maximum held-out probability difference is `3.109e-15`; held-out predictions match.
- Analysis completed on 227 training rows and 76 held-out rows, using 2,000 bootstrap replicates.
- Hybrid Naive Bayes bootstrap percentile 95% intervals:
  - Accuracy: median `0.8816`, interval `[0.8026, 0.9474]`.
  - Class-1 recall: median `0.9429`, interval `[0.8571, 1.0000]`.
  - ROC-AUC: median `0.9763`, interval `[0.9394, 0.9951]`.
  - Brier score: median `0.0776`, interval `[0.0393, 0.1268]`.
- Held-out point metrics at threshold 0.5:
  - Hybrid Naive Bayes: accuracy `0.8816`, class-1 recall `0.9429`, ROC-AUC `0.9742`, Brier score `0.0796`.
  - QDA comparator: accuracy `0.8947`, class-1 recall `0.9143`, ROC-AUC `0.9519`, Brier score `0.0949`.
- GitHub Actions completed successfully for the Step 9 commit series, including commit `9d55ca5fd16f6ff77a56fc5e5c90a85165ae7b58`.

The comparison is mixed rather than a clean victory: QDA had slightly higher accuracy at the 0.5 threshold, while hybrid Naive Bayes had higher recall and ROC-AUC and lower Brier score on this split.

## Interpretation limits

- Bootstrap intervals resample the 76 held-out rows while preserving their class counts. They describe sampling variation conditional on this particular split, not the full uncertainty from refitting on new data, collecting a new dataset, or population shift.
- A threshold's error-cost score depends entirely on the hypothetical weights and this held-out sample. No threshold is recommended for clinical use.
- QDA and hybrid Naive Bayes make different assumptions; a single split cannot establish superiority or reliable generalization.
- Cramér's V, Spearman correlation, and mutual information describe statistical dependence, not causation. Estimates are sensitive to sample size, encoding, and imputation.
- Dependence diagnostics describe a potential weakness of the Naive Bayes independence assumption; they do not remove it.
- BayesPulse remains an educational project, not a validated medical device or patient-care tool.
