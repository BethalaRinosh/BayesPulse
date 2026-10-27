# Step 8: interpretable feature evidence

## Goal

Explain a binary prediction as the model's prior log-odds plus one log-likelihood-ratio contribution per observed feature. This makes the Naive Bayes arithmetic inspectable without claiming that features cause disease.

## Definition

For positive class 1 and negative class 0, the prior term is:

`log P(class 1) - log P(class 0)`

For each observed feature value x_j, its contribution is:

`log P(x_j | class 1) - log P(x_j | class 0)`

The total posterior log-odds equal the prior term plus the sum of feature contributions. Positive contributions increase the model's odds for class 1; negative contributions decrease those odds. Missing features contribute zero, matching the classifier's existing missing-value policy. Categorical values not observed during fitting use the model's smoothed OTHER bucket.

## Implementation

- Added `HybridNaiveBayes.explain_log_odds(X, positive_class=None)` to `bayespulse/naive_bayes.py`.
- The API returns the positive/negative class labels, modelled feature indices, prior log-odds, a contribution matrix, and total log-odds for each row.
- The method requires a fitted binary classifier, supports selecting either fitted class as the positive class, and validates class selection and feature count.
- Added `scripts/feature_contributions_demo.py`, which uses the same deterministic stratified split as the evaluation script and prints the held-out sample's per-feature contributions.
- Added tests for reconstruction of posterior log-odds, missing values, reversing the selected positive class, and invalid model/class configurations.

## Run

From the repository root:

```powershell
git pull origin main
python -m unittest discover -s tests -v
python scripts/feature_contributions_demo.py
```

## Verification status

**Pending local verification.** The last user-confirmed run passed the previous 16-test suite and all existing scripts before Step 8 was added. The new suite is expected to contain 20 tests. Do not mark Step 8 complete until the new suite passes and the demo output is reviewed.

## Interpretation limits

- These contributions are log-likelihood ratios under the fitted Naive Bayes model, not causal effects.
- A positive contribution means that the observed value is more likely under the selected positive class than under the other class according to the model's fitted likelihoods. It does not mean the feature medically causes or independently predicts disease.
- The feature-independence assumption remains in force; correlated features can make contributions appear more decisive than the underlying evidence warrants.
- The dataset and output remain educational, not clinically validated or appropriate for patient-care decisions.
