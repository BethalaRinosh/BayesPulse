# Step 4: From-scratch Naive Bayes implementation

## Scope

Implement a transparent hybrid Naive Bayes classifier using NumPy for the core probability calculations. This step adds the estimator and unit tests; it does not train a clinical model, choose a final feature-distribution family from the exploratory Step 3 plots, or report predictive performance.

## Model design

- **Class priors:** empirical class frequencies from the training labels.
- **Continuous features:** class-conditional Gaussian density with a variance floor to prevent division by zero for constant or nearly constant columns.
- **Categorical features:** category-code likelihoods with Laplace smoothing, with categories learned from the training matrix. An explicit shared `OTHER` bucket represents unseen codes so probabilities normalize over known categories plus `OTHER`. Numeric category codes are treated as labels, not as continuous distances.
- **Posterior:** log prior plus observed feature log-likelihoods, normalized using a NumPy log-sum-exp calculation.
- **Missing predictors:** NaN values are skipped feature-by-feature, both while estimating each feature's parameters and when scoring an observation. They are not replaced with zero. A missing label is rejected.
- **Unknown categories:** map to the shared `OTHER` bucket, whose smoothing mass is included in the categorical denominator.
- **Input safeguards:** reject malformed dimensions, mismatched labels, infinities, non-integer/out-of-range/overlapping feature indices, non-finite or invalid smoothing parameters, and prediction before fitting. A continuous feature with no observed training values in one class is rejected.

## Files added

- `bayespulse/naive_bayes.py`: NumPy-only model implementation.
- `bayespulse/__init__.py`: package marker.
- `tests/test_naive_bayes.py`: unit tests for class priors, class separation on synthetic data, normalized finite posteriors, missing predictors, unseen categories, constant continuous values, invalid inputs, and lifecycle/shape errors.
- `requirements.txt`: declared minimum dependency versions for the full project.

## Run

From the repository root:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Verification status

This section is a historical snapshot from Step 4. Later user-run evidence is recorded in the Step 5–7 reports. The repository health audit introduced additional edge-case guards and tests; the latest suite must be rerun before the current code is considered verified.

## Limitations and next checks

- Continuous likelihoods are currently Gaussian only. Step 3 diagnostics are exploratory and do not automatically select a distribution.
- All model parameters must be fitted on training data only. No test-set statistics should enter model fitting or preprocessing.
- This hybrid model assumes conditional independence between features given the class. Correlated clinical measurements can violate that assumption.
- The estimator is educational and is not a medical device, diagnosis, or clinical risk calculator.
