# Step 4: From-scratch Naive Bayes implementation

## Scope

Implement a transparent hybrid Naive Bayes classifier using NumPy for the core probability calculations. This step adds the estimator and unit tests; it does not train a clinical model, choose a final feature-distribution family from the exploratory Step 3 plots, or report predictive performance.

## Model design

- **Class priors:** empirical class frequencies from the training labels.
- **Continuous features:** class-conditional Gaussian density with a variance floor to prevent division by zero for constant or nearly constant columns.
- **Categorical features:** category-code likelihoods with Laplace smoothing, with categories learned from the training matrix. Numeric category codes are treated as labels, not as continuous distances.
- **Posterior:** log prior plus observed feature log-likelihoods, normalized using a NumPy log-sum-exp calculation.
- **Missing predictors:** NaN values are skipped feature-by-feature, both while estimating each feature's parameters and when scoring an observation. They are not replaced with zero. A missing label is rejected.
- **Unknown categories:** receive the smoothed fallback probability.
- **Input safeguards:** reject malformed dimensions, mismatched labels, infinities, invalid or overlapping feature indices, non-positive smoothing, and prediction before fitting.

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

The implementation and tests have been reviewed statically, but they have **not yet been executed in a clean clone** in this environment. No passing test result is claimed. Run the command above and inspect the output before relying on the implementation. Step 5 is reserved for comparison against scikit-learn GaussianNB and the planned broader validation/audit; it has not been started.

## Limitations and next checks

- Continuous likelihoods are currently Gaussian only. Step 3 diagnostics are exploratory and do not automatically select a distribution.
- All model parameters must be fitted on training data only. No test-set statistics should enter model fitting or preprocessing.
- This hybrid model assumes conditional independence between features given the class. Correlated clinical measurements can violate that assumption.
- The estimator is educational and is not a medical device, diagnosis, or clinical risk calculator.
