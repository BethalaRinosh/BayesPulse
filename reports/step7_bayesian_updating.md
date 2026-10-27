# Step 7: feature-by-feature Bayesian updating

## Goal

Make the BayesPulse posterior update process observable. Starting from class priors, the demonstration adds one feature's likelihood at a time and shows the resulting normalized class probabilities.

## Implementation

- Added `HybridNaiveBayes.predict_proba_progression(X)` in `bayespulse/naive_bayes.py`.
- The method returns posterior snapshots with shape `(n_samples, n_features_modelled + 1, n_classes)`.
- Snapshot zero contains the class priors before observing any features.
- Each following snapshot adds the likelihood for one feature, in the documented order: continuous features first, then categorical features.
- Missing values contribute no likelihood, so the posterior stays unchanged at that feature step.
- Added `scripts/bayesian_updating_demo.py`, which trains on the same deterministic stratified training split used in Step 6 and traces one held-out sample. The sample's true label is shown only after the posterior trace for retrospective context; it is not used for fitting.
- Added tests for progression shape, prior initialization, missing-feature behavior, agreement between the final sequential posterior and `predict_proba`, and prediction-before-fit rejection.

## Run

From the repository root:

```powershell
git pull origin main
python -m unittest discover -s tests -v
python scripts/bayesian_updating_demo.py
```

## Interpretation

The demonstration is a visualization aid for the Naive Bayes factorization. It shows how each assumed conditionally independent feature changes the posterior under the model. It does not prove that the independence assumption is true, that each feature causally affects the outcome, or that the output is clinically valid.

## User-confirmed results after repository health fixes

The user pulled the audit fixes and ran the full verification sequence. All **16 unit tests passed**, and the Bayesian updating demo completed without an error.

For the deterministic held-out sample:

- Dataset row index: 256
- Split: 227 training rows and 76 held-out rows, stratified with random state 42
- Prior: P(class 0) = 0.541850; P(class 1) = 0.458150
- Posterior after all 13 features: P(class 0) = 0.838254; P(class 1) = 0.161746
- Predicted class: 0
- Retrospective dataset label: 0

The posterior shifts as each feature likelihood is incorporated. This illustrates the model's assumptions, not causal influence or clinical validity. The sample is a single held-out record and does not establish generalization.
