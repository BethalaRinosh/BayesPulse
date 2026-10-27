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

## Historical results from the pre-audit implementation

The user ran the commands above locally after pulling commit `6d37b44`. All **13 unit tests passed** (`Ran 13 tests in 0.047s; OK`), and the demo completed without a traceback. These results predate the repository health fixes. The updated tests and demo must be rerun because the categorical smoothing logic has changed.

For the deterministic held-out sample:

- Dataset row index: 256
- Split: 227 training rows and 76 held-out rows, stratified with random state 42
- Prior: P(class 0) = 0.541850; P(class 1) = 0.458150
- Final posterior after all 13 features: P(class 0) = 0.836706; P(class 1) = 0.163294
- Predicted class: 0
- Retrospective dataset label: 0

The posterior changes as each feature likelihood is incorporated. This is expected behavior for a Naive Bayes model and is not evidence that the features causally change risk. The displayed row is a single example; it does not establish clinical validity or generalization. The numeric trace above is historical and should be replaced with the next user-verified run.
