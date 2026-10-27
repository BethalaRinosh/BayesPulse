# BayesPulse
## Medical Risk Prediction Using Probabilistic Feature Modelling

BayesPulse is an educational micro-project for Probability and Statistics for AI & ML. It will explore probabilistic feature modelling and a transparent Naive Bayes implementation for a binary heart-disease classification task.

> **Educational disclaimer:** BayesPulse is a student project. Its output is not a medical diagnosis, treatment recommendation, or clinically validated estimate of an individual's health risk.

## Project status

- **Current stage:** Step 1 — repository reconnaissance and source review.
- **Dataset selected:** UCI Heart Disease, Cleveland subset.
- **Implementation status:** No model code has been written yet.
- **Repository audit:** The repository was empty when inspected; this README establishes the initial project record. No existing project files were overwritten or removed.

## Planned approach

1. Acquire and document the UCI Cleveland heart-disease dataset and target definition.
2. Explore data quality, class balance, feature types, and summary statistics.
3. Fit and diagnose per-class feature distributions. Assess Gaussian assumptions with histograms, Q-Q plots, and Shapiro-Wilk tests where appropriate; consider log-normal or KDE alternatives when justified. Use Bernoulli/categorical probabilities with Laplace smoothing for discrete features.
4. Implement Naive Bayes from scratch in NumPy using Bayes' theorem and log-space calculations.
5. Validate the implementation against scikit-learn's GaussianNB where assumptions align, then audit code and dependencies.
6. Evaluate on a proper held-out split or cross-validation using confusion matrix, precision, recall, F1, ROC-AUC, and a calibration curve.
7. Add a feature-by-feature Bayesian updating demonstration.
8. Add interpretable per-feature log-likelihood ratios.
9. Add bootstrap uncertainty intervals, cost-sensitive thresholds, and feature-dependence/model comparison analysis where feasible.
10. Build a Streamlit demo with input validation, risk estimates, contribution breakdowns, distribution context, and a visible disclaimer.
11. Run end-to-end checks and prepare reproducible setup instructions and viva notes.

The project will be implemented one step at a time. Core probability calculations, distribution fitting logic, and Bayesian updating will be written in readable, commented code rather than copied from an existing implementation.

## Dataset

**UCI Machine Learning Repository: Heart Disease**  
https://archive.ics.uci.edu/dataset/45/heart+disease

The UCI listing describes 303 Cleveland instances and 13 commonly used predictors. The original target `num` has values 0–4; this project plans to define the binary target as 0 = no recorded heart disease and 1 = recorded heart disease (original values 1–4). The exact selected file, missing-value handling, and class counts will be recorded during dataset acquisition.

Dataset license: **CC BY 4.0**. Attribution will be retained in the project documentation.

## Sources & Credits

These are planned reference sources for the implementation. No third-party project code has been copied into this repository at Step 1.

| Source | Planned use | License / attribution notes |
|---|---|---|
| UCI Heart Disease dataset — https://archive.ics.uci.edu/dataset/45/heart+disease | Dataset and dataset documentation | CC BY 4.0; cite the UCI dataset and its creators. |
| SciPy Shapiro documentation — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html | Reference for normality-test behavior | Official SciPy documentation; SciPy is distributed under a modified BSD license. We will call the library API, not copy implementation code. |
| scikit-learn Naive Bayes guide — https://scikit-learn.org/stable/modules/naive_bayes.html | Independent sanity-check reference for GaussianNB and Naive Bayes assumptions | scikit-learn is BSD-3-Clause licensed. Used as documentation/reference only; the core model will be our own NumPy implementation. |
| scikit-learn calibration guide — https://scikit-learn.org/stable/modules/calibration.html | Understand and validate reliability diagrams | scikit-learn is BSD-3-Clause licensed. Prefer library APIs for evaluation plumbing rather than copying example code. |
| Streamlit app tutorial — https://docs.streamlit.io/get-started/tutorials/create-an-app | Reference for basic app layout and interaction | Official documentation; use the documented API, not copied tutorial source. |
| NumPy license — https://github.com/numpy/numpy/blob/main/LICENSE.txt | Reference for numerical-computing dependency | BSD-3-Clause; dependencies retain their own licenses. |
| Matplotlib license — https://matplotlib.org/stable/project/license.html | Reference for plotting dependency | PSF/BSD-compatible licensing as described by the project. |

### Reviewed existing implementation

- Example repository: https://github.com/MastersAbh/Heart-Disease-Prediction-using-Naive-Bayes-Classifier
- Its README describes a from-scratch Naive Bayes implementation and a comparison with scikit-learn GaussianNB.
- A LICENSE file was not found at the repository's root during this review, so its reuse permissions are **unclear**. **No code from this repository will be copied.** It may only serve as a high-level topic reference unless a compatible license is confirmed.

## Planned environment

Python with NumPy, pandas, SciPy, matplotlib, scikit-learn (validation and evaluation only), and Streamlit. The exact dependency pins and installation instructions will be added after the implementation environment is tested.

## Run instructions

There is no runnable application yet. Setup and run commands will be added as the project becomes executable; do not treat this initial README as a working install guide.

## Decisions and open questions

- Dataset choice confirmed: UCI Heart Disease, Cleveland subset.
- Existing repository files: none were present at inspection time.
- No model architecture, distribution choice, evaluation result, or performance metric has been claimed as implemented or measured yet.
