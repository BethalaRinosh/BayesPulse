# Step 3: Per-class distribution fitting and diagnostics

## Purpose

This step examines the candidate likelihood distributions that will later be used by the from-scratch Naive Bayes model. It is exploratory: it does not train or evaluate the final classifier and does not automatically choose a distribution based only on a statistical test.

## Implementation added

- `scripts/fit_distributions.py` loads `data/cleveland.csv` and creates separate continuous-feature diagnostics for binary classes 0 and 1.
- For each continuous feature and class, it records sample size, Gaussian mean and sample standard deviation, Shapiro-Wilk W statistic and p-value, and whether a log-normal candidate is mathematically valid (all observed values must be strictly positive).
- It writes a histogram with fitted Gaussian density and a KDE curve when fitting is numerically possible, alongside a normal Q-Q plot.
- It writes categorical likelihoods using Laplace smoothing:
  `P(feature=v | class=c) = (count(v,c) + 1) / (nonmissing_count(c) + K)`,
  where K is the number of observed categories for that feature. Missing feature values are excluded from that feature's count denominator rather than treated as an extra clinical category.
- Outputs are written to `reports/distribution_fitting/`.

## Run

From the repository root, install the relevant dependencies and run:

```bash
pip install numpy pandas scipy matplotlib
python scripts/fit_distributions.py
```

Expected outputs:

- `continuous_fit_diagnostics.csv`
- `categorical_laplace_probabilities.csv`
- One `*_distribution_diagnostics.png` image per continuous feature

## How to interpret the tests

- The Shapiro-Wilk null hypothesis is that a sample was drawn from a normal distribution. A small p-value (for example, below 0.05) is evidence against normality; it is not proof that a particular alternative distribution is better.
- Q-Q plots and histograms must be considered alongside the p-value. With enough observations, small deviations can become statistically detectable; with small class-specific samples, tests can have limited power.
- A log-normal candidate is not fitted to features containing zero or negative observations. In this dataset, that rule makes it unavailable for `oldpeak` because its observed minimum is zero.
- KDE is a flexible exploratory density estimate, but bandwidth choice affects its shape. It is not automatically superior to a Gaussian fit and should be judged with the downstream training-only validation process.
- These diagnostics are descriptive on the full dataset. They must not be used to claim held-out model performance. In the modelling/evaluation steps, every distribution parameter must be fitted using the training partition only.

## Current verification status

The script was reviewed statically and its required inputs/outputs were checked against the repository structure. It has **not yet been executed end-to-end in this environment**, so no Shapiro-Wilk p-values, fitted plots, or categorical probability tables are claimed as generated results. Run it locally to produce the measured diagnostics before deciding which continuous likelihoods to use in the classifier.

## Sources & credits

- SciPy `shapiro` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html. The test's null hypothesis is normality; the p-value is evidence, not a guarantee of fit. SciPy is distributed under its BSD-style license; this project calls the library API and does not copy its implementation.
- SciPy `probplot` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html. Used as an API reference for normal Q-Q plots; no example code copied.
- pandas KDE documentation: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.kde.html. Reference for the interpretation and bandwidth dependence of kernel density estimates; the script uses SciPy's `gaussian_kde` API.
