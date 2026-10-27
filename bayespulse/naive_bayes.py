"""A small, transparent hybrid Naive Bayes classifier implemented with NumPy.

Continuous features use class-conditional Gaussian likelihoods. Discrete-coded
features use categorical likelihoods with Laplace smoothing. Missing feature
values are skipped feature-by-feature at prediction time; they are never
silently converted to zero. Fit all parameters on training data only.

This is educational code, not a clinical decision tool.
"""
from __future__ import annotations

import numpy as np


class HybridNaiveBayes:
    """Naive Bayes with Gaussian numeric and smoothed categorical likelihoods.

    Parameters
    ----------
    continuous_features:
        Column indices treated as continuous and modelled with Gaussian PDFs.
    categorical_features:
        Column indices treated as category codes, not continuous quantities.
    alpha:
        Laplace smoothing strength for categorical likelihoods. Must be > 0.
    variance_smoothing:
        Relative floor added to continuous-feature variances to avoid division
        by zero for constant or nearly constant training features.
    """

    def __init__(
        self,
        continuous_features: tuple[int, ...],
        categorical_features: tuple[int, ...],
        alpha: float = 1.0,
        variance_smoothing: float = 1e-9,
    ) -> None:
        self.continuous_features = tuple(continuous_features)
        self.categorical_features = tuple(categorical_features)
        self.alpha = float(alpha)
        self.variance_smoothing = float(variance_smoothing)
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than zero.")
        if self.variance_smoothing < 0:
            raise ValueError("variance_smoothing must be non-negative.")
        if set(self.continuous_features) & set(self.categorical_features):
            raise ValueError("A feature cannot be both continuous and categorical.")
        self._fitted = False

    @staticmethod
    def _as_matrix(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature.")
        if np.isinf(values).any():
            raise ValueError("X may contain NaN for missing values, but not infinity.")
        return values

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridNaiveBayes":
        """Estimate priors and per-class likelihood parameters from training data."""
        X = self._as_matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be a 1D array with one label per row.")
        if pd_is_missing_label(y):
            raise ValueError("y cannot contain missing labels.")
        n_features = X.shape[1]
        all_indices = self.continuous_features + self.categorical_features
        if not all_indices:
            raise ValueError("At least one modelled feature is required.")
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("Feature indices must not be repeated.")
        if any(i < 0 or i >= n_features for i in all_indices):
            raise ValueError(f"Feature indices must be in [0, {n_features - 1}].")
        if any(np.all(np.isnan(X[:, i])) for i in all_indices):
            raise ValueError("A modelled feature cannot be entirely missing in training data.")

        self.classes_, class_counts = np.unique(y, return_counts=True)
        if self.classes_.size < 2:
            raise ValueError("At least two distinct classes are required for classification.")
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())
        self.n_features_in_ = n_features
        self.gaussian_params_: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.category_values_: dict[int, np.ndarray] = {}
        self.category_log_probs_: dict[int, np.ndarray] = {}
        self.category_unknown_log_probs_: dict[int, np.ndarray] = {}

        for class_pos, class_value in enumerate(self.classes_):
            class_X = X[y == class_value]
            means = np.full(n_features, np.nan, dtype=float)
            variances = np.full(n_features, np.nan, dtype=float)
            for j in self.continuous_features:
                observed = class_X[:, j]
                observed = observed[np.isfinite(observed)]
                if observed.size == 0:
                    continue
                means[j] = np.mean(observed)
                raw_var = np.var(observed, ddof=0)
                global_observed = X[:, j]
                global_observed = global_observed[np.isfinite(global_observed)]
                scale = float(np.var(global_observed, ddof=0)) if global_observed.size else 0.0
                floor = self.variance_smoothing * max(scale, 1.0)
                variances[j] = max(float(raw_var), floor, np.finfo(float).tiny)
            self.gaussian_params_[class_pos] = (means, variances)

        for j in self.categorical_features:
            observed_all = X[:, j]
            observed_all = observed_all[np.isfinite(observed_all)]
            categories = np.unique(observed_all)
            if categories.size == 0:
                raise ValueError(f"Categorical feature {j} has no observed training values.")
            self.category_values_[j] = categories
            log_probs = np.empty((len(self.classes_), len(categories)), dtype=float)
            unknown_log_probs = np.empty(len(self.classes_), dtype=float)
            for class_pos, class_value in enumerate(self.classes_):
                values = X[y == class_value, j]
                values = values[np.isfinite(values)]
                denominator = values.size + self.alpha * categories.size
                for category_pos, category in enumerate(categories):
                    count = np.count_nonzero(values == category)
                    log_probs[class_pos, category_pos] = np.log(
                        (count + self.alpha) / denominator
                    )
                unknown_log_probs[class_pos] = np.log(self.alpha / denominator)
            self.category_log_probs_[j] = log_probs
            self.category_unknown_log_probs_[j] = unknown_log_probs

        self._fitted = True
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalized log posterior probabilities, one row per sample."""
        if not self._fitted:
            raise RuntimeError("Call fit before predict_log_proba.")
        X = self._as_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; model was fitted with {self.n_features_in_}."
            )
        n_samples = X.shape[0]
        joint = np.tile(self.class_log_prior_, (n_samples, 1))

        for class_pos in range(len(self.classes_)):
            means, variances = self.gaussian_params_[class_pos]
            for j in self.continuous_features:
                observed_mask = np.isfinite(X[:, j])
                if not np.isfinite(means[j]):
                    continue
                values = X[observed_mask, j]
                variance = variances[j]
                log_pdf = -0.5 * (
                    np.log(2.0 * np.pi * variance) + ((values - means[j]) ** 2 / variance)
                )
                joint[observed_mask, class_pos] += log_pdf

            for j in self.categorical_features:
                values = X[:, j]
                observed_mask = np.isfinite(values)
                if not observed_mask.any():
                    continue
                categories = self.category_values_[j]
                for row_index in np.flatnonzero(observed_mask):
                    matches = np.flatnonzero(categories == values[row_index])
                    if matches.size:
                        log_likelihood = self.category_log_probs_[j][class_pos, matches[0]]
                    else:
                        log_likelihood = self.category_unknown_log_probs_[j][class_pos]
                    joint[row_index, class_pos] += log_likelihood

        # Stable log-sum-exp normalization without SciPy.
        row_max = np.max(joint, axis=1, keepdims=True)
        log_normalizer = row_max + np.log(np.sum(np.exp(joint - row_max), axis=1, keepdims=True))
        return joint - log_normalizer

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probabilities in the order of classes_."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class with the largest posterior probability."""
        if not self._fitted:
            raise RuntimeError("Call fit before predict.")
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


def pd_is_missing_label(y: np.ndarray) -> bool:
    """NumPy-only missing-label check that also works with object/string labels."""
    try:
        missing = np.asarray(y != y)
        if missing.any():
            return True
    except (TypeError, ValueError):
        pass
    if y.dtype.kind in "fc":
        return bool(np.isnan(y).any())
    return any(value is None for value in y)
