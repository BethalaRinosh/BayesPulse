"""Unit tests for Step 9 statistical-analysis helpers."""
import unittest

import numpy as np
import pandas as pd

from scripts.uncertainty_decision_analysis import bootstrap_intervals, cramers_v


class TestStep9AnalysisHelpers(unittest.TestCase):
    def test_bootstrap_intervals_are_finite_and_repeatable(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.2, 0.8, 0.6, 0.9])
        first = bootstrap_intervals(y, probabilities, replicates=100, seed=7)
        second = bootstrap_intervals(y, probabilities, replicates=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {"Accuracy", "Recall (class 1)", "ROC-AUC", "Brier score"},
        )
        for median, low, high in first.values():
            self.assertTrue(np.isfinite([median, low, high]).all())
            self.assertLessEqual(low, median)
            self.assertLessEqual(median, high)

    def test_bootstrap_requires_both_classes(self):
        with self.assertRaisesRegex(ValueError, "both classes"):
            bootstrap_intervals(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]), replicates=10)

    def test_cramers_v_is_one_for_identical_nonconstant_columns(self):
        values = pd.Series([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(cramers_v(values, values), 1.0)

    def test_cramers_v_is_undefined_for_constant_column(self):
        left = pd.Series([1, 1, 1, 1])
        right = pd.Series([0, 1, 0, 1])
        self.assertTrue(np.isnan(cramers_v(left, right)))


if __name__ == "__main__":
    unittest.main()
