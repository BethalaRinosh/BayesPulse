"""Unit tests for the NumPy-only BayesPulse classifier.

Run from repository root:
    python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from bayespulse.naive_bayes import HybridNaiveBayes


class TestHybridNaiveBayes(unittest.TestCase):
    def setUp(self):
        # Feature 0 is continuous; feature 1 is a categorical code.
        self.X = np.array([
            [1.0, 0.0], [1.2, 0.0], [0.8, 1.0], [1.1, 0.0],
            [5.0, 1.0], [5.2, 1.0], [4.8, 0.0], [5.1, 1.0],
        ])
        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.model = HybridNaiveBayes((0,), (1,)).fit(self.X, self.y)

    def test_priors_sum_to_one(self):
        self.assertAlmostEqual(float(np.exp(self.model.class_log_prior_).sum()), 1.0)

    def test_predicts_separated_synthetic_classes(self):
        predicted = self.model.predict(np.array([[1.0, 0.0], [5.0, 1.0]]))
        np.testing.assert_array_equal(predicted, np.array([0, 1]))

    def test_log_odds_contributions_reconstruct_posterior(self):
        samples = np.array([[1.0, 0.0], [5.0, 1.0], [np.nan, np.nan]])
        explanation = self.model.explain_log_odds(samples)
        contributions = explanation["feature_log_likelihood_ratios"]
        total_log_odds = explanation["total_log_odds"]
        self.assertEqual(contributions.shape, (3, 2))
        np.testing.assert_allclose(
            total_log_odds,
            explanation["prior_log_odds"] + contributions.sum(axis=1),
        )
        expected_log_odds = (
            np.log(self.model.predict_proba(samples)[:, 1])
            - np.log(self.model.predict_proba(samples)[:, 0])
        )
        np.testing.assert_allclose(total_log_odds, expected_log_odds, atol=1e-12)

    def test_missing_features_have_zero_log_likelihood_contribution(self):
        explanation = self.model.explain_log_odds(np.array([[np.nan, 0.0]]))
        contributions = explanation["feature_log_likelihood_ratios"]
        self.assertEqual(contributions[0, 0], 0.0)

        all_missing = self.model.explain_log_odds(np.array([[np.nan, np.nan]]))
        np.testing.assert_allclose(
            all_missing["feature_log_likelihood_ratios"], [[0.0, 0.0]]
        )
        np.testing.assert_allclose(
            all_missing["total_log_odds"], [all_missing["prior_log_odds"]]
        )

    def test_log_odds_explanation_can_reverse_positive_class(self):
        samples = np.array([[1.0, 0.0], [5.0, 1.0]])
        default = self.model.explain_log_odds(samples)
        reversed_explanation = self.model.explain_log_odds(samples, positive_class=0)
        np.testing.assert_allclose(
            reversed_explanation["feature_log_likelihood_ratios"],
            -default["feature_log_likelihood_ratios"],
        )
        np.testing.assert_allclose(
            reversed_explanation["total_log_odds"], -default["total_log_odds"]
        )

    def test_log_odds_explanation_rejects_invalid_class_configuration(self):
        with self.assertRaisesRegex(RuntimeError, "Call fit before explain_log_odds"):
            HybridNaiveBayes((0,), (1,)).explain_log_odds(self.X)
        with self.assertRaisesRegex(ValueError, "not a fitted class"):
            self.model.explain_log_odds(self.X, positive_class=9)
        three_class_model = HybridNaiveBayes((0,), (1,)).fit(
            np.array([[1.0, 0.0], [3.0, 1.0], [5.0, 0.0]]),
            np.array([0, 1, 2]),
        )
        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            three_class_model.explain_log_odds(np.array([[2.0, 0.0]]))

    def test_posterior_progression_ends_at_predict_proba(self):
        samples = np.array([[1.0, 0.0], [5.0, 1.0], [np.nan, np.nan]])
        progression = self.model.predict_proba_progression(samples)
        self.assertEqual(progression.shape, (3, 3, 2))
        np.testing.assert_allclose(progression[:, -1, :], self.model.predict_proba(samples))
        np.testing.assert_allclose(progression[:, 0, :], np.tile([0.5, 0.5], (3, 1)))
        np.testing.assert_allclose(progression[2, :, :], np.tile([0.5, 0.5], (3, 1)))

    def test_progression_before_fit_is_rejected(self):
        model = HybridNaiveBayes((0,), (1,))
        with self.assertRaisesRegex(RuntimeError, "Call fit before predict_proba_progression"):
            model.predict_proba_progression(self.X)

    def test_posteriors_are_normalized_and_finite(self):
        probabilities = self.model.predict_proba(self.X)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(self.X)))

    def test_missing_features_are_skipped(self):
        probabilities = self.model.predict_proba(np.array([[np.nan, np.nan]]))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities, [[0.5, 0.5]])

    def test_unseen_category_is_smoothed(self):
        probabilities = self.model.predict_proba(np.array([[1.0, 99.0]]))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_categorical_probabilities_include_unknown_bucket_and_sum_to_one(self):
        known = np.exp(self.model.category_log_probs_[1])
        unknown = np.exp(self.model.category_unknown_log_probs_[1])
        np.testing.assert_allclose(known.sum(axis=1) + unknown, np.ones(len(self.model.classes_)))

    def test_continuous_feature_missing_for_one_class_is_rejected(self):
        X = np.array([
            [np.nan, 0.0], [np.nan, 1.0],
            [3.0, 0.0], [4.0, 1.0],
        ])
        y = np.array([0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "no observed values in class"):
            HybridNaiveBayes((0,), (1,)).fit(X, y)

    def test_constant_continuous_feature_does_not_crash(self):
        X = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 1.0]])
        model = HybridNaiveBayes((0,), (1,)).fit(X, np.array([0, 0, 1, 1]))
        self.assertTrue(np.isfinite(model.predict_proba(X)).all())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0,), (0,))
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0,), (1,), alpha=0)
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0,), (1,), alpha=np.nan)
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0,), (1,), alpha=np.inf)
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0,), (1,), variance_smoothing=np.nan)
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0,), (1,), variance_smoothing=np.inf)
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0.5,), (1,))
        with self.assertRaises(ValueError):
            HybridNaiveBayes((True,), (1,))
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0,), (1,)).fit(np.array([[1.0, np.inf]]), np.array([0]))

    def test_single_class_training_data_is_rejected(self):
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0,), (1,)).fit(np.array([[1.0, 0.0], [2.0, 1.0]]), np.array([0, 0]))

    def test_missing_target_label_is_rejected(self):
        with self.assertRaises(ValueError):
            HybridNaiveBayes((0,), (1,)).fit(np.array([[1.0, 0.0], [2.0, 1.0]]), np.array([0.0, np.nan]))

    def test_predict_before_fit_is_rejected(self):
        model = HybridNaiveBayes((0,), (1,))
        with self.assertRaisesRegex(RuntimeError, "Call fit before predict"):
            model.predict(self.X)

    def test_failed_refit_does_not_leave_model_marked_fitted(self):
        with self.assertRaises(ValueError):
            self.model.fit(self.X, np.zeros(len(self.y), dtype=int))
        with self.assertRaisesRegex(RuntimeError, "Call fit before predict"):
            self.model.predict(self.X)

    def test_wrong_feature_count_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict(np.array([[1.0]]))


if __name__ == "__main__":
    unittest.main()
