"""Smoke tests for the Streamlit app's data/model integration."""
import unittest

import numpy as np

import app


class TestStreamlitAppIntegration(unittest.TestCase):
    def test_demo_model_returns_consistent_posterior_and_evidence(self):
        frame = app.load_dataset()
        self.assertEqual(frame.shape, (303, 14))

        model = app.train_demo_model()
        row = frame[app.FEATURES].median(numeric_only=True).reindex(app.FEATURES).to_numpy(dtype=float).reshape(1, -1)
        # Categorical medians may be non-observed codes; replace them with observed modes.
        for feature in app.CATEGORICAL:
            row[0, app.FEATURES.index(feature)] = float(frame[feature].mode(dropna=True).iloc[0])

        posterior = model.predict_proba(row)
        explanation = model.explain_log_odds(row, positive_class=1)
        contributions = explanation["feature_log_likelihood_ratios"]
        progression = model.predict_proba_progression(row)

        self.assertEqual(posterior.shape, (1, 2))
        self.assertTrue(np.isfinite(posterior).all())
        self.assertAlmostEqual(float(posterior.sum()), 1.0, places=12)
        self.assertEqual(contributions.shape, (1, len(app.FEATURES)))
        self.assertEqual(progression.shape, (1, len(app.FEATURES) + 1, 2))
        reconstructed = float(explanation["prior_log_odds"] + contributions.sum())
        self.assertAlmostEqual(reconstructed, float(explanation["total_log_odds"][0]), places=10)


if __name__ == "__main__":
    unittest.main()
