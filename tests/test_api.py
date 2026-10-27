"""Contract tests for the FastAPI adapter. Run with unittest discovery."""
import unittest

import numpy as np
from fastapi.testclient import TestClient

from backend.main import FEATURES, app, get_model, load_dataset


class TestBayesPulseAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        get_model.cache_clear()

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_metadata_preserves_canonical_feature_order(self):
        response = self.client.get("/api/v1/metadata")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual([item["name"] for item in payload["features"]], FEATURES)
        self.assertEqual(payload["dataset"]["records"], 303)
        self.assertEqual(len(payload["limitations"]), 3)

    def test_openapi_uses_real_feature_names_and_example(self):
        schema = self.client.get("/openapi.json").json()
        request_schema = schema["components"]["schemas"]["FeatureValues"]
        self.assertEqual(list(request_schema["properties"]), FEATURES)
        example = request_schema["example"]
        # JSON object key order is not semantically significant; assert the
        # complete field set while schema properties retain canonical order.
        self.assertEqual(set(example), set(FEATURES))
        self.assertEqual(example["age"], 55)
        self.assertEqual(example["thal"], 3)

    def test_unknown_feature_inside_features_is_rejected(self):
        response = self.client.post(
            "/api/v1/predict",
            json={"features": {"age": 55, "made_up_feature": 1}},
        )
        self.assertEqual(response.status_code, 422)

    def test_prediction_matches_direct_model_output(self):
        frame = load_dataset()
        row = frame[FEATURES].median(numeric_only=True).reindex(FEATURES).to_numpy(dtype=float).copy()
        for name in ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"):
            mode = frame[name].mode(dropna=True)
            row[FEATURES.index(name)] = float(mode.iloc[0])
        request_values = {name: float(row[index]) for index, name in enumerate(FEATURES)}
        response = self.client.post("/api/v1/predict", json={"features": request_values})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()

        expected = get_model().predict_proba(row.reshape(1, -1))[0]
        self.assertAlmostEqual(payload["probabilities"]["class_0"], float(expected[0]), places=12)
        self.assertAlmostEqual(payload["probabilities"]["class_1"], float(expected[1]), places=12)
        self.assertAlmostEqual(sum(payload["probabilities"].values()), 1.0, places=12)
        self.assertEqual(len(payload["feature_evidence"]), len(FEATURES))
        self.assertEqual(len(payload["updating_trace"]), len(FEATURES) + 1)

    def test_null_and_omitted_features_are_supported(self):
        response = self.client.post("/api/v1/predict", json={"features": {"age": 55, "ca": None}})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIsNone(payload["feature_evidence"][FEATURES.index("ca")]["value"])
        self.assertEqual(payload["feature_evidence"][FEATURES.index("ca")]["evidence"], "missing")
        self.assertTrue(np.isfinite(payload["probabilities"]["class_1"]))

    def test_unknown_feature_is_rejected(self):
        response = self.client.post("/api/v1/predict", json={"features": {"made_up_feature": 1}})
        self.assertEqual(response.status_code, 422)

    def test_non_finite_json_value_is_rejected_by_request_validation(self):
        response = self.client.post("/api/v1/predict", json={"features": {"age": "NaN"}})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
