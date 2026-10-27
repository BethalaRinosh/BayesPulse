"""FastAPI adapter for the existing BayesPulse model.

The model implementation remains in bayespulse.naive_bayes. This module only
translates JSON requests into the model's established NumPy API.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from bayespulse.naive_bayes import HybridNaiveBayes

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleveland.csv"

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = CONTINUOUS + CATEGORICAL

FEATURE_HELP = {
    "age": "Age in years.",
    "trestbps": "Resting blood pressure (mm Hg).",
    "chol": "Serum cholesterol (mg/dL).",
    "thalach": "Maximum heart rate recorded in the dataset.",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "sex": "Dataset code: 0 = female, 1 = male.",
    "cp": "Chest-pain type code, 1–4.",
    "fbs": "Fasting blood sugar > 120 mg/dL: 0 = no, 1 = yes.",
    "restecg": "Resting ECG code: 0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy.",
    "exang": "Exercise-induced angina: 0 = no, 1 = yes.",
    "slope": "Peak exercise ST segment slope: 1 = upsloping, 2 = flat, 3 = downsloping.",
    "ca": "Number of major vessels (0–3).",
    "thal": "Original thal code: 3 = normal, 6 = fixed defect, 7 = reversible defect.",
}
CATEGORY_OPTIONS = {
    "sex": [{"value": 0, "label": "Female (0)"}, {"value": 1, "label": "Male (1)"}],
    "cp": [
        {"value": 1, "label": "Typical angina (1)"},
        {"value": 2, "label": "Atypical angina (2)"},
        {"value": 3, "label": "Non-anginal pain (3)"},
        {"value": 4, "label": "Asymptomatic (4)"},
    ],
    "fbs": [{"value": 0, "label": "No (0)"}, {"value": 1, "label": "Yes (1)"}],
    "restecg": [
        {"value": 0, "label": "Normal (0)"},
        {"value": 1, "label": "ST-T abnormality (1)"},
        {"value": 2, "label": "LV hypertrophy (2)"},
    ],
    "exang": [{"value": 0, "label": "No (0)"}, {"value": 1, "label": "Yes (1)"}],
    "slope": [
        {"value": 1, "label": "Upsloping (1)"},
        {"value": 2, "label": "Flat (2)"},
        {"value": 3, "label": "Downsloping (3)"},
    ],
    "ca": [{"value": value, "label": str(value)} for value in range(4)],
    "thal": [
        {"value": 3, "label": "Normal (3)"},
        {"value": 6, "label": "Fixed defect (6)"},
        {"value": 7, "label": "Reversible defect (7)"},
    ],
}

app = FastAPI(
    title="BayesPulse API",
    version="1.0.0",
    description=(
        "Educational API adapter for the existing BayesPulse classifier. "
        "Not for diagnosis or patient-care decisions."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class FeatureValues(BaseModel):
    """Explicit predictor schema so OpenAPI documents real BayesPulse fields."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 55,
                "trestbps": 130,
                "chol": 240,
                "thalach": 150,
                "oldpeak": 1.2,
                "sex": 1,
                "cp": 2,
                "fbs": 0,
                "restecg": 1,
                "exang": 0,
                "slope": 2,
                "ca": 0,
                "thal": 3,
            }
        },
    )

    age: float | None = Field(default=None, description=FEATURE_HELP["age"])
    trestbps: float | None = Field(default=None, description=FEATURE_HELP["trestbps"])
    chol: float | None = Field(default=None, description=FEATURE_HELP["chol"])
    thalach: float | None = Field(default=None, description=FEATURE_HELP["thalach"])
    oldpeak: float | None = Field(default=None, description=FEATURE_HELP["oldpeak"])
    sex: float | None = Field(default=None, description=FEATURE_HELP["sex"])
    cp: float | None = Field(default=None, description=FEATURE_HELP["cp"])
    fbs: float | None = Field(default=None, description=FEATURE_HELP["fbs"])
    restecg: float | None = Field(default=None, description=FEATURE_HELP["restecg"])
    exang: float | None = Field(default=None, description=FEATURE_HELP["exang"])
    slope: float | None = Field(default=None, description=FEATURE_HELP["slope"])
    ca: float | None = Field(default=None, description=FEATURE_HELP["ca"])
    thal: float | None = Field(default=None, description=FEATURE_HELP["thal"])


class PredictionRequest(BaseModel):
    """Request wrapper for a prediction."""

    model_config = ConfigDict(extra="forbid")
    features: FeatureValues = Field(
        default_factory=FeatureValues,
        description="Predictor values. Omitted or null features are treated as missing.",
    )


def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH, header=0, na_values=["?"])
    if frame.shape != (303, 14) or frame["num"].isna().any():
        raise RuntimeError("Expected the complete Cleveland dataset with 303 rows and target num.")
    return frame


@lru_cache(maxsize=1)
def get_model() -> HybridNaiveBayes:
    frame = load_dataset()
    X = frame[FEATURES].to_numpy(dtype=float)
    y = (frame["num"] > 0).astype(int).to_numpy()
    return HybridNaiveBayes(
        continuous_features=tuple(range(len(CONTINUOUS))),
        categorical_features=tuple(range(len(CONTINUOUS), len(FEATURES))),
        alpha=1.0,
        variance_smoothing=1e-9,
    ).fit(X, y)


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "bayespulse-api"}


@app.get("/api/v1/metadata")
def metadata() -> dict[str, Any]:
    frame = load_dataset()
    features = []
    for name in FEATURES:
        continuous = name in CONTINUOUS
        observed = pd.to_numeric(frame[name], errors="coerce").dropna()
        item: dict[str, Any] = {
            "name": name,
            "kind": "continuous" if continuous else "categorical",
            "description": FEATURE_HELP[name],
            "required": False,
        }
        if continuous:
            item.update({"min": float(observed.min()), "max": float(observed.max()), "default": float(observed.median())})
        else:
            values = sorted(int(value) for value in observed.unique())
            mode = frame[name].mode(dropna=True)
            default = int(mode.iloc[0]) if not mode.empty else values[0]
            item.update({"options": CATEGORY_OPTIONS[name], "default": default})
        features.append(item)
    return {
        "name": "BayesPulse",
        "dataset": {"name": "UCI Heart Disease, Cleveland subset", "records": len(frame)},
        "target": {"name": "num > 0", "class_0": "num = 0", "class_1": "num > 0"},
        "features": features,
        "limitations": [
            "Educational demonstration only, not a medical device or clinical decision tool.",
            "The interactive model is fitted on all 303 dataset records; outputs are not held-out estimates.",
            "Do not enter real patient information.",
        ],
    }


@app.post("/api/v1/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    supplied_features = request.features.model_dump(exclude_unset=True)
    row_values: list[float] = []
    for name in FEATURES:
        value = supplied_features.get(name)
        if value is None:
            row_values.append(float("nan"))
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"Feature '{name}' must be numeric or null.")
        if not np.isfinite(numeric):
            raise HTTPException(status_code=422, detail=f"Feature '{name}' must be finite or null.")
        row_values.append(numeric)

    row = np.asarray([row_values], dtype=float)
    model = get_model()
    posterior = model.predict_proba(row)[0]
    explanation = model.explain_log_odds(row, positive_class=1)
    contributions = np.asarray(explanation["feature_log_likelihood_ratios"], dtype=float)[0]
    progression = model.predict_proba_progression(row)[0]

    feature_evidence = []
    for index, name in enumerate(FEATURES):
        value = row_values[index]
        contribution = float(contributions[index])
        feature_evidence.append({
            "feature": name,
            "value": None if not np.isfinite(value) else value,
            "log_likelihood_ratio": contribution,
            "evidence": (
                "missing"
                if not np.isfinite(value)
                else "toward_class_1"
                if contribution > 0
                else "toward_class_0"
                if contribution < 0
                else "neutral"
            ),
        })

    trace = [{"evidence_included": "Prior", "class_0": float(step[0]), "class_1": float(step[1])}
             for step in progression]
    trace_names = ["Prior"] + FEATURES
    for index, item in enumerate(trace):
        item["evidence_included"] = trace_names[index]

    return {
        "predicted_class": int(model.classes_[int(np.argmax(posterior))]),
        "probabilities": {"class_0": float(posterior[0]), "class_1": float(posterior[1])},
        "prior_log_odds": float(explanation["prior_log_odds"]),
        "total_log_odds": float(np.asarray(explanation["total_log_odds"])[0]),
        "feature_evidence": feature_evidence,
        "updating_trace": trace,
        "interpretation": "Model output for recorded dataset class, not an estimate of future individual disease risk.",
    }
