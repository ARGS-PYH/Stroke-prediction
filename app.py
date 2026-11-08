"""FastAPI service for the BRFSS 2025 stroke-history research demonstrator.

This application classifies a respondent profile against the BRFSS outcome
"ever told by a health professional that they had a stroke". It does not
estimate future stroke incidence or provide a clinical assessment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import gzip
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_primary_eight_features.joblib.gz"
INDEX_PATH = BASE_DIR / "index.html"

# Development-only out-of-fold threshold selected using weighted Youden's J.
CLASSIFICATION_THRESHOLD = 0.033864642623036026
MODEL_NAME = "Survey-weighted Logistic Regression"
FEATURE_COLUMNS = [
    "age_years_capped_at_80",
    "sex",
    "hypertension",
    "chd_or_mi_history",
    "diabetes_status",
    "bmi",
    "smoking_status",
    "physical_activity_30d",
]

try:
    with gzip.open(MODEL_PATH, "rb") as model_stream:
        MODEL = joblib.load(model_stream)
    MODEL_LOAD_ERROR: str | None = None
except Exception as exc:  # Keep API health route available to explain setup errors.
    MODEL = None
    MODEL_LOAD_ERROR = f"Could not load model artifact: {exc}"

app = FastAPI(
    title="BRFSS 2025 Stroke-History Explorer",
    description=(
        "Academic demonstrator for classifying self-reported previous stroke "
        "history in the CDC BRFSS 2025 survey. Not a clinical tool."
    ),
    version="1.0.0",
)


class RespondentProfile(BaseModel):
    """BRFSS-aligned input fields; unanswered predictors remain missing."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=120, description="Age in years; 80 represents age 80 or older")
    sex: Literal["Male", "Female"] | None = None
    hypertension: Literal[0, 1] | None = None
    chd_or_mi_history: Literal[0, 1] | None = None
    diabetes_status: Literal[
        "Diabetes",
        "Pregnancy-only diabetes",
        "No diabetes",
        "Prediabetes or borderline diabetes",
    ] | None = None
    bmi: float | None = Field(default=None, ge=10, le=100)
    smoking_status: Literal[
        "Current smoker - every day",
        "Current smoker - some days",
        "Former smoker",
        "Never smoked",
    ] | None = None
    physical_activity_30d: Literal[0, 1] | None = None


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    if not INDEX_PATH.is_file():
        raise HTTPException(status_code=404, detail="The interface file was not found.")
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if MODEL is not None else "model_unavailable",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_NAME,
        "model_load_error": MODEL_LOAD_ERROR,
    }


@app.post("/predict")
def classify_stroke_history(profile: RespondentProfile) -> dict[str, object]:
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The research model is not available. Check that the model artifact "
                "is present in the models directory."
            ),
        )

    values = profile.model_dump()
    values["age_years_capped_at_80"] = min(values.pop("age"), 80)

    # The fitted numeric imputers were trained with floating-point nullable
    # BRFSS columns. Preserve that dtype at inference, including missing values.
    for column in ("hypertension", "chd_or_mi_history", "physical_activity_30d"):
        values[column] = float(values[column]) if values[column] is not None else float("nan")
    values["bmi"] = float(values["bmi"]) if values["bmi"] is not None else float("nan")
    frame = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])

    try:
        positive_probability = float(MODEL.predict_proba(frame)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The model could not process this profile.") from exc

    is_positive = positive_probability >= CLASSIFICATION_THRESHOLD
    return {
        "model": MODEL_NAME,
        "classification": 1 if is_positive else 0,
        "classification_label": (
            "Classified in the BRFSS “Yes” response group"
            if is_positive
            else "Classified in the BRFSS “No” response group"
        ),
        "positive_class_probability": positive_probability,
        "positive_class_label": "Yes — ever told by a health professional that they had a stroke",
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_method": "Weighted Youden's J on development-only out-of-fold predictions",
        "outcome_definition": (
            "Self-reported answer to whether a health professional had ever told the "
            "respondent they had a stroke."
        ),
        "interpretation": (
            "This is a classification of a hypothetical BRFSS-style respondent profile. "
            "It is not an estimate of future stroke risk, a diagnosis, or an individual's "
            "personal probability of having had a stroke."
        ),
    }
