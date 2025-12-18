from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Explainable Stroke Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: int
    hypertension: int = 0
    heart_disease: int = 0
    avg_glucose_level: float = 0.0
    bmi: float = 0.0

@app.post("/predict")
def predict_stroke(data: PatientData):
    risk_score = min(
        0.9,
        0.01 * data.age +
        0.15 * data.hypertension +
        0.10 * data.heart_disease
    )

    if risk_score < 0.2:
        level = "Low"
    elif risk_score < 0.5:
        level = "Moderate"
    else:
        level = "High"

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": level,
        "explanation": "Age and hypertension contributed most to the risk."
    }
