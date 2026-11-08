from fastapi.testclient import TestClient

from app import CLASSIFICATION_THRESHOLD, MODEL, app

client = TestClient(app)


def valid_profile():
    return {
        "age": 58,
        "sex": "Female",
        "hypertension": 1,
        "chd_or_mi_history": 0,
        "diabetes_status": "No diabetes",
        "bmi": 27.4,
        "smoking_status": "Former smoker",
        "physical_activity_30d": 1,
    }


def test_root_serves_interface():
    response = client.get("/")
    assert response.status_code == 200
    assert "Stroke History Explorer" in response.text
    assert "not a medical assessment" in response.text


def test_health_reports_model_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is (MODEL is not None)


def test_predict_returns_reproducible_classification():
    if MODEL is None:
        return
    response = client.post("/predict", json=valid_profile())
    assert response.status_code == 200
    result = response.json()
    assert result["classification"] in (0, 1)
    assert 0 <= result["positive_class_probability"] <= 1
    assert result["threshold"] == CLASSIFICATION_THRESHOLD
    assert "future stroke risk" in result["interpretation"]


def test_optional_predictors_can_remain_missing():
    if MODEL is None:
        return
    response = client.post("/predict", json={"age": 80})
    assert response.status_code == 200
    assert response.json()["classification"] in (0, 1)


def test_invalid_age_is_rejected():
    response = client.post("/predict", json={"age": 17})
    assert response.status_code == 422
