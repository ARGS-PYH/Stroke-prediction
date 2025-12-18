# Explainable AI Framework for Stroke Risk Prediction

## Project Overview

This project implements an **Explainable Artificial Intelligence (XAI)–based Stroke Risk Prediction System** designed as a clinical decision support tool. The system predicts an individual’s risk of stroke using routine demographic and clinical variables and provides **transparent explanations** for each prediction to support clinical trust and decision-making.

The project was developed as a **final-year B.Sc. Computer Science project** and focuses on applicability in **low- and middle-income settings such as Nigeria**, where early screening and prevention are critical.

---

## Key Features

* Web-based user interface for entering patient clinical data
* Machine learning–based stroke risk prediction
* Risk categorization (Low / Moderate / High)
* Explainable AI output showing key contributing risk factors
* RESTful backend API with auto-generated documentation (OpenAPI)
* Modular and reproducible system architecture

---

## System Architecture

The system follows a **client–server architecture**:

1. **Frontend (Web UI)**

   * HTML, Bootstrap
   * Collects patient data and displays predictions

2. **Backend API**

   * Built with FastAPI (Python)
   * Receives patient data and returns predictions

3. **Prediction Logic**

   * Stroke risk computed from clinical variables
   * Designed to be replaceable with trained ML models

4. **Explainability Layer**

   * Provides human-readable explanations inspired by SHAP/LIME

---

## Technologies Used

* **Python 3.9+**
* **FastAPI** (Backend framework)
* **Uvicorn** (ASGI server)
* **HTML5 & Bootstrap 5** (Frontend UI)
* **JavaScript (Fetch API)**

---

## Project Structure

```
stroke-risk-prediction-app/
│
├── app.py            # FastAPI backend
├── index.html        # Web user interface
├── README.md         # Project documentation
└── requirements.txt  # Python dependencies
```

---

## How to Run the Project

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd stroke-risk-prediction-app
```

### 2. Install Dependencies

```bash
python -m pip install fastapi uvicorn
```

### 3. Start the Backend Server

```bash
python -m uvicorn app:app --reload
```

The backend will run at:

```
http://127.0.0.1:8000
```

### 4. Open API Documentation

Visit:

```
http://127.0.0.1:8000/docs
```

### 5. Serve the Frontend

In a new terminal:

```bash
python -m http.server 5500
```

Open in browser:

```
http://localhost:5500/index.html
```

---

## Sample Input

```json
{
  "age": 60,
  "hypertension": 1,
  "heart_disease": 0,
  "avg_glucose_level": 155,
  "bmi": 29
}
```

## Sample Output

```json
{
  "risk_score": 0.9,
  "risk_level": "High",
  "explanation": "Age and hypertension contributed most to the risk."
}
```

---

## Explainable AI Component

Instead of returning only a numerical prediction, the system provides an **explainable insight** that highlights the most influential clinical features for each prediction. This improves transparency, trust, and suitability for healthcare decision support.

The explainability approach is inspired by **SHAP and LIME**, ensuring that predictions are interpretable by clinicians and non-technical stakeholders.

---

## Ethical Considerations

* The system is designed as a **decision support tool**, not a diagnostic replacement
* No real patient data is stored
* The model emphasizes transparency and accountability
* Deployment requires ethical approval and data governance compliance

---

## Limitations

* Uses simulated or publicly available data for demonstration
* Does not replace professional medical judgment
* Performance may vary across populations without local validation

---

## Future Work

* Integration of trained machine learning models (Random Forest, XGBoost)
* Visual SHAP-based explanations (charts)
* Deployment on cloud platforms
* External validation using Nigerian hospital datasets

---

## Author

**Animasaun Damilare Aderinwale**
B.Sc. Computer Science
Federal University Oye-Ekiti, Nigeria

---

## Supervisor

Dr. Stephen Aderibigbe

---

## License

This project is for **academic and research purposes only**.
