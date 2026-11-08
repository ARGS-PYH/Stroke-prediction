# Explainable Machine Learning for Stroke-History Classification

A B.Sc. Computer Science research project exploring whether machine-learning models can classify **self-reported history of stroke** from health-survey responses and explain the patterns they learn.

> **Research scope:** The selected CDC BRFSS source is a cross-sectional survey. Its stroke outcome records reported history of stroke; it does not establish whether a respondent will experience a stroke in the future. This project is not a clinical diagnostic system, medical device, or validated personal risk calculator.

## Current status

- The CDC BRFSS 2025 dataset has been audited and prepared using a reproducible pipeline.
- The prepared extract retains all 356,158 source respondents; 1,208 non-binary/missing stroke outcomes remain missing and are excluded only from binary-model fitting.
- The primary eight-feature set and survey-aware validation procedure are documented.
- A weighted Logistic Regression baseline and a matched primary-feature Random Forest comparison have been evaluated on the same held-out respondents.
- The redesigned FastAPI interface now uses the trained primary eight-feature Logistic Regression pipeline. The old hand-written score has been removed.
- The interface collects the same eight model inputs and displays a classification against the BRFSS self-reported stroke-history outcome.
- Model comparison shows near parity; Logistic Regression is the current app demonstrator, not a proven superior model.
- SHAP/LIME explanations, repeated outer validation, design-based confidence intervals, external validation, and clinical evaluation have not been implemented.

The app is for academic demonstration. Use hypothetical profiles only; do not enter real patient or identifiable information.

## Run the web demonstrator

The application serves the interface and API from the same FastAPI process.

    python -m venv .venv
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    # Git Bash / macOS / Linux: source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m uvicorn app:app --reload

Open http://127.0.0.1:8000. API documentation is available at http://127.0.0.1:8000/docs, and the model health check is at http://127.0.0.1:8000/health.

The app loads the versioned artifact at models/logistic_regression_primary_eight_features.joblib. The selected classification threshold is 0.0338646426, chosen using weighted Youden's J on development-only out-of-fold predictions. The final holdout was not used to choose the threshold.

See docs/INTERFACE-AND-API.md for input mapping, threshold handling, and interpretation guidance.

## Selected research data source

**CDC 2025 Behavioral Risk Factor Surveillance System (BRFSS)**

- [Official 2025 annual data and documentation](https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html)
- [Official public-use XPT archive](https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip)

The inspected file contains 356,158 respondents and 284 variables. The outcome CVDSTRK3 asks whether a health professional has ever told the respondent they had a stroke. It is a measure of reported prior history, not future stroke incidence.

## Data handling and research workflow

Download the ZIP archive locally and keep it under data/raw/. You can pass the ZIP directly to the scripts; manual extraction is not required. Raw and processed respondent-level data are excluded from Git.

- [Dataset handling guide](data/README.md)
- [Initial dataset audit](docs/DATA-AUDIT-2025.md)
- [Preprocessing report and recoding rules](docs/PREPARATION-2025.md)
- [EDA report](docs/EDA-2025.md)
- [Feature selection and survey-aware validation](docs/FEATURE-SELECTION-VALIDATION.md)
- [Random Forest and Logistic Regression comparison](docs/RANDOM-FOREST-2025.md)
- [Research plan](docs/RESEARCH-PLAN.md)

To reproduce the research data preparation and model comparison:

    python -m pip install -r requirements-research.txt
    python scripts/audit_brfss.py --input data/raw/LLCP2025XPT.zip
    python scripts/prepare_brfss.py --input data/raw/LLCP2025XPT.zip
    python scripts/eda_brfss.py --input data/processed/brfss2025_stroke_history_analysis.csv.gz
    python scripts/train_model_comparison.py --primary-only

The preparation script produces a local compressed analysis file at data/processed/brfss2025_stroke_history_analysis.csv.gz and an aggregate preparation profile. The respondent-level output is not stored in GitHub.

## Tests

    python -m pip install -r requirements-dev.txt
    python -m pytest -q

## Academic and ethical considerations

- The primary outcome is self-reported previous stroke, not future stroke incidence.
- Survey design, weighting, missingness, outcome imbalance, and measurement limitations must be described.
- Model explanations must not be presented as causal effects.
- A U.S. survey does not establish performance or clinical suitability in Nigeria.
- The app's positive-class score is a model estimate for the survey response category, not a personal medical probability.
- Weighted performance values are point estimates; design-based confidence intervals have not been calculated.

## Technology

- Python, FastAPI, Uvicorn, pandas, scikit-learn and joblib
- HTML, CSS and JavaScript

## Author

**Animasaun Damilare Aderinwale**  
B.Sc. Computer Science  
Federal University Oye-Ekiti, Nigeria

## Supervisor

Dr. Stephen Aderibigbe

## License

For academic and research purposes. A specific open-source licence has not yet been selected.
