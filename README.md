# Explainable Machine Learning for Stroke-History Classification

A B.Sc. Computer Science research project exploring whether machine-learning models can classify **self-reported history of stroke** from health-survey responses and explain the patterns they learn.

> **Research scope:** The selected CDC BRFSS source is a cross-sectional survey. Its stroke outcome records reported history of stroke; it does not establish whether a respondent will experience a stroke in the future. This project is not a clinical diagnostic system, medical device, or validated personal risk calculator.

## Current status

This repository contains an early application prototype and an emerging research pipeline.

- A small FastAPI backend and HTML/Bootstrap interface exist.
- The current `app.py` uses a hand-written scoring formula, not a trained machine-learning model.
- The UI collects some variables that the current scoring formula does not use.
- The CDC BRFSS 2025 codebook and dataset have been inspected.
- A reproducible data audit and full-sample preparation script have been added.
- The prepared extract retains all 356,158 respondents; 1,208 non-binary/missing stroke outcomes remain missing and are excluded only from binary-model fitting.
- The initial eight-feature set and survey-aware validation procedure are documented, with a fold-assignment helper and unit tests.
- No model has been trained or evaluated; SHAP/LIME explanations and clinical evaluation have not been implemented.

The prototype's current output must not be interpreted as a medically meaningful probability or recommendation.

## Selected research data source

**CDC 2025 Behavioral Risk Factor Surveillance System (BRFSS)**

- [Official 2025 annual data and documentation](https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html)
- [Official public-use XPT archive](https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip)

The inspected file contains 356,158 respondents and 283 variables. The outcome `CVDSTRK3` asks whether a health professional has ever told the respondent they had a stroke. It is a measure of reported prior history, not future stroke incidence.

## Data handling

Download the ZIP archive locally and keep it under `data/raw/`. You can pass the ZIP directly to the scripts; manual extraction is not required. Raw and processed respondent-level data are excluded from Git.

- [Dataset handling guide](data/README.md)
- [Initial dataset audit](docs/DATA-AUDIT-2025.md)
- [Preprocessing report and recoding rules](docs/PREPARATION-2025.md)
- [EDA report](docs/EDA-2025.md)
- [Feature selection and survey-aware validation](docs/FEATURE-SELECTION-VALIDATION.md)
- [Research plan](docs/RESEARCH-PLAN.md)

### Reproduce the data preparation

```bash
python -m pip install -r requirements-research.txt
python scripts/audit_brfss.py --input data/raw/LLCP2025XPT.zip
python scripts/prepare_brfss.py --input data/raw/LLCP2025XPT.zip
python scripts/eda_brfss.py --input data/processed/brfss2025_stroke_history_analysis.csv.gz
```

The preparation script produces a local compressed analysis file at `data/processed/brfss2025_stroke_history_analysis.csv.gz` and aggregate preparation profile. It preserves the full 356,158-row source sample; only the 354,950 valid Yes/No outcome records are eligible for binary modelling. The respondent-level output is not stored in GitHub.

Run preprocessing tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Initial cohort definition

Only `CVDSTRK3` responses coded 1 (Yes) and 2 (No) are retained for binary classification. Codes 7 (Don't know/Not sure), 9 (Refused), and blank responses are excluded from the outcome cohort. Unknown/refused predictor values remain missing; the preparation step does not impute them.

The initial valid-outcome cohort has 354,950 records: 16,353 Yes and 338,597 No. The class ratio is approximately 20.7 to 1. The prepared extract retains the final survey weight (`_LLCPWT`), stratum (`_STSTR`), and primary sampling unit (`_PSU`) separately from model predictors.

## Current analysis status

The initial exploratory analysis is complete. The valid Yes/No cohort contains 354,950 respondents (16,353 Yes; 338,597 No). The unweighted Yes share is 4.61%, and the weighted share among valid Yes/No records is 3.66%. The initial descriptive comparisons show higher reported prior-stroke prevalence in older age bands and among respondents reporting hypertension, heart attack/coronary heart disease, diabetes, current smoking, or no recent leisure-time physical activity. These are unadjusted, cross-sectional comparisons, not causal or prospective risk estimates.

- [BRFSS 2025 EDA report](docs/EDA-2025.md)
- [Reproducible EDA script](scripts/eda_brfss.py)

No model has been trained. Survey-design standard errors and confidence intervals have not yet been calculated; the EDA report documents the singleton-PSU issue that must be handled before inferential survey estimates.

## Planned research workflow

1. **Data quality and EDA:** profile missingness, response distributions, survey-weighted prevalence, and feature relationships.
2. **Feature rationale:** confirm a defensible predictor set and consider possible temporal ambiguity in self-reported diagnoses.
3. **Survey-aware validation:** specify how weights, strata, and PSUs inform descriptive estimates and evaluation.
4. **Modelling:** establish a Logistic Regression baseline, then compare appropriate tree-based models.
5. **Evaluation:** report sensitivity/recall, specificity, precision, F1, PR-AUC, ROC-AUC, calibration, and confusion matrix. Accuracy alone is insufficient.
6. **Explainability:** add SHAP only after selecting and validating a model; explanations describe model behaviour, not causality.
7. **Application integration:** connect the trained, versioned model to the API and communicate the study limitations.

## Technology

- Python
- FastAPI and Uvicorn
- HTML5, Bootstrap 5 and JavaScript
- Research pipeline: pandas; scikit-learn and survey-analysis tooling to be added after the validation design is specified

## Run the current prototype

The commands below run the existing demonstration only; they do not train a model.

1. Clone the repository:

   ```bash
   git clone https://github.com/ARGS-PYH/Stroke-prediction.git
   cd Stroke-prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Git Bash
   source .venv/Scripts/activate
   ```

3. Install the current API dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the API:

   ```bash
   python -m uvicorn app:app --reload
   ```

5. Open the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

6. In a second terminal, serve the frontend:

   ```bash
   python -m http.server 5500
   ```

   Then open [http://127.0.0.1:5500/index.html](http://127.0.0.1:5500/index.html).

## Academic and ethical considerations

- The primary outcome is self-reported previous stroke, not future stroke incidence.
- Survey design, weighting, missingness, outcome imbalance and measurement limitations must be described.
- Model explanations must not be presented as causal effects.
- A US survey does not establish performance or clinical suitability in Nigeria.
- Do not enter real patient-identifiable information into the prototype.
- Outputs are for academic research and demonstration, not diagnosis, treatment or medical advice.

## Author

**Animasaun Damilare Aderinwale**  
B.Sc. Computer Science  
Federal University Oye-Ekiti, Nigeria

## Supervisor

Dr. Stephen Aderibigbe

## License

For academic and research purposes. A specific open-source licence has not yet been selected.
