# Explainable Machine Learning for Stroke-History Classification

A B.Sc. Computer Science research project exploring whether machine-learning models can classify **self-reported history of stroke** from health-survey responses and explain the patterns they learn.

> **Research scope:** The selected CDC BRFSS source is a cross-sectional survey. Its stroke outcome records reported history of stroke; it does not establish whether a respondent will experience a stroke in the future. This project is not a clinical diagnostic system, medical device, or validated personal risk calculator.

## Current status

This repository is an early prototype and research scaffold.

- A small FastAPI backend and HTML/Bootstrap interface exist.
- The current `app.py` uses a hand-written scoring formula, not a trained machine-learning model.
- The UI collects some variables that the current scoring formula does not use.
- SHAP/LIME explanations, model training, validation and clinical evaluation have **not** yet been implemented.

The prototype's current output must not be interpreted as a medically meaningful probability or recommendation. The research pipeline will replace the demonstration score after the dataset has been inspected and models have been trained and evaluated.

## Selected research data source

**CDC 2025 Behavioral Risk Factor Surveillance System (BRFSS)**

- [Official 2025 annual data and documentation](https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html)
- [Official public-use XPT archive](https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip)

BRFSS is a US cross-sectional health survey. We will verify the outcome coding, candidate predictors, missing-value codes and survey design variables against the official 2025 codebook before analysis. Because the source represents US survey respondents, findings must not automatically be generalised to Nigerian clinical populations.

### Data handling

Download and extract the XPT archive locally. Keep the original data under `data/raw/`. Raw respondent-level data, processed respondent-level exports and local analysis outputs are excluded from Git by `.gitignore`.

Do not commit the CDC source file, patient-level exports, private data or secrets. See [data/README.md](data/README.md) for handling instructions and [docs/RESEARCH-PLAN.md](docs/RESEARCH-PLAN.md) for the analysis plan.

## Planned research workflow

1. **Dataset inspection:** verify the CDC codebook, outcome definition, field names, survey weights, strata and primary sampling units.
2. **Data quality and EDA:** assess record counts, missingness, duplicates, coding, distributions and outcome prevalence.
3. **Preprocessing:** build documented, reproducible transformations without treating unknown/refused responses as negative responses.
4. **Modelling:** establish a Logistic Regression baseline, then compare appropriate tree-based models.
5. **Evaluation:** assess recall/sensitivity, specificity, precision, F1, precision-recall AUC, ROC-AUC and calibration. Account for class imbalance and the survey design.
6. **Explainability:** add SHAP only after selecting and validating a model; explanations describe model behaviour, not medical causation.
7. **Application integration:** connect the trained, versioned model to the API and clearly communicate the research outcome and limitations.

## Technology

- Python
- FastAPI and Uvicorn
- HTML5, Bootstrap 5 and JavaScript
- Planned: pandas, NumPy, scikit-learn, a suitable survey-analysis package and SHAP (selected versions will be pinned after the analysis environment is established)

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

## Project documentation

- [Research plan](docs/RESEARCH-PLAN.md)
- [Dataset handling guide](data/README.md)

## Author

**Animasaun Damilare Aderinwale**  
B.Sc. Computer Science  
Federal University Oye-Ekiti, Nigeria

## Supervisor

Dr. Stephen Aderibigbe

## License

For academic and research purposes. A specific open-source licence has not yet been selected.
