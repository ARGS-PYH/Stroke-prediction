# Research demonstrator interface and API

## Purpose

The web interface lets a user submit a **hypothetical BRFSS-style respondent profile** to the primary eight-feature survey-weighted Logistic Regression model. It is an academic model demonstrator, not a clinical application.

## Model integration

- Model artifact: models/logistic_regression_primary_eight_features.joblib.gz
- API: POST /predict
- Health check: GET /health
- Interface: GET /
- Input variables: age, sex, hypertension, CHD/MI history, diabetes status, BMI, smoking status, and leisure-time physical activity in the past 30 days.
- Unknown or unanswered predictor values are sent as missing. They are not recoded as No; the model's fitted preprocessing pipeline handles them.
- Age 80 and above is represented as 80, matching the source's top-coded _AGE80 field.

The API uses the development-selected Logistic Regression threshold of 0.0338646426, chosen by weighted Youden's J on development-only out-of-fold scores. The held-out test set was not used to select this threshold. The response includes the positive-class model score, the resulting BRFSS Yes/No classification, and the threshold used.

## Interpreting the result

The positive outcome corresponds to CVDSTRK3: the respondent says a health professional has ever told them they had a stroke. This is a cross-sectional self-reported history outcome. It does **not** represent future stroke incidence. The probability returned by predict_proba is a model estimate for the survey response class, not an individual's medical probability.

On the held-out sample, the Logistic Regression model had weighted ROC-AUC 0.8134, weighted PR-AUC 0.1421, weighted sensitivity 78.71%, and weighted precision 8.80%. Precision is low; many positive classifications did not correspond to respondents reporting a previous stroke. These are weighted point estimates only; design-based confidence intervals have not been calculated.

## Run locally

    python -m pip install -r requirements.txt
    python -m uvicorn app:app --reload

Open http://127.0.0.1:8000. API documentation is available at http://127.0.0.1:8000/docs.

## Limitations

- BRFSS represents a U.S. adult survey population; results do not validate use in Nigeria.
- Some predictors may have occurred after the reported stroke; the source does not establish temporal order.
- There are no design-based confidence intervals for model performance in this version.
- SHAP/LIME explanations, external validation, and clinical validation have not been implemented.
- Use hypothetical profiles only. Do not enter real patient or identifiable information.


## Interface usability features

- The **Try example** control fills the form with a hypothetical profile. It does not submit the profile automatically; the user must choose to run the classification.
- The header checks the `/health` endpoint and shows whether the model artifact is available.
- The result meter adjusts its visible range to keep the model score and selected threshold readable together. The displayed range is a visual aid and does not change the model score or classification.
- The form identifies age as required and the remaining seven predictors as optional. Unanswered predictors remain missing and are handled by the fitted preprocessing pipeline.
