# Dataset handling

The primary research dataset is the CDC 2025 Behavioral Risk Factor Surveillance System (BRFSS) annual public-use file.

- Official data page: https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html
- XPT archive: https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip
- Official codebook: https://www.cdc.gov/brfss/annual_data/2025/zip/codebook25_llcp-v2-508.zip
- Complex sampling guidance: https://www.cdc.gov/brfss/annual_data/2025/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2025-508.pdf

## Local layout

Keep the original ZIP archive locally:

\`\`\`text
data/
└── raw/
    └── LLCP2025XPT.zip
\`\`\`

The scripts accept the ZIP archive directly; manual extraction is not required.

## Prepare the full sample

\`\`\`bash
python -m pip install -r requirements-research.txt
python scripts/prepare_brfss.py --input data/raw/LLCP2025XPT.zip
\`\`\`

The preparation script retains all 356,158 source respondents in \`data/processed/brfss2025_stroke_history_analysis.csv.gz\`. The 1,208 unknown/refused/blank stroke-history outcomes remain missing in \`stroke_history\`; only rows coded Yes or No are eligible for binary-model fitting and evaluation.

Keeping the full source file allows the survey design variables to be defined before subsetting to the valid outcome domain. The output is respondent-level research data and stays local.

## Exploratory data analysis

\`\`\`bash
python -m pip install -r requirements-research.txt
python scripts/eda_brfss.py --input data/processed/brfss2025_stroke_history_analysis.csv.gz
\`\`\`

The EDA script loads the full prepared file for survey-design diagnostics, then filters to valid Yes/No outcomes for the reported-stroke descriptive tables and charts. It writes aggregate outputs to \`reports/generated/\`.

## Survey-aware validation folds

The helper \`scripts/survey_cv.py\` assigns random folds separately within each \`survey_stratum\` and checks that a PSU is not split across folds. It deliberately does not use the stroke outcome when assigning folds.

\`\`\`python
from scripts.survey_cv import assign_survey_folds, fold_adjusted_weights

folds = assign_survey_folds(prepared_data, n_splits=5, random_state=42)
train_weights, test_weights = fold_adjusted_weights(
    prepared_data, folds, heldout_fold=0
)
\`\`\`

Only after assigning folds should the modelling code subset to valid stroke-history outcomes. The fold-adjusted weights are multiplied by \`survey_weight\` for weighted training and evaluation.

See [feature selection and validation design](../docs/FEATURE-SELECTION-VALIDATION.md).

## Important handling rules

- Do not commit the source XPT/ZIP file, respondent-level CSV exports, processed respondent-level data, or notebooks containing copied rows to GitHub.
- Keep the original archive unchanged. Perform transformations on copies or in a reproducible pipeline.
- Record the download date, source URL, archive/file name, file size, and SHA-256 checksum in the research log.
- Verify selected variable definitions and response codes against the official 2025 codebook.
- Do not treat refused, unknown, blank, or otherwise missing responses as a “No” outcome.
- Do not publish identifiable or sensitive respondent-level information.
- Keep generated outputs and local model artifacts out of version control.

## Research scope

The dataset is a cross-sectional US health survey. The outcome \`CVDSTRK3\` concerns self-reported history of stroke, not future stroke incidence. Results should not be presented as clinical diagnosis or as a validated prediction of an individual's future stroke.

See the [research plan](../docs/RESEARCH-PLAN.md), [initial dataset audit](../docs/DATA-AUDIT-2025.md), [preparation report](../docs/PREPARATION-2025.md), and [EDA report](../docs/EDA-2025.md).
