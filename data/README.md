# Dataset handling

The primary research dataset is the CDC 2025 Behavioral Risk Factor Surveillance System (BRFSS) annual public-use file.

- Official data page: https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html
- XPT archive: https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip
- Official codebook: https://www.cdc.gov/brfss/annual_data/2025/zip/codebook25_llcp-v2-508.zip

## Local layout

Keep the original ZIP archive locally:

```text
data/
└── raw/
    └── LLCP2025XPT.zip
```

Both scripts accept the ZIP archive directly; you do not need to extract it yourself.

## Run the data audit

```bash
python -m pip install -r requirements-research.txt
python scripts/audit_brfss.py --input data/raw/LLCP2025XPT.zip
```

The audit script reads the XPT in chunks and writes aggregate summaries to `reports/generated/`. It does not write respondent-level rows.

## Prepare the analysis extract

```bash
python scripts/prepare_brfss.py --input data/raw/LLCP2025XPT.zip
```

This creates `data/processed/brfss2025_stroke_history_analysis.csv.gz` plus an aggregate `preparation_profile.json`. The analysis extract is respondent-level data and stays local; the script does not impute missing predictors or resample classes. It preserves survey weights, strata, and PSUs as separate metadata columns.

The CDC ZIP member currently has a trailing space in its internal XPT filename; the scripts normalize the name when extracting to a temporary directory.

## Important handling rules

- Do not commit the source XPT/ZIP file, respondent-level CSV exports, processed respondent-level data, or notebooks containing copied rows to GitHub.
- Keep the original archive unchanged. Perform transformations on copies or in a reproducible pipeline.
- Record the download date, source URL, archive/file name, file size, and SHA-256 checksum in the research log.
- Verify selected variable definitions and response codes against the official 2025 codebook.
- Do not treat refused, unknown, blank, or otherwise missing responses as a “No” outcome.
- Do not publish identifiable or sensitive respondent-level information.
- Keep generated outputs and local model artifacts out of version control.

## Research scope

The dataset is a cross-sectional US health survey. The outcome `CVDSTRK3` concerns self-reported history of stroke, not future stroke incidence. Results should not be presented as clinical diagnosis or as a validated prediction of an individual's future stroke.

See [the research plan](../docs/RESEARCH-PLAN.md), [the initial data audit](../docs/DATA-AUDIT-2025.md), and [the preparation report](../docs/PREPARATION-2025.md).
