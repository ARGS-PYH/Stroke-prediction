# Dataset handling

The primary research candidate is the CDC 2025 Behavioral Risk Factor Surveillance System (BRFSS) annual public-use dataset.

- Official data page: https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html
- XPT download: https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip

## Local layout

After downloading and extracting the official archive, keep the original XPT file locally under:

```text
data/
└── raw/
    └── LLCP2025.XPT
```

The exact extracted file name may differ; use the name supplied by the CDC archive.

## Important handling rules

- Do not commit the source XPT file, respondent-level CSV exports, processed respondent-level data, or notebooks containing copied rows to GitHub.
- The repository .gitignore excludes local raw/processed data and common data export formats.
- Keep the original source file unchanged. Perform transformations on copies or in a reproducible pipeline.
- Record the download date, source URL, archive/file name, file size and SHA-256 checksum in your local research log.
- Use the CDC's official 2025 codebook and technical documentation to verify every selected variable and its response codes before analysis.
- Do not treat refused, unknown, blank or otherwise missing responses as a "No" outcome.
- Do not publish identifiable or sensitive respondent-level information.

## Planned first inspection

1. Confirm the extracted XPT file can be read.
2. Record row and column counts and inspect data types.
3. Verify the stroke-history outcome variable and response coding from the official codebook.
4. Identify the documented survey weight, stratum and primary sampling-unit fields.
5. Profile missingness, duplicate records and outcome prevalence.
6. Produce a data dictionary for the variables considered for analysis.

The dataset is a cross-sectional US health survey. The outcome concerns self-reported history of stroke, not future stroke incidence. Results should not be presented as clinical diagnosis or as a validated prediction of an individual's future stroke.
