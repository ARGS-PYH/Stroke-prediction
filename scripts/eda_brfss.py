#!/usr/bin/env python3
"""Generate aggregate EDA tables, charts and summary for the prepared BRFSS cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED = {
    "stroke_history", "age_years_capped_at_80", "sex", "hypertension",
    "chd_or_mi_history", "diabetes_status", "bmi", "smoking_status",
    "physical_activity_30d", "survey_weight", "survey_stratum", "survey_psu",
}
GROUPS = [
    "sex", "hypertension", "chd_or_mi_history", "diabetes_status",
    "smoking_status", "physical_activity_30d", "age_band", "bmi_band",
]
ORDERS = {
    "sex": ["Female", "Male"],
    "hypertension": ["No", "Yes"],
    "chd_or_mi_history": ["No", "Yes"],
    "diabetes_status": ["No diabetes", "Diabetes", "Prediabetes or borderline diabetes", "Pregnancy-only diabetes"],
    "smoking_status": ["Never smoked", "Former smoker", "Current smoker - some days", "Current smoker - every day"],
    "physical_activity_30d": ["No", "Yes"],
    "age_band": ["18–34", "35–44", "45–54", "55–64", "65–74", "75–80+"],
    "bmi_band": ["Underweight (<18.5)", "Normal (18.5–24.9)", "Overweight (25–29.9)", "Obesity (≥30)"],
}


def label_value(variable: str, value: object) -> str:
    mappings = {
        "hypertension": {0.0: "No", 1.0: "Yes"},
        "chd_or_mi_history": {0.0: "No", 1.0: "Yes"},
        "physical_activity_30d": {0.0: "No", 1.0: "Yes"},
    }
    return mappings.get(variable, {}).get(value, str(value))


def subgroup_table(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    rows = []
    for category, group in df.loc[df[variable].notna()].groupby(variable, observed=True, sort=False):
        weight_total = group["survey_weight"].sum()
        weighted_yes = group.loc[group.stroke_history.eq(1), "survey_weight"].sum()
        rows.append({
            "variable": variable,
            "category": label_value(variable, category),
            "respondents_n": len(group),
            "reported_stroke_yes_n": int(group.stroke_history.sum()),
            "unweighted_prevalence_pct": 100 * group.stroke_history.mean(),
            "weighted_prevalence_pct": 100 * weighted_yes / weight_total,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        order = {value: index for index, value in enumerate(ORDERS[variable])}
        result["_order"] = result.category.map(order).fillna(999)
        result = result.sort_values("_order").drop(columns="_order")
    return result


def weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return None
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/brfss2025_stroke_history_analysis.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated"))
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Prepared analysis file not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_df = pd.read_csv(args.input, compression="infer", low_memory=False)
    missing_fields = sorted(REQUIRED - set(all_df.columns))
    if missing_fields:
        raise SystemExit("Missing required fields: " + ", ".join(missing_fields))
    if not set(all_df.stroke_history.dropna().unique()).issubset({0, 1}):
        raise SystemExit("stroke_history must contain only 0 and 1")
    if all_df.survey_weight.isna().any() or all_df.survey_weight.le(0).any():
        raise SystemExit("Survey weights must be positive and non-missing")

    # Keep the full file available for design diagnostics, then use the valid
    # Yes/No outcome domain for descriptive classifier EDA.
    source_respondent_rows = len(all_df)
    df = all_df.loc[all_df.stroke_history.notna()].copy()

    df["age_band"] = pd.cut(
        df.age_years_capped_at_80, [17, 34, 44, 54, 64, 74, 80],
        labels=ORDERS["age_band"], include_lowest=True,
    )
    df["bmi_band"] = pd.cut(
        df.bmi, [0, 18.5, 25, 30, 100],
        labels=ORDERS["bmi_band"], right=False,
    )

    n = len(df)
    yes_n = int(df.stroke_history.sum())
    no_n = n - yes_n
    total_weight = float(df.survey_weight.sum())
    yes_weight = float(df.loc[df.stroke_history.eq(1), "survey_weight"].sum())
    weighted_yes_pct = 100 * yes_weight / total_weight

    subgroup = pd.concat([subgroup_table(df, variable) for variable in GROUPS], ignore_index=True)
    subgroup.to_csv(args.output_dir / "eda_subgroup_prevalence.csv", index=False)

    missing_cols = [
        "age_years_capped_at_80", "sex", "hypertension", "chd_or_mi_history",
        "diabetes_status", "bmi", "smoking_status", "physical_activity_30d",
    ]
    missingness = pd.DataFrame({
        "variable": missing_cols,
        "missing_n": [int(df[column].isna().sum()) for column in missing_cols],
    })
    missingness["missing_pct"] = 100 * missingness.missing_n / n
    missingness.to_csv(args.output_dir / "eda_missingness.csv", index=False)

    by_outcome = {}
    for outcome, group in df.groupby("stroke_history", sort=True):
        by_outcome[str(int(outcome))] = {
            "n": len(group),
            "age_unweighted_mean": float(group.age_years_capped_at_80.mean()),
            "age_weighted_mean": weighted_mean(group.age_years_capped_at_80, group.survey_weight),
            "bmi_unweighted_mean": float(group.bmi.mean()),
            "bmi_weighted_mean": weighted_mean(group.bmi, group.survey_weight),
        }

    strata_psu = df.groupby("survey_stratum").survey_psu.nunique()
    summary = {
        "dataset": "CDC BRFSS 2025 LLCP prepared valid Yes/No outcome cohort",
        "source_respondent_rows": source_respondent_rows,
        "outcome_missing_or_nonbinary_rows_preserved": source_respondent_rows - n,
        "respondents": n,
        "source_prepared_extract_columns": len(REQUIRED),
        "derived_eda_variables": ["age_band", "bmi_band"],
        "reported_prior_stroke_yes_n": yes_n,
        "no_reported_prior_stroke_n": no_n,
        "unweighted_yes_pct": 100 * yes_n / n,
        "weighted_yes_pct": weighted_yes_pct,
        "no_to_yes_ratio": no_n / yes_n,
        "exact_duplicate_rows_full_source": int(all_df.duplicated().sum()),
        "unique_strata_full_source": int(all_df.survey_stratum.nunique()),
        "unique_psu_codes_full_source": int(all_df.survey_psu.nunique()),
        "unique_stratum_psu_units_full_source": int(
            all_df[["survey_stratum", "survey_psu"]].drop_duplicates().shape[0]
        ),
        "single_respondent_strata_full_source": int(
            (all_df.groupby("survey_stratum").size() == 1).sum()
        ),
        "single_psu_strata_after_outcome_filter": int((strata_psu == 1).sum()),
        "weight_min": float(df.survey_weight.min()),
        "weight_median": float(df.survey_weight.median()),
        "weight_max": float(df.survey_weight.max()),
        "continuous_summaries_by_outcome": by_outcome,
        "missingness_pct": dict(zip(missingness.variable, missingness.missing_pct.round(4))),
        "interpretation": (
            "Descriptive associations with self-reported prior stroke history, not future incidence, "
            "causation, or clinical validation. Weighted subgroup figures are point estimates only; "
            "no survey-design standard errors or confidence intervals are calculated."
        ),
    }
    (args.output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(["No reported prior stroke", "Reported prior stroke"], [no_n, yes_n])
    ax.set_ylabel("Respondents")
    ax.set_title("Stroke-history outcome distribution (unweighted)")
    ax.ticklabel_format(axis="y", style="plain")
    ax.bar_label(bars, labels=[f"{no_n:,}", f"{yes_n:,}"], padding=3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "eda_outcome_distribution.png", dpi=180)
    plt.close(fig)

    age = subgroup[subgroup.variable.eq("age_band")].copy()
    age["_order"] = age.category.map({value: index for index, value in enumerate(ORDERS["age_band"])})
    age = age.sort_values("_order")
    positions = list(range(len(age)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    unweighted = ax.bar([i - width / 2 for i in positions], age.unweighted_prevalence_pct, width, label="Unweighted")
    weighted = ax.bar([i + width / 2 for i in positions], age.weighted_prevalence_pct, width, label="Weighted")
    ax.set_xticks(positions, age.category)
    ax.set_ylabel("Respondents reporting prior stroke (%)")
    ax.set_xlabel("Age band (80 is top-coded: 80 years or older)")
    ax.set_title("Reported stroke history by age band")
    ax.legend()
    ax.bar_label(unweighted, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(weighted, fmt="%.1f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "eda_age_prevalence.png", dpi=180)
    plt.close(fig)

    selected = subgroup[subgroup.variable.isin([
        "hypertension", "chd_or_mi_history", "diabetes_status", "smoking_status",
        "physical_activity_30d", "bmi_band",
    ])].copy()
    names = {
        "hypertension": "Hypertension",
        "chd_or_mi_history": "Heart attack / CHD history",
        "diabetes_status": "Diabetes",
        "smoking_status": "Smoking",
        "physical_activity_30d": "Exercise (past 30 days)",
        "bmi_band": "BMI",
    }
    selected["plot_label"] = selected.apply(
        lambda row: f"{names[row.variable]}: {row.category}", axis=1
    )
    selected = selected.sort_values("weighted_prevalence_pct")
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    bars = ax.barh(selected.plot_label, selected.weighted_prevalence_pct)
    ax.set_xlabel("Weighted respondents reporting prior stroke (%)")
    ax.set_title("Reported prior stroke prevalence across selected categories")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "eda_categorical_prevalence.png", dpi=180)
    plt.close(fig)

    ordered_missing = missingness.sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.barh(ordered_missing.variable, ordered_missing.missing_pct)
    ax.set_xlabel("Missing responses (%)")
    ax.set_title("Missingness among valid Yes/No outcome records")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "eda_missingness.png", dpi=180)
    plt.close(fig)

    print(
        f"N={n:,}; Yes={yes_n:,} ({100 * yes_n / n:.2f}% unweighted, "
        f"{weighted_yes_pct:.2f}% weighted); No={no_n:,}"
    )
    print(
        f"Duplicates={summary['exact_duplicate_rows']}; strata={summary['unique_strata']}; "
        f"PSU={summary['unique_psu_codes']}; single-PSU strata="
        f"{summary['single_psu_strata_after_outcome_filter']}"
    )
    print("Aggregate EDA outputs written to", args.output_dir)


if __name__ == "__main__":
    main()
