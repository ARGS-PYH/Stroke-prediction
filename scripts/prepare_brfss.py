#!/usr/bin/env python3
"""Create a reproducible BRFSS 2025 stroke-history analysis extract.

The respondent-level output is local research data and must not be committed.
Unknown/refused/missing outcome records are preserved with a missing target;
they are excluded only from binary-model fitting. Missing predictors are
preserved for later train-only imputation. Survey design fields are retained
separately and are not ordinary model predictors.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

SOURCE_FIELDS = [
    "CVDSTRK3", "_AGE80", "SEXVAR", "_RFHYPE6", "_MICHD", "DIABETE4",
    "_BMI5", "_SMOKER3", "EXERANY2", "_LLCPWT", "_STSTR", "_PSU",
]
DIABETES = {
    1: "Diabetes",
    2: "Pregnancy-only diabetes",
    3: "No diabetes",
    4: "Prediabetes or borderline diabetes",
}
SMOKING = {
    1: "Current smoker - every day",
    2: "Current smoker - some days",
    3: "Former smoker",
    4: "Never smoked",
}

PREDICTOR_FIELDS = [
    "age_years_capped_at_80", "sex", "hypertension", "chd_or_mi_history",
    "diabetes_status", "bmi", "smoking_status", "physical_activity_30d",
]
CATEGORY_FIELDS = [
    "sex", "hypertension", "chd_or_mi_history", "diabetes_status",
    "smoking_status", "physical_activity_30d",
]


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def get_xpt(input_path: Path, temp_dir: Path) -> tuple[Path, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "input_file": input_path.name,
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": sha256_file(input_path),
    }
    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as archive:
            members = [
                item for item in archive.infolist()
                if item.filename.strip().upper().endswith(".XPT")
            ]
            if len(members) != 1:
                raise ValueError(f"Expected one XPT in archive; found {len(members)}")
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"ZIP integrity check failed: {bad_member}")
            member = members[0]
            xpt = temp_dir / "LLCP2025.XPT"
            with archive.open(member) as source, xpt.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            metadata.update({
                "archive_member": member.filename,
                "xpt_size_bytes": member.file_size,
                "zip_integrity_check": "passed",
                "xpt_sha256": sha256_file(xpt),
            })
            return xpt, metadata
    if input_path.suffix.lower() != ".xpt":
        raise ValueError("Input must be the CDC ZIP archive or an extracted XPT file")
    metadata["xpt_sha256"] = metadata["input_sha256"]
    return input_path, metadata


def make_analysis_frame(chunk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recode every source row; non-binary outcomes remain nullable."""
    raw = chunk[SOURCE_FIELDS]
    out = pd.DataFrame(index=raw.index)
    out["stroke_history"] = raw["CVDSTRK3"].map({1: 1, 2: 0}).astype("Int8")
    # _AGE80 is in years; 80 is top-coded and means age 80 or older.
    out["age_years_capped_at_80"] = pd.to_numeric(
        raw["_AGE80"], errors="coerce"
    ).astype("Int64")
    out["sex"] = raw["SEXVAR"].map({1: "Male", 2: "Female"})
    # _RFHYPE6: 1=No, 2=Yes, 9=unknown/refused/missing.
    out["hypertension"] = raw["_RFHYPE6"].map({1: 0, 2: 1})
    # _MICHD: 1=reported MI or CHD, 2=did not report either; blank=missing.
    out["chd_or_mi_history"] = raw["_MICHD"].map({1: 1, 2: 0})
    # Preserve pregnancy-only and prediabetes as distinct codebook categories.
    out["diabetes_status"] = raw["DIABETE4"].map(DIABETES)
    # _BMI5 has two implied decimal places; 9999 is a missing sentinel.
    bmi_raw = pd.to_numeric(raw["_BMI5"], errors="coerce").replace(9999, pd.NA)
    out["bmi"] = bmi_raw.astype("Float64") / 100.0
    out["smoking_status"] = raw["_SMOKER3"].map(SMOKING)
    out["physical_activity_30d"] = raw["EXERANY2"].map({1: 1, 2: 0})
    out["survey_weight"] = pd.to_numeric(raw["_LLCPWT"], errors="coerce")
    out["survey_stratum"] = pd.to_numeric(raw["_STSTR"], errors="coerce").astype("Int64")
    out["survey_psu"] = pd.to_numeric(raw["_PSU"], errors="coerce").astype("Int64")

    valid = out["stroke_history"].notna()
    counts = {
        1: int((out.loc[valid, "stroke_history"] == 1).sum()),
        0: int((out.loc[valid, "stroke_history"] == 0).sum()),
    }
    raw_codes = raw["CVDSTRK3"].value_counts(dropna=False).to_dict()
    excluded = {
        ("blank_or_missing" if pd.isna(code) else str(int(code))): int(number)
        for code, number in raw_codes.items()
        if pd.isna(code) or int(code) not in (1, 2)
    }
    return out.reset_index(drop=True), {
        "rows_seen": int(len(chunk)),
        "rows_written": int(len(out)),
        "valid_binary_outcome_n": int(valid.sum()),
        "valid_outcome_counts": counts,
        "excluded_from_binary_ml_counts": excluded,
    }


def prepare(input_path: Path, output_dir: Path, chunksize: int) -> dict[str, Any]:
    """Write all 2025 respondents; keep invalid outcome rows for survey analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "brfss2025_stroke_history_analysis.csv.gz"
    profile_file = output_dir / "preparation_profile.json"
    if output_file.exists():
        output_file.unlink()

    rows_read = 0
    rows_written = 0
    valid_outcome_n = 0
    target_counts: Counter[int] = Counter()
    excluded_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    weighted_targets: Counter[int] = Counter()
    category_counts: dict[str, Counter[str]] = {
        col: Counter() for col in CATEGORY_FIELDS
    }
    strata_counts: Counter[int] = Counter()
    stratum_psu_pairs: set[tuple[int, int]] = set()
    bmi_min: float | None = None
    bmi_max: float | None = None
    age_min: int | None = None
    age_max: int | None = None
    header_written = False
    source_metadata: dict[str, Any] = {}
    source_column_count: int | None = None

    with tempfile.TemporaryDirectory(prefix="brfss2025-prepare-") as temp_name:
        xpt_path, source_metadata = get_xpt(input_path, Path(temp_name))
        reader = pd.read_sas(
            xpt_path,
            format="xport",
            iterator=True,
            chunksize=chunksize,
            encoding="latin1",
        )
        for chunk in reader:
            if source_column_count is None:
                source_column_count = len(chunk.columns)
            absent = set(SOURCE_FIELDS) - set(chunk.columns)
            if absent:
                raise ValueError("Missing source fields: " + ", ".join(sorted(absent)))

            clean, stats = make_analysis_frame(chunk)
            rows_read += stats["rows_seen"]
            rows_written += stats["rows_written"]
            valid_outcome_n += stats["valid_binary_outcome_n"]
            target_counts.update(stats["valid_outcome_counts"])
            excluded_counts.update(stats["excluded_from_binary_ml_counts"])

            # Predictor summaries are calculated only among valid Yes/No outcomes.
            valid = clean["stroke_history"].notna()
            model_rows = clean.loc[valid]
            for col in PREDICTOR_FIELDS:
                missing_counts[col] += int(model_rows[col].isna().sum())
            for col in CATEGORY_FIELDS:
                category_counts[col].update(
                    model_rows[col].fillna("Missing").astype(str).value_counts().to_dict()
                )
            if model_rows["bmi"].notna().any():
                lo, hi = float(model_rows["bmi"].min()), float(model_rows["bmi"].max())
                bmi_min = lo if bmi_min is None else min(bmi_min, lo)
                bmi_max = hi if bmi_max is None else max(bmi_max, hi)
            if model_rows["age_years_capped_at_80"].notna().any():
                lo = int(model_rows["age_years_capped_at_80"].min())
                hi = int(model_rows["age_years_capped_at_80"].max())
                age_min = lo if age_min is None else min(age_min, lo)
                age_max = hi if age_max is None else max(age_max, hi)

            # Keep design counts on the complete source sample, before outcome subsetting.
            for stratum, number in clean["survey_stratum"].value_counts().items():
                strata_counts[int(stratum)] += int(number)
            stratum_psu_pairs.update(
                (int(stratum), int(psu))
                for stratum, psu in clean[["survey_stratum", "survey_psu"]]
                .dropna().drop_duplicates().itertuples(index=False, name=None)
            )
            weighted_by_outcome = (
                model_rows.groupby("stroke_history")["survey_weight"].sum()
            )
            for code, value in weighted_by_outcome.items():
                weighted_targets[int(code)] += float(value)

            with gzip.open(
                output_file,
                "at" if header_written else "wt",
                encoding="utf-8",
                newline="",
            ) as handle:
                clean.to_csv(handle, index=False, header=not header_written, na_rep="")
            header_written = True

    if rows_written == 0 or valid_outcome_n == 0:
        raise ValueError("The source file contained no valid binary stroke-history outcomes")

    yes_n = int(target_counts[1])
    no_n = int(target_counts[0])
    weight_total = sum(weighted_targets.values())
    singleton_strata_n = sum(1 for size in strata_counts.values() if size == 1)

    profile = {
        "dataset": "CDC BRFSS 2025 LLCP",
        "source": source_metadata,
        "source_column_count": source_column_count,
        "rows_read": rows_read,
        "rows_written_to_full_sample_file": rows_written,
        "rows_with_valid_binary_outcome": valid_outcome_n,
        "rows_without_valid_binary_outcome_but_preserved": rows_written - valid_outcome_n,
        "outcome": {
            "source_variable": "CVDSTRK3",
            "question": "(Ever told) (you had) a stroke.",
            "mapping": {"1": 1, "2": 0},
            "yes_count": yes_n,
            "no_count": no_n,
            "unweighted_yes_percent_among_valid_yes_no": round(100 * yes_n / valid_outcome_n, 4),
            "weighted_yes_percent_among_valid_yes_no": round(
                100 * weighted_targets[1] / weight_total, 4
            ) if weight_total else None,
            "no_to_yes_ratio": round(no_n / yes_n, 3) if yes_n else None,
            "excluded_from_binary_ml_counts": dict(excluded_counts),
        },
        "predictor_missingness_among_valid_outcome_records": {
            col: {
                "missing_n": int(missing_counts[col]),
                "missing_percent": round(100 * missing_counts[col] / valid_outcome_n, 3),
            }
            for col in PREDICTOR_FIELDS
        },
        "categorical_distributions_among_valid_outcome_records": {
            col: dict(counter) for col, counter in category_counts.items()
        },
        "ranges_among_valid_outcome_records": {
            "age_min": age_min,
            "age_max": age_max,
            "age_80_note": "80 is top-coded and represents age 80 or older.",
            "bmi_min": round(bmi_min, 2) if bmi_min is not None else None,
            "bmi_max": round(bmi_max, 2) if bmi_max is not None else None,
        },
        "survey_design_full_sample": {
            "weight_variable": "_LLCPWT",
            "strata_variable": "_STSTR",
            "psu_variable": "_PSU",
            "unique_strata": len(strata_counts),
            "unique_stratum_psu_pairs": len(stratum_psu_pairs),
            "strata_with_one_respondent_psu": singleton_strata_n,
            "psu_note": (
                "The codebook labels _PSU as Primary Sampling Unit (equal to Annual Sequence Number). "
                "PSU identifiers are nested within strata; preserve _STSTR and _PSU for survey analysis."
            ),
            "weights_preserved_not_used_as_predictors": True,
        },
        "output_file": output_file.name,
        "output_rows": rows_written,
        "output_size_bytes": output_file.stat().st_size,
        "output_sha256": sha256_file(output_file),
        "preprocessing_policy": {
            "full_sample_preservation": "Every source respondent is retained in the prepared file.",
            "binary_outcome": "Only CVDSTRK3 codes 1 and 2 map to 1 and 0; codes 7, 9 and blank remain missing.",
            "model_eligibility": "Filter to non-missing stroke_history only when fitting or evaluating a binary classifier.",
            "unknown_predictors": "Unknown, refused and blank predictor values stay missing, not No.",
            "bmi": "_BMI5 is divided by 100; blank and 9999 become missing.",
            "diabetes": "Codes 1 to 4 remain distinct categories; 7 and 9 become missing.",
            "smoking": "Codes 1 to 4 remain distinct categories; 9 becomes missing.",
            "age": "_AGE80 is retained in years; 80 represents age 80 or older.",
            "imputation": "No imputation is done here; fit imputers on training data only.",
        },
        "limitations": [
            "Cross-sectional self-reported history outcome; not future stroke incidence.",
            "Weighted descriptive proportion only; no design-based standard errors or confidence intervals.",
            "Predictor diagnoses do not establish whether they occurred before or after a stroke.",
            "BRFSS is a US adult survey; do not generalize directly to Nigeria.",
        ],
    }
    profile_file.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/LLCP2025XPT.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunksize", type=int, default=20_000)
    args = parser.parse_args()
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be positive")
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    profile = prepare(args.input, args.output_dir, args.chunksize)
    print(
        f"Rows read/written: {profile['rows_read']:,}/"
        f"{profile['rows_written_to_full_sample_file']:,}"
    )
    print(f"Valid Yes/No outcomes: {profile['rows_with_valid_binary_outcome']:,}")
    print(
        "Preserved unknown/refused/blank outcomes:",
        f"{profile['rows_without_valid_binary_outcome_but_preserved']:,}",
    )
    print(
        "Weighted Yes share among valid Yes/No outcomes:",
        profile["outcome"]["weighted_yes_percent_among_valid_yes_no"], "%",
    )
    print("Output:", args.output_dir / profile["output_file"])
    print("Profile:", args.output_dir / "preparation_profile.json")


if __name__ == "__main__":
    main()
