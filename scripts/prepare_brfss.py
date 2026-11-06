#!/usr/bin/env python3
"""Create a reproducible BRFSS 2025 stroke-history analysis extract.

The respondent-level output is local research data and must not be committed.
Unknown/refused/missing outcome records are excluded. Missing predictors are
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
    valid_outcome = chunk["CVDSTRK3"].isin([1, 2])
    raw_target_counts = chunk["CVDSTRK3"].value_counts(dropna=False).to_dict()
    raw = chunk.loc[valid_outcome, SOURCE_FIELDS]
    if raw.empty:
        return pd.DataFrame(), {"rows_seen": len(chunk), "rows_retained": 0}

    out = pd.DataFrame(index=raw.index)
    out["stroke_history"] = raw["CVDSTRK3"].map({1: 1, 2: 0}).astype("int8")
    # _AGE80 is in years; 80 is top-coded and means 80 years or older.
    out["age_years_capped_at_80"] = pd.to_numeric(
        raw["_AGE80"], errors="coerce"
    ).astype("Int64")
    out["sex"] = raw["SEXVAR"].map({1: "Male", 2: "Female"})
    # _RFHYPE6: 1=No, 2=Yes, 9=unknown/refused/missing.
    out["hypertension"] = raw["_RFHYPE6"].map({1: 0, 2: 1})
    # _MICHD: 1=reported MI or CHD, 2=did not report either; blank=missing.
    out["chd_or_mi_history"] = raw["_MICHD"].map({1: 1, 2: 0})
    # Preserve pregnancy-only and prediabetes as separate codebook categories.
    out["diabetes_status"] = raw["DIABETE4"].map(DIABETES)
    # _BMI5 has two implied decimal places; 9999 is a missing sentinel.
    bmi_raw = pd.to_numeric(raw["_BMI5"], errors="coerce").replace(9999, pd.NA)
    out["bmi"] = bmi_raw.astype("Float64") / 100.0
    out["smoking_status"] = raw["_SMOKER3"].map(SMOKING)
    out["physical_activity_30d"] = raw["EXERANY2"].map({1: 1, 2: 0})
    out["survey_weight"] = pd.to_numeric(raw["_LLCPWT"], errors="coerce")
    out["survey_stratum"] = pd.to_numeric(raw["_STSTR"], errors="coerce").astype("Int64")
    out["survey_psu"] = pd.to_numeric(raw["_PSU"], errors="coerce").astype("Int64")

    counts = {
        "yes": int((out["stroke_history"] == 1).sum()),
        "no": int((out["stroke_history"] == 0).sum()),
    }
    excluded = {
        str(k): int(v) for k, v in raw_target_counts.items()
        if pd.isna(k) or int(k) not in (1, 2)
    }
    return out.reset_index(drop=True), {
        "rows_seen": int(len(chunk)),
        "rows_retained": int(valid_outcome.sum()),
        "retained_target_counts": counts,
        "excluded_outcome_counts": excluded,
    }


def prepare(input_path: Path, output_dir: Path, chunksize: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "brfss2025_stroke_history_analysis.csv.gz"
    profile_file = output_dir / "preparation_profile.json"
    if output_file.exists():
        output_file.unlink()

    rows_seen = 0
    rows_retained = 0
    target_counts: Counter[int] = Counter()
    excluded_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    weighted_targets: Counter[int] = Counter()
    category_counts: dict[str, Counter[str]] = {
        col: Counter() for col in CATEGORY_FIELDS
    }
    strata: set[int] = set()
    psus: set[int] = set()
    bmi_min: float | None = None
    bmi_max: float | None = None
    age_min: int | None = None
    age_max: int | None = None
    header_written = False

    with tempfile.TemporaryDirectory(prefix="brfss2025-prepare-") as temp_name:
        xpt_path, source = get_xpt(input_path, Path(temp_name))
        reader = pd.read_sas(
            xpt_path, format="xport", iterator=True,
            chunksize=chunksize, encoding="latin1",
        )
        for chunk in reader:
            absent = set(SOURCE_FIELDS) - set(chunk.columns)
            if absent:
                raise ValueError("Missing source fields: " + ", ".join(sorted(absent)))
            rows_seen += len(chunk)
            outcome = chunk["CVDSTRK3"]
            for code, count in outcome.value_counts(dropna=False).items():
                if pd.isna(code):
                    excluded_counts["blank_or_missing"] += int(count)
                elif int(code) not in (1, 2):
                    excluded_counts[str(int(code))] += int(count)

            clean, _ = make_analysis_frame(chunk)
            if clean.empty:
                continue
            rows_retained += len(clean)
            target_counts.update(clean["stroke_history"].value_counts().to_dict())
            for col in clean.columns:
                missing_counts[col] += int(clean[col].isna().sum())
            for col in CATEGORY_FIELDS:
                category_counts[col].update(
                    clean[col].fillna("Missing").astype(str).value_counts().to_dict()
                )
            if clean["bmi"].notna().any():
                lo, hi = float(clean["bmi"].min()), float(clean["bmi"].max())
                bmi_min = lo if bmi_min is None else min(bmi_min, lo)
                bmi_max = hi if bmi_max is None else max(bmi_max, hi)
            if clean["age_years_capped_at_80"].notna().any():
                lo = int(clean["age_years_capped_at_80"].min())
                hi = int(clean["age_years_capped_at_80"].max())
                age_min = lo if age_min is None else min(age_min, lo)
                age_max = hi if age_max is None else max(age_max, hi)
            strata.update(int(v) for v in clean["survey_stratum"].dropna().unique())
            psus.update(int(v) for v in clean["survey_psu"].dropna().unique())
            for code, value in clean.groupby("stroke_history")["survey_weight"].sum().items():
                weighted_targets[int(code)] += float(value)

            # Reopen in append mode for each chunk; gzip supports concatenated members.
            with gzip.open(
                output_file, "at" if header_written else "wt",
                encoding="utf-8", newline="",
            ) as handle:
                clean.to_csv(handle, index=False, header=not header_written, na_rep="")
            header_written = True

    if not rows_retained:
        raise ValueError("No respondents have a valid binary stroke-history outcome")
    yes_n, no_n = int(target_counts[1]), int(target_counts[0])
    weight_total = sum(weighted_targets.values())
    profile = {
        "dataset": "CDC BRFSS 2025 LLCP",
        "source": source,
        "rows_read": rows_seen,
        "rows_retained_for_binary_outcome": rows_retained,
        "rows_excluded_for_nonbinary_or_missing_outcome": rows_seen - rows_retained,
        "outcome": {
            "source_variable": "CVDSTRK3",
            "question": "(Ever told) (you had) a stroke.",
            "mapping": {"1": 1, "2": 0},
            "yes_count": yes_n,
            "no_count": no_n,
            "unweighted_yes_percent_among_valid_yes_no": round(100 * yes_n / rows_retained, 4),
            "weighted_yes_percent_among_valid_yes_no": round(
                100 * weighted_targets[1] / weight_total, 4
            ) if weight_total else None,
            "no_to_yes_ratio": round(no_n / yes_n, 3) if yes_n else None,
            "excluded_codes_and_counts": dict(excluded_counts),
        },
        "predictor_missingness_in_retained_cohort": {
            col: {
                "missing_n": int(missing_counts[col]),
                "missing_percent": round(100 * missing_counts[col] / rows_retained, 3),
            }
            for col in [
                "sex", "age_years_capped_at_80", "hypertension",
                "chd_or_mi_history", "diabetes_status", "bmi",
                "smoking_status", "physical_activity_30d",
            ]
        },
        "categorical_distributions_in_retained_cohort": {
            col: dict(counter) for col, counter in category_counts.items()
        },
        "ranges": {
            "age_min": age_min,
            "age_max": age_max,
            "age_80_note": "80 is top-coded and represents age 80 or older.",
            "bmi_min": round(bmi_min, 2) if bmi_min is not None else None,
            "bmi_max": round(bmi_max, 2) if bmi_max is not None else None,
        },
        "survey_design": {
            "weight_variable": "_LLCPWT",
            "strata_variable": "_STSTR",
            "psu_variable": "_PSU",
            "unique_strata_in_retained_cohort": len(strata),
            "unique_psu_values_in_retained_cohort": len(psus),
            "weight_preserved_not_used_as_predictor": True,
        },
        "output_file": output_file.name,
        "output_size_bytes": output_file.stat().st_size,
        "output_sha256": sha256_file(output_file),
        "preprocessing_policy": {
            "unknown_outcome": "CVDSTRK3 codes 7, 9 and blank are excluded; only codes 1 and 2 are retained.",
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
    print(f"Rows read: {profile['rows_read']:,}")
    print(f"Rows retained: {profile['rows_retained_for_binary_outcome']:,}")
    print(f"Rows excluded: {profile['rows_excluded_for_nonbinary_or_missing_outcome']:,}")
    print(
        "Weighted Yes share among valid Yes/No outcomes:",
        profile["outcome"]["weighted_yes_percent_among_valid_yes_no"], "%",
    )
    print("Output:", args.output_dir / profile["output_file"])
    print("Profile:", args.output_dir / "preparation_profile.json")


if __name__ == "__main__":
    main()
