#!/usr/bin/env python3
"""Audit the CDC BRFSS 2025 LLCP XPT file without exporting respondent rows.

Accepts either the official ZIP archive or an extracted XPT file. Writes
aggregate JSON/CSV summaries only; it does not train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

VARIABLE_LABELS = {
    "CVDSTRK3": "Ever told by a health professional that the respondent had a stroke (outcome)",
    "_AGE80": "Age in years; the highest released value is 80",
    "SEXVAR": "Sex of respondent",
    "_RFHYPE6": "Calculated high blood pressure indicator",
    "_MICHD": "History of myocardial infarction or coronary heart disease indicator",
    "DIABETE4": "Self-reported diabetes status, including pregnancy-only and prediabetes responses",
    "_BMI5": "Calculated body mass index, stored as BMI multiplied by 100",
    "_SMOKER3": "Calculated four-level smoking status",
    "EXERANY2": "Any leisure-time physical activity or exercise in the past 30 days",
    "_LLCPWT": "Final landline and cellphone respondent weight",
    "_STSTR": "Sample design stratification variable",
    "_PSU": "Primary sampling unit",
}

TARGET_LABELS = {
    1: "Yes",
    2: "No",
    7: "Don't know / Not sure",
    9: "Refused",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_xpt(input_path: Path, temp_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Return a normalized XPT path and source-archive metadata."""
    metadata: dict[str, Any] = {
        "input_name": input_path.name,
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": sha256_file(input_path),
    }

    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as archive:
            members = [
                item for item in archive.infolist()
                if item.filename.rstrip().upper().endswith(".XPT")
            ]
            if len(members) != 1:
                raise ValueError(
                    "Expected exactly one .XPT file in the ZIP archive; "
                    f"found {len(members)}."
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"ZIP integrity check failed at member: {bad_member}")

            member = members[0]
            normalized_path = temp_dir / "LLCP2025.XPT"
            with archive.open(member) as source, normalized_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)

            metadata.update(
                {
                    "archive_member_name": member.filename,
                    "archive_member_size_bytes": member.file_size,
                    "zip_integrity_check": "passed",
                    "xpt_sha256": sha256_file(normalized_path),
                }
            )
            return normalized_path, metadata

    if input_path.suffix.lower() != ".xpt":
        raise ValueError("Input must be the CDC ZIP archive or an extracted .XPT file.")

    metadata.update(
        {
            "archive_member_name": None,
            "xpt_sha256": metadata["input_sha256"],
        }
    )
    return input_path, metadata


def audit_dataset(input_path: Path, output_dir: Path, chunksize: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="brfss2025-") as temp_name:
        xpt_path, source_metadata = prepare_xpt(input_path, Path(temp_name))
        reader = pd.read_sas(
            xpt_path,
            format="xport",
            iterator=True,
            chunksize=chunksize,
            encoding="latin1",
        )

        row_count = 0
        column_count: int | None = None
        present: set[str] = set()
        missing = Counter()
        non_missing = Counter()
        observed_min: dict[str, float] = {}
        observed_max: dict[str, float] = {}
        distinct_values: dict[str, set[float]] = {
            name: set()
            for name in VARIABLE_LABELS
            if name not in {"_LLCPWT", "_STSTR", "_PSU"}
        }
        target_counts: Counter[int] = Counter()
        target_weight: Counter[int] = Counter()
        strata: set[float] = set()
        psus: set[float] = set()

        expected = set(VARIABLE_LABELS)
        for chunk in reader:
            row_count += len(chunk)
            if column_count is None:
                column_count = len(chunk.columns)
                present = set(chunk.columns)
                absent = sorted(expected - present)
                if absent:
                    raise ValueError(
                        "Required audit variables are missing from the XPT file: "
                        + ", ".join(absent)
                    )

            for name in VARIABLE_LABELS:
                series = chunk[name]
                missing[name] += int(series.isna().sum())
                non_missing[name] += int(series.notna().sum())

                if name in {"_PSU", "_STSTR"}:
                    values = series.dropna().unique().tolist()
                    (psus if name == "_PSU" else strata).update(
                        float(value) for value in values
                    )
                    continue

                if name == "_LLCPWT":
                    continue

                numeric = series.dropna()
                if name == "_BMI5":
                    numeric = numeric / 100.0
                if not numeric.empty:
                    low = float(numeric.min())
                    high = float(numeric.max())
                    observed_min[name] = min(observed_min.get(name, low), low)
                    observed_max[name] = max(observed_max.get(name, high), high)
                    if name in distinct_values:
                        distinct_values[name].update(
                            float(value) for value in series.dropna().unique().tolist()
                        )

            outcome = chunk["CVDSTRK3"]
            weight = chunk["_LLCPWT"]
            for code, number in outcome.value_counts(dropna=True).items():
                target_counts[int(code)] += int(number)

            valid_weight = outcome.notna() & weight.notna()
            weighted_by_code = (
                chunk.loc[valid_weight, ["CVDSTRK3", "_LLCPWT"]]
                .groupby("CVDSTRK3")["_LLCPWT"]
                .sum()
            )
            for code, value in weighted_by_code.items():
                target_weight[int(code)] += float(value)

        if row_count == 0 or column_count is None:
            raise ValueError("The XPT file contained no readable rows.")

        valid_binary_n = target_counts[1] + target_counts[2]
        yes_weight = target_weight[1]
        all_nonblank_target_weight = sum(target_weight.values())

        target_summary = {
            "question": "(Ever told) (you had) a stroke.",
            "codes": {
                str(code): {
                    "label": TARGET_LABELS.get(code, "Unmapped response code"),
                    "count": int(target_counts[code]),
                }
                for code in sorted(target_counts)
            },
            "blank_or_missing_count": int(missing["CVDSTRK3"]),
            "valid_yes_no_n": int(valid_binary_n),
            "unweighted_yes_percent_of_valid_yes_no": round(
                100 * target_counts[1] / valid_binary_n, 4
            ) if valid_binary_n else None,
            "weighted_yes_percent_of_nonblank_responses": round(
                100 * yes_weight / all_nonblank_target_weight, 4
            ) if all_nonblank_target_weight else None,
            "no_to_yes_class_ratio": round(
                target_counts[2] / target_counts[1], 3
            ) if target_counts[1] else None,
        }

        variables = []
        for name, label in VARIABLE_LABELS.items():
            variables.append(
                {
                    "variable": name,
                    "label": label,
                    "non_missing": int(non_missing[name]),
                    "missing": int(missing[name]),
                    "missing_percent": round(100 * missing[name] / row_count, 3),
                    "distinct_nonmissing_values": (
                        len(distinct_values.get(name, set())) or None
                    ),
                    "observed_min": observed_min.get(name),
                    "observed_max": observed_max.get(name),
                }
            )

        summary = {
            "dataset": "CDC BRFSS 2025 LLCP",
            "respondent_rows": int(row_count),
            "column_count": int(column_count),
            "audit_variables_present": sorted(expected & present),
            "audit_variable_count": len(expected & present),
            "unique_strata_count": len(strata),
            "unique_psu_count": len(psus),
            "source": source_metadata,
            "target": target_summary,
            "interpretation": (
                "CVDSTRK3 measures reported prior stroke history, not future stroke "
                "incidence. This audit is descriptive and does not train or validate "
                "a clinical prediction model."
            ),
        }

        (output_dir / "dataset_audit.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(variables).to_csv(
            output_dir / "variable_audit.csv",
            index=False,
        )
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/LLCP2025XPT.zip"),
        help="Path to the official ZIP archive or extracted LLCP2025.XPT file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated"),
        help="Directory for aggregate audit outputs; no respondent-level rows are written.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=25_000,
        help="Number of rows read per chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be a positive integer.")
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    summary = audit_dataset(args.input, args.output_dir, args.chunksize)
    target = summary["target"]
    print(f"Rows: {summary['respondent_rows']:,} | Columns: {summary['column_count']}")
    print(
        f"Audit fields present: {summary['audit_variable_count']}/"
        f"{len(VARIABLE_LABELS)}"
    )
    print(f"Stroke-history yes/no records: {target['valid_yes_no_n']:,}")
    print(
        "Unweighted yes among yes/no: "
        f"{target['unweighted_yes_percent_of_valid_yes_no']}%"
    )
    print(
        "Weighted yes among nonblank answers: "
        f"{target['weighted_yes_percent_of_nonblank_responses']}%"
    )
    print(f"Aggregate outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
