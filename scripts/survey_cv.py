"""Stratum-aware fold assignment utilities for BRFSS respondent-level CV."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = {"survey_stratum", "survey_psu", "survey_weight"}


def assign_survey_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Assign respondent-level PSU units to folds separately within _STSTR.

    In the 2025 LLCP file, _PSU is the annual sequence number and each
    (survey_stratum, survey_psu) pair identifies one respondent. The target
    is deliberately not inspected when assigning folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[["survey_stratum", "survey_psu", "survey_weight"]].isna().any().any():
        raise ValueError("Stratum, PSU, and survey weight must be present")
    weights = frame["survey_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Survey weights must be finite and positive")
    if frame.duplicated(["survey_stratum", "survey_psu"]).any():
        raise ValueError("The (survey_stratum, survey_psu) design-unit key is not unique")

    rng = np.random.default_rng(random_state)
    folds = np.full(len(frame), -1, dtype=np.int16)
    strata = frame["survey_stratum"].to_numpy()

    for stratum in pd.unique(strata):
        positions = np.flatnonzero(strata == stratum)
        shuffled = rng.permutation(positions)
        offset = int(rng.integers(0, n_splits))
        folds[shuffled] = (np.arange(len(shuffled)) + offset) % n_splits

    if (folds < 0).any():
        raise RuntimeError("Some respondents were not assigned a fold")
    return folds


def fold_adjusted_weights(
    frame: pd.DataFrame,
    folds: np.ndarray,
    *,
    heldout_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust final BRFSS weights for the randomized second-phase fold split.

    The random starting offset makes each respondent's marginal probability
    of assignment to any specified fold exactly 1/K, even when a stratum has
    a singleton PSU or its realised fold counts are uneven. The training
    inclusion probability is (K-1)/K. Therefore use base_weight*K for the
    held-out fold and base_weight*K/(K-1) for training. These arrays are zero
    outside their respective partitions.
    """
    folds = np.asarray(folds)
    if folds.ndim != 1 or len(frame) != len(folds):
        raise ValueError("folds must be a one-dimensional array with one entry per row")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("At least two folds are required")
    if heldout_fold not in unique_folds:
        raise ValueError("heldout_fold is not present in folds")
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(REQUIRED - set(frame.columns))}")

    base_weights = frame["survey_weight"].to_numpy(dtype=float)
    if not np.isfinite(base_weights).all() or np.any(base_weights <= 0):
        raise ValueError("Survey weights must be finite and positive")

    k = len(unique_folds)
    is_test = folds == heldout_fold
    is_train = ~is_test
    train_weights = np.zeros(len(frame), dtype=float)
    test_weights = np.zeros(len(frame), dtype=float)
    train_weights[is_train] = base_weights[is_train] * k / (k - 1)
    test_weights[is_test] = base_weights[is_test] * k
    return train_weights, test_weights
