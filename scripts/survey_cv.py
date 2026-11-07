"""Stratum-aware fold assignment utilities for BRFSS respondent-level CV.

Fold allocation is an SRS-style random partition performed separately within
each official BRFSS stratum. Outcome labels are not used to assign folds.
"""
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
    """Assign every row to one fold, balancing counts within each stratum.

    Respondents are randomly partitioned separately inside each _STSTR
    stratum. _PSU is checked as the design-unit identifier nested in stratum.
    The target is deliberately not read or used when assigning folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["survey_stratum"].isna().any() or frame["survey_psu"].isna().any():
        raise ValueError("Survey stratum and PSU fields must be present for every row")
    if frame.duplicated(["survey_stratum", "survey_psu"]).any():
        raise ValueError("The (survey_stratum, survey_psu) design-unit key is not unique")

    rng = np.random.default_rng(random_state)
    folds = np.full(len(frame), -1, dtype=np.int16)
    strata = frame["survey_stratum"].to_numpy()

    for stratum in pd.unique(strata):
        positions = np.flatnonzero(strata == stratum)
        shuffled = rng.permutation(positions)
        offset = int(rng.integers(0, n_splits))
        assignment = (np.arange(len(shuffled)) + offset) % n_splits
        folds[shuffled] = assignment.astype(np.int16)

    if (folds < 0).any():
        raise RuntimeError("Some respondents were not assigned a fold")
    return folds


def fold_adjusted_weights(
    frame: pd.DataFrame,
    folds: np.ndarray,
    *,
    heldout_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return training and held-out weights adjusted for fold assignment.

    For each stratum h, p_h is the realised fraction assigned to the held-out
    fold. Fold-adjusted weights are the final BRFSS weight divided by 1-p_h
    for training records and by p_h for held-out records. The arrays contain
    zeros for rows outside the corresponding partition.
    """
    if len(frame) != len(folds):
        raise ValueError("folds must have one entry for every data row")
    if heldout_fold not in set(np.asarray(folds).tolist()):
        raise ValueError("heldout_fold is not present in folds")

    fold_series = pd.Series(np.asarray(folds), index=frame.index)
    stratum_sizes = frame.groupby("survey_stratum", sort=False).size()
    heldout_sizes = (
        frame.loc[fold_series.eq(heldout_fold)]
        .groupby("survey_stratum", sort=False)
        .size()
    )
    probability = (heldout_sizes / stratum_sizes).reindex(
        frame["survey_stratum"].to_numpy()
    ).fillna(0.0).to_numpy(dtype=float)
    base_weights = frame["survey_weight"].to_numpy(dtype=float)
    is_test = np.asarray(folds) == heldout_fold

    train_weights = np.zeros(len(frame), dtype=float)
    test_weights = np.zeros(len(frame), dtype=float)
    train_mask = ~is_test
    if np.any(train_mask & (probability >= 1)):
        raise RuntimeError("A training respondent is in a stratum wholly held out")
    if np.any(is_test & (probability <= 0)):
        raise RuntimeError("A test respondent has zero fold-inclusion probability")

    train_weights[train_mask] = base_weights[train_mask] / (1.0 - probability[train_mask])
    test_weights[is_test] = base_weights[is_test] / probability[is_test]
    return train_weights, test_weights
