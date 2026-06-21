"""
src/features/normalization.py
------------------------------
Per-cell log-odds baseline initialisation and regime-stratified Z-score
normalisation. Statistics are estimated on the training period only to
prevent data leakage into the test set.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Raw columns to normalise
NORM_COLS: list[str] = [
    "omori_local",
    "rolling_4w_count",
    "dist_to_fault_km",
    "fault_slip_rate",
    "fault_dip",
    "fault_rake",
]

# Final feature vector fed to all models
FEATURE_COLS: list[str] = [f"{c}_z" for c in NORM_COLS] + ["omori_x_prob", "hist_prob"]


def compute_log_odds_baseline(
    df_full: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    """Compute per-cell historical positive rate and its log-odds.

    Estimated from the training period only to prevent leakage.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full weekly panel (train + test).
    train_mask : pd.Series[bool]
        Boolean mask selecting the training rows.

    Returns
    -------
    pd.DataFrame
        Input with added columns ``hist_prob`` and ``log_odds_base``.
    """
    hist = (
        df_full[train_mask]
        .groupby("cell_id")
        .apply(lambda x: (x["target_4w_bin"] > 0).mean(), include_groups=False)
        .reset_index(name="hist_prob")
    )
    hist["hist_prob"] = np.clip(hist["hist_prob"], 1e-5, 1 - 1e-5)
    df_full = df_full.merge(hist, on="cell_id", how="left").fillna({"hist_prob": 1e-5})
    df_full["log_odds_base"] = np.log(df_full["hist_prob"] / (1 - df_full["hist_prob"]))
    logger.info("Log-odds baseline computed for %d cells.", hist["cell_id"].nunique())
    return df_full


def regime_zscore_normalize(
    df_full: pd.DataFrame,
    train_mask: pd.Series,
    norm_cols: list[str] | None = None,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Apply regime-stratified Z-score normalisation.

    Statistics (mean, std) are fitted within each tectonic regime using
    training data only, then applied to the full panel (train + test).

    Parameters
    ----------
    df_full : pd.DataFrame
        Panel with ``tect_reg`` and all columns in ``norm_cols``.
    train_mask : pd.Series[bool]
        Boolean mask for training rows.
    norm_cols : list[str] | None
        Columns to normalise.  Defaults to :data:`NORM_COLS`.
    epsilon : float
        Numerical stability constant added to the standard deviation.

    Returns
    -------
    pd.DataFrame
        Input with added ``<col>_z`` columns and the ``omori_x_prob``
        interaction term (``omori_local_z × hist_prob``).
    """
    if norm_cols is None:
        norm_cols = NORM_COLS

    regime_stats = (
        df_full[train_mask]
        .groupby("tect_reg")[norm_cols]
        .agg(["mean", "std"])
    )
    regime_stats.columns = ["_".join(c) for c in regime_stats.columns]
    df_full = df_full.merge(regime_stats.reset_index(), on="tect_reg", how="left")

    for col in norm_cols:
        df_full[f"{col}_z"] = (
            (df_full[col] - df_full[f"{col}_mean"])
            / (df_full[f"{col}_std"] + epsilon)
        )

    df_full["omori_x_prob"] = df_full["omori_local_z"] * df_full["hist_prob"]
    return df_full
