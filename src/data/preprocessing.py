"""
src/data/preprocessing.py
--------------------------
Weekly cell-×-week panel construction, target engineering,
and temporal physics-informed features.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TECT_FEAT_COLS = ["dist_to_fault_km", "fault_slip_rate", "fault_dip", "fault_rake"]


def build_weekly_dataframe(
    df_raw: pd.DataFrame,
    gdf_cells: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Expand the event catalogue into a balanced cell × week panel.

    Every (cell, week) pair between the first and last observed week is
    represented.  Quiescent cell-weeks are filled with ``eq_count = 0``.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Catalogue with columns ``cell_id``, ``time``, ``mag``
        (output of :func:`src.features.spatial_grid.build_grid`).
    gdf_cells : gpd.GeoDataFrame
        Cell catalogue.

    Returns
    -------
    pd.DataFrame
        Panel with columns: ``cell_id``, ``year_week``, ``eq_count``, ``max_mag``.
    """
    df = df_raw.copy()
    df["year_week"] = df["time"].dt.to_period("W").dt.start_time

    df_weekly = (
        df.groupby(["cell_id", "year_week"])
        .agg(eq_count=("mag", "count"), max_mag=("mag", "max"))
        .reset_index()
    )

    all_weeks = pd.date_range(
        start=df_weekly["year_week"].min(),
        end=df_weekly["year_week"].max(),
        freq="W-MON",
    )
    multi_idx = pd.MultiIndex.from_product(
        [gdf_cells["cell_id"].unique(), all_weeks],
        names=["cell_id", "year_week"],
    )
    df_full = (
        pd.DataFrame(index=multi_idx)
        .reset_index()
        .merge(df_weekly, on=["cell_id", "year_week"], how="left")
        .fillna({"eq_count": 0, "max_mag": 0})
        .sort_values(["cell_id", "year_week"])
        .reset_index(drop=True)
    )
    logger.info(
        "Panel: %d rows  (%d cells × %d weeks).",
        len(df_full), gdf_cells["cell_id"].nunique(), len(all_weeks),
    )
    return df_full


def compute_target(df_full: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """Add a binary forecast target: any M≥3.0 event in the next ``horizon`` weeks.

    Parameters
    ----------
    df_full : pd.DataFrame
        Weekly panel from :func:`build_weekly_dataframe`.
    horizon : int
        Forecast window in weeks (default: 4).

    Returns
    -------
    pd.DataFrame
        Input with additional column ``target_4w_bin``.
    """
    df_full = df_full.copy()
    future_eq = sum(
        df_full.groupby("cell_id")["eq_count"].shift(-k).fillna(0)
        for k in range(1, horizon + 1)
    )
    df_full["target_4w_bin"] = (future_eq > 0).astype(int)
    logger.info("Target prevalence: %.2f%%.", df_full["target_4w_bin"].mean() * 100)
    return df_full


def compute_temporal_features(df_full: pd.DataFrame) -> pd.DataFrame:
    """Add physics-informed temporal seismicity proxies per cell.

    Features added:

    - ``rolling_4w_count`` — 4-week rolling event count.
    - ``seismic_energy``   — energy proxy 10^{1.5 M} for the largest event.
    - ``omori_local``      — EWMA of seismic energy (Omori-type decay, half-life 2 weeks).

    Parameters
    ----------
    df_full : pd.DataFrame
        Panel from :func:`compute_target`.

    Returns
    -------
    pd.DataFrame
        Input with additional temporal feature columns.
    """
    df_full = df_full.copy()
    df_full["rolling_4w_count"] = (
        df_full.groupby("cell_id")["eq_count"]
        .transform(lambda x: x.rolling(4, min_periods=1).sum())
    )
    df_full["seismic_energy"] = np.where(
        df_full["max_mag"] > 0, 10 ** (1.5 * df_full["max_mag"]), 0.0
    )
    df_full["omori_local"] = (
        df_full.groupby("cell_id")["seismic_energy"]
        .transform(lambda x: x.ewm(halflife=2, ignore_na=True).mean())
    )
    return df_full


def merge_tectonic_context(
    df_full: pd.DataFrame,
    gdf_cells: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Merge static tectonic attributes from the cell catalogue into the panel.

    Parameters
    ----------
    df_full : pd.DataFrame
        Weekly panel.
    gdf_cells : gpd.GeoDataFrame
        Cell catalogue with ``tect_reg`` and fault-proximity columns.

    Returns
    -------
    pd.DataFrame
        Panel with ``tect_reg``, ``dist_to_fault_km``, ``fault_slip_rate``,
        ``fault_dip``, ``fault_rake``.
    """
    merge_cols = ["cell_id", "tect_reg"] + TECT_FEAT_COLS
    return df_full.merge(gdf_cells[merge_cols], on="cell_id", how="left")
