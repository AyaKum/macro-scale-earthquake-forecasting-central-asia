"""
src/features/spatial_grid.py
-----------------------------
1-degree spatial grid construction and EMCA tectonic regime assignment.
"""
from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_grid(
    df_raw: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Assign each event to a 1° × 1° cell and build the cell catalogue.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw seismic catalogue from :func:`src.data.acquisition.fetch_catalog`.
    cfg : dict
        The ``grid`` block from ``config.yaml``.

    Returns
    -------
    df_raw : pd.DataFrame
        Catalogue with added columns: ``lat_grid``, ``lon_grid``, ``cell_id``.
    gdf_cells : gpd.GeoDataFrame
        One row per unique cell with point geometry at the cell's SW corner.
    """
    lat_bins = np.arange(cfg["lat_range"][0], cfg["lat_range"][1], cfg["lat_step"])
    lon_bins = np.arange(cfg["lon_range"][0], cfg["lon_range"][1], cfg["lon_step"])

    df_raw = df_raw.copy()
    df_raw["lat_grid"] = pd.cut(df_raw["latitude"],  bins=lat_bins, labels=lat_bins[:-1])
    df_raw["lon_grid"] = pd.cut(df_raw["longitude"], bins=lon_bins, labels=lon_bins[:-1])
    df_raw = df_raw.dropna(subset=["lat_grid", "lon_grid"])
    df_raw["cell_id"] = (
        df_raw["lat_grid"].astype(str) + "_" + df_raw["lon_grid"].astype(str)
    )

    unique_cells = df_raw[["cell_id", "lat_grid", "lon_grid"]].drop_duplicates()
    gdf_cells = gpd.GeoDataFrame(
        unique_cells,
        geometry=gpd.points_from_xy(unique_cells["lon_grid"], unique_cells["lat_grid"]),
        crs="EPSG:4326",
    ).reset_index(drop=True)

    logger.info(
        "Grid: %d active cells  (%.0f°–%.0f° N, %.0f°–%.0f° E).",
        len(gdf_cells),
        cfg["lat_range"][0], cfg["lat_range"][1],
        cfg["lon_range"][0], cfg["lon_range"][1],
    )
    return df_raw, gdf_cells


def assign_regimes(
    gdf_cells: gpd.GeoDataFrame,
    emca_path: str,
    default_regime: str = "Stable Continental Region",
) -> gpd.GeoDataFrame:
    """Attach EMCA v1.0 tectonic regime labels to each grid cell.

    Parameters
    ----------
    gdf_cells : gpd.GeoDataFrame
        Cell catalogue from :func:`build_grid`.
    emca_path : str
        Path to ``EMCA_seismozonesv1.0.shp``.
    default_regime : str
        Fallback label when a cell falls outside all EMCA polygons.

    Returns
    -------
    gpd.GeoDataFrame
        ``gdf_cells`` with an additional ``tect_reg`` column.
    """
    try:
        emca = gpd.read_file(emca_path)[["tect_reg", "geometry"]]
        gdf_cells = gpd.sjoin(gdf_cells, emca, how="left", predicate="within")
        gdf_cells["tect_reg"] = gdf_cells["tect_reg"].fillna(default_regime)
        if "index_right" in gdf_cells.columns:
            gdf_cells = gdf_cells.drop(columns=["index_right"])
        logger.info(
            "EMCA regimes assigned:\n%s",
            gdf_cells["tect_reg"].value_counts().to_string(),
        )
    except Exception as exc:
        logger.warning(
            "EMCA shapefile not found (%s). All cells → '%s'.", exc, default_regime
        )
        gdf_cells["tect_reg"] = default_regime

    return gdf_cells
