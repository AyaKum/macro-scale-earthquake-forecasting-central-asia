"""
src/features/tectonic.py
------------------------
GEM Global Active Faults download and kinematic feature extraction
(slip rate, dip, rake, fault proximity).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import geopandas as gpd
import requests

logger = logging.getLogger(__name__)


def load_gem_faults(cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """Download (or load cached) GEM Global Active Faults clipped to the study region.

    Parameters
    ----------
    cfg : dict
        Full config dict (reads ``paths.gem_faults_url``, ``paths.gem_faults_local``,
        ``grid.lon_range``, ``grid.lat_range``).

    Returns
    -------
    gpd.GeoDataFrame
        Fault linestrings within the study bounding box.
    """
    local = cfg["paths"]["gem_faults_local"]
    url   = cfg["paths"]["gem_faults_url"]

    if not os.path.exists(local):
        logger.info("Downloading GEM Active Faults → %s", local)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        with open(local, "wb") as fh:
            fh.write(resp.content)

    faults = gpd.read_file(local)
    lon_min, lon_max = cfg["grid"]["lon_range"]
    lat_min, lat_max = cfg["grid"]["lat_range"]
    faults = faults.cx[lon_min:lon_max, lat_min:lat_max].reset_index(drop=True)
    logger.info("GEM faults loaded: %d segments in study region.", len(faults))
    return faults


def _parse_gem_value(val: object) -> float:
    """Parse GEM tuple-style field values such as ``(1.5, 0.5, 3.0)``."""
    try:
        if isinstance(val, str) and "(" in val:
            return float(val.replace("(", "").replace(")", "").split(",")[0])
        return float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def compute_fault_features(
    gdf_cells: gpd.GeoDataFrame,
    faults: gpd.GeoDataFrame,
    utm_epsg: int = 32642,
) -> gpd.GeoDataFrame:
    """Compute nearest-fault distance and kinematic attributes for each grid cell.

    Uses a nearest-neighbour spatial join in metric CRS (UTM Zone 42N by default).

    Parameters
    ----------
    gdf_cells : gpd.GeoDataFrame
        Cell catalogue (output of :func:`src.features.spatial_grid.assign_regimes`).
    faults : gpd.GeoDataFrame
        GEM fault dataset from :func:`load_gem_faults`.
    utm_epsg : int
        EPSG code for a metric projection covering the study area.

    Returns
    -------
    gpd.GeoDataFrame
        ``gdf_cells`` with added columns:
        ``dist_to_fault_km``, ``fault_slip_rate``, ``fault_dip``, ``fault_rake``.
    """
    logger.info("Computing fault proximity features (EPSG:%d)…", utm_epsg)
    tectonic_cols = ["net_slip_rate", "average_dip", "average_rake", "geometry"]

    gdf_m    = gdf_cells.to_crs(epsg=utm_epsg)
    faults_m = faults[tectonic_cols].to_crs(epsg=utm_epsg)

    gdf_m = gpd.sjoin_nearest(gdf_m, faults_m, distance_col="dist_to_fault_m")
    gdf_m["fault_slip_rate"]  = gdf_m["net_slip_rate"].apply(_parse_gem_value)
    gdf_m["fault_dip"]        = gdf_m["average_dip"].apply(_parse_gem_value)
    gdf_m["fault_rake"]       = gdf_m["average_rake"].apply(_parse_gem_value)
    gdf_m["dist_to_fault_km"] = gdf_m["dist_to_fault_m"] / 1_000.0
    gdf_m = gdf_m.drop_duplicates(subset=["cell_id"])

    feat_cols = ["cell_id", "dist_to_fault_km", "fault_slip_rate", "fault_dip", "fault_rake"]
    gdf_cells = gdf_cells.merge(gdf_m[feat_cols], on="cell_id", how="left")
    gdf_cells[["fault_slip_rate", "fault_dip", "fault_rake"]] = (
        gdf_cells[["fault_slip_rate", "fault_dip", "fault_rake"]].fillna(0.0)
    )
    gdf_cells["dist_to_fault_km"] = gdf_cells["dist_to_fault_km"].fillna(100.0)
    return gdf_cells
