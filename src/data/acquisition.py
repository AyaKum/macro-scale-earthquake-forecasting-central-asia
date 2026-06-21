"""
src/data/acquisition.py
-----------------------
USGS FDSN seismic catalog download for Central Asia.
"""
from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

USGS_FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_catalog(cfg: dict[str, Any]) -> pd.DataFrame:
    """Download the Central Asia seismic catalog from USGS FDSN.

    Parameters
    ----------
    cfg : dict
        The ``data`` block from ``config.yaml``.

    Returns
    -------
    pd.DataFrame
        Events sorted by UTC time with columns: time, latitude, longitude, depth, mag.
    """
    params = {
        "format":       "csv",
        "starttime":    cfg["starttime"],
        "endtime":      cfg["endtime"],
        "minlatitude":  cfg["bbox"]["min_lat"],
        "maxlatitude":  cfg["bbox"]["max_lat"],
        "minlongitude": cfg["bbox"]["min_lon"],
        "maxlongitude": cfg["bbox"]["max_lon"],
        "minmagnitude": cfg["min_magnitude"],
        "eventtype":    "earthquake",
        "orderby":      "time",
    }
    logger.info(
        "Fetching USGS catalog  %s → %s  (M≥%.1f)",
        cfg["starttime"], cfg["endtime"], cfg["min_magnitude"],
    )
    resp = requests.get(USGS_FDSN_URL, params=params, timeout=180)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    logger.info("Retrieved %d events.", len(df))
    return df
