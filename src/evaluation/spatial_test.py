"""
src/evaluation/spatial_test.py
-------------------------------
pyCSEP binary spatial test (Bayano et al. 2022).

Asks: does each model assign high probability to the cells where
earthquakes actually occurred?

Reference
---------
Bayona, J.A., Savran, W., Strader, A., Marzocchi, W., & Werner, M.J. (2022).
Prospective evaluation of multiplicative hybrid earthquake forecasting models
in California. *Geophysical Journal International*, 229(3), 1736–1753.
https://doi.org/10.1093/gji/ggac018
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from csep.core import binomial_evaluations as binom

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal duck-typed wrappers required by binary_spatial_test
# ──────────────────────────────────────────────────────────────────────────────

class _BinaryForecast:
    """Minimal forecast object compatible with :func:`binom.binary_spatial_test`."""

    def __init__(self, rates: np.ndarray, name: str) -> None:
        self._rates     = rates.astype(float)
        self.magnitudes = np.array([3.0, 10.0])   # single M≥3.0 bin
        self.name       = name

    def spatial_counts(self) -> np.ndarray:
        return self._rates


class _BinaryCatalog:
    """Minimal observed catalog compatible with :func:`binom.binary_spatial_test`."""

    def __init__(self, counts: np.ndarray, name: str = "ComCat 2019–2024") -> None:
        self._counts = counts.astype(float)
        self.name    = name

    def spatial_counts(self) -> np.ndarray:
        return self._counts


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_cell_rates(
    pred_array: np.ndarray,
    cell_ids: pd.Series,
    sorted_cells: list[str],
    floor: float = 1e-4,
) -> np.ndarray:
    """Aggregate per-(cell, week) probabilities to per-cell expected active weeks.

    The binary spatial test cares only about relative spatial distribution,
    so summing weekly P(event | cell, week) across all test weeks is a valid
    aggregation even though the model predicts 4-week horizons.
    """
    df_tmp = pd.DataFrame({"cell_id": cell_ids.values, "p": pred_array})
    agg    = df_tmp.groupby("cell_id")["p"].sum()
    return np.array([max(float(agg.get(c, 0.0)), floor) for c in sorted_cells])


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────

def run_binary_spatial_test(
    df_full: pd.DataFrame,
    test_mask: pd.Series,
    model_preds: dict[str, np.ndarray],
    num_simulations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the pyCSEP binary spatial test for each model.

    Interpretation
    --------------
    δ > 0.05  →  spatial pattern consistent with observations at α = 0.05.
    δ ≈ 0.50  →  model concentrates probability in the right cells.
    δ < 0.05  →  model's spatial distribution is inconsistent with observations.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full weekly panel (train + test).
    test_mask : pd.Series[bool]
        Boolean selector for test-period rows.
    model_preds : dict[str, np.ndarray]
        ``{model_name: probability_array}`` aligned with the test rows.
    num_simulations : int
        Monte Carlo simulations for the null distribution (default: 1000).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: Model, delta, observed_ll, result.
    """
    df_test      = df_full[test_mask]
    sorted_cells = sorted(df_test["cell_id"].unique())

    obs_agg  = df_test.groupby("cell_id")["eq_count"].sum()
    obs_arr  = np.array([float(obs_agg.get(c, 0.0)) for c in sorted_cells])
    observed = _BinaryCatalog(obs_arr)

    rows = []
    for name, preds in model_preds.items():
        rates = _to_cell_rates(preds, df_test["cell_id"], sorted_cells)
        fc    = _BinaryForecast(rates, name)
        res   = binom.binary_spatial_test(
            fc, observed, num_simulations=num_simulations, seed=seed
        )
        delta  = float(res.quantile)
        obs_ll = float(res.observed_statistic)
        result = "consistent" if delta > 0.05 else "inconsistent"
        rows.append({"Model": name, "delta": delta, "observed_ll": obs_ll, "result": result})
        logger.info("%-18s  δ=%.4f  LL=%.2f  [%s]", name, delta, obs_ll, result)

    return pd.DataFrame(rows)
