"""
src/evaluation/regime_eval.py
------------------------------
Spatial generalisability disaggregation by tectonic regime.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def evaluate_by_regime(
    y_test: pd.Series,
    preds_dict: dict[str, np.ndarray],
    regimes_test: pd.Series,
) -> pd.DataFrame:
    """Compute PR-AUC and ROC-AUC for each tectonic regime and model.

    Regimes with no positive events or only positive events are skipped
    (AUC metrics are undefined in those cases).

    Parameters
    ----------
    y_test : pd.Series
        Binary ground-truth labels for the test period.
    preds_dict : dict[str, np.ndarray]
        ``{model_name: probability_array}`` from the final robustness seed.
    regimes_test : pd.Series
        Tectonic regime label aligned with ``y_test``.

    Returns
    -------
    pd.DataFrame
        Columns: Regime, Model, Events, PR-AUC, ROC-AUC.
    """
    rows = []
    for regime in sorted(regimes_test.unique()):
        mask     = (regimes_test == regime).values
        y_reg    = y_test.values[mask]
        n_events = int(y_reg.sum())

        if n_events == 0 or n_events == len(y_reg):
            logger.warning(
                "Regime '%s': skipped (n_pos=%d / n=%d).", regime, n_events, len(y_reg)
            )
            continue

        for name, preds in preds_dict.items():
            p_reg = preds[mask]
            rows.append({
                "Regime":  regime,
                "Model":   name,
                "Events":  n_events,
                "PR-AUC":  round(float(average_precision_score(y_reg, p_reg)), 4),
                "ROC-AUC": round(float(roc_auc_score(y_reg, p_reg)), 4),
            })

    df = pd.DataFrame(rows)
    logger.info("\n%s", df.to_string(index=False))
    return df
