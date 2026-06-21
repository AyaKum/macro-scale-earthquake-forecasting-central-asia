"""
src/evaluation/metrics.py
--------------------------
Publication metrics: PR-AUC, ROC-AUC, Precision@k, and a summary table.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: float = 0.01,
) -> float:
    """Precision among the top-k% highest-scored cells.

    Parameters
    ----------
    y_true : np.ndarray of {0, 1}
    y_score : np.ndarray of float
    k : float
        Top fraction to evaluate (default: 0.01 → top 1%).

    Returns
    -------
    float
    """
    n_top   = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_score)[-n_top:]
    return float(np.mean(np.asarray(y_true)[top_idx]))


def summarize_seed_results(
    results: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    """Aggregate per-seed metrics into a publication-ready summary table.

    Parameters
    ----------
    results : dict
        ``{model_name: {"pr": [...], "roc": [...]}}`` from
        :func:`src.models.train.run_robustness_seeds`.

    Returns
    -------
    pd.DataFrame
        Index = model name; columns = PR-AUC mean/std, ROC-AUC mean/std.
    """
    rows = []
    for model, scores in results.items():
        pr_mean  = np.mean(scores["pr"])
        pr_std   = np.std(scores["pr"])
        roc_mean = np.mean(scores["roc"])
        roc_std  = np.std(scores["roc"])
        rows.append({
            "Model":       model,
            "PR-AUC":      round(pr_mean, 4),
            "PR-AUC std":  round(pr_std,  4),
            "ROC-AUC":     round(roc_mean, 4),
            "ROC-AUC std": round(roc_std,  4),
        })

    df = pd.DataFrame(rows).set_index("Model")
    logger.info("\n%s", df.to_string())
    return df
