"""
src/models/ensemble.py
-----------------------
Two ensemble strategies:
  1. LGBM-CatBoost — equal-weight probability blend.
  2. LSTM-CatBoost — CatBoost meta-learner with Bi-LSTM scores as a feature.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import expit

logger = logging.getLogger(__name__)


def lgbm_catboost_blend(
    lgbm_preds: np.ndarray,
    cat_preds: np.ndarray,
    lgbm_weight: float = 0.5,
) -> np.ndarray:
    """Convex combination of LightGBM and CatBoost probability scores.

    Parameters
    ----------
    lgbm_preds : np.ndarray
        Probability scores from the LightGBM model.
    cat_preds : np.ndarray
        Probability scores from the CatBoost model.
    lgbm_weight : float
        Weight on LightGBM; CatBoost weight = 1 − lgbm_weight.

    Returns
    -------
    np.ndarray
        Blended probability scores.
    """
    return lgbm_weight * lgbm_preds + (1.0 - lgbm_weight) * cat_preds


def lstm_catboost_meta(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    init_test: np.ndarray,
    lstm_train_preds: np.ndarray,
    lstm_test_preds: np.ndarray,
    cfg: dict,
    seed: int,
) -> np.ndarray:
    """Train a CatBoost meta-learner augmented with Bi-LSTM temporal risk scores.

    The Bi-LSTM predictions on the training set serve as a temporal risk proxy
    that CatBoost learns to weight alongside the original tabular features.

    Parameters
    ----------
    X_train, y_train, init_train : training features, labels, log-odds baseline.
    X_test, y_test, init_test    : test features, labels, log-odds baseline.
    lstm_train_preds : np.ndarray
        Bi-LSTM probability scores on the **training** set.
    lstm_test_preds : np.ndarray
        Bi-LSTM probability scores on the **test** set.
    cfg : dict
        The ``model.catboost`` block from ``config.yaml``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Test-set probability scores from the LSTM-CatBoost meta-learner.
    """
    X_train_meta = X_train.copy()
    X_test_meta  = X_test.copy()
    X_train_meta["lstm_temporal_risk"] = lstm_train_preds
    X_test_meta["lstm_temporal_risk"]  = lstm_test_preds

    model = CatBoostClassifier(
        iterations=cfg["iterations"],
        learning_rate=cfg["learning_rate"],
        depth=cfg["depth"],
        logging_level="Silent",
        random_seed=seed,
    )
    model.fit(Pool(X_train_meta, y_train, baseline=init_train))
    raw = model.predict(
        Pool(X_test_meta, y_test, baseline=init_test),
        prediction_type="RawFormulaVal",
    )
    logger.debug("LSTM-CatBoost meta-learner trained (seed=%d).", seed)
    return expit(raw)
