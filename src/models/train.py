"""
src/models/train.py
--------------------
Per-model training functions and the multi-seed robustness evaluation loop.
"""
from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostClassifier, Pool
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .bilstm import BiLSTMModel
from .ensemble import lgbm_catboost_blend, lstm_catboost_meta
from .neuro_fuzzy import NeuroFuzzyLayer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Bi-LSTM
# ──────────────────────────────────────────────────────────────────────────────

def train_bilstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[BiLSTMModel, np.ndarray]:
    """Train Bi-LSTM and return the model plus its training-set predictions.

    Training-set predictions are needed by the LSTM-CatBoost meta-learner.

    Returns
    -------
    model : BiLSTMModel
    train_preds : np.ndarray  shape (n_train,)
    """
    torch.manual_seed(seed)

    X_tr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    model = BiLSTMModel(X_train.shape[1], hidden_size=cfg["hidden_size"]).to(device)
    opt   = optim.Adam(model.parameters(), lr=cfg["lr"])
    crit  = nn.BCELoss()

    model.train()
    for epoch in range(cfg["num_epochs"]):
        total_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        logger.debug("Bi-LSTM epoch %d/%d  loss=%.4f", epoch + 1, cfg["num_epochs"], total_loss)

    model.eval()
    with torch.no_grad():
        train_preds = model(X_tr).cpu().numpy().flatten()
    return model, train_preds


def predict_bilstm(
    model: BiLSTMModel,
    X_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Generate test-set probability predictions from a trained Bi-LSTM."""
    X_te = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        return model(X_te).cpu().numpy().flatten()


# ──────────────────────────────────────────────────────────────────────────────
# Neuro-Fuzzy
# ──────────────────────────────────────────────────────────────────────────────

def train_neuro_fuzzy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Train the Neuro-Fuzzy model and return test-set probability predictions."""
    torch.manual_seed(seed)

    X_tr   = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr   = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_te   = torch.tensor(X_test,  dtype=torch.float32).to(device)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=cfg.get("batch_size", 2048),
        shuffle=True,
    )

    model = NeuroFuzzyLayer(
        X_train.shape[1], n_mf=cfg.get("num_membership_funcs", 3)
    ).to(device)
    opt  = optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.BCELoss()

    model.train()
    for _ in range(cfg["num_epochs"]):
        for bx, by in loader:
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        return model(X_te).cpu().numpy().flatten()


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM
# ──────────────────────────────────────────────────────────────────────────────

def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_test: pd.DataFrame,
    init_test: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
    """Train LightGBM with log-odds initialisation and return test probabilities."""
    params = {
        "objective":     cfg["objective"],
        "learning_rate": cfg["learning_rate"],
        "verbose":       -1,
        "seed":          seed,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, init_score=init_train)
    model  = lgb.train(params, dtrain, num_boost_round=cfg["num_boost_round"])
    raw    = model.predict(X_test, raw_score=True)
    return expit(raw + init_test)


# ──────────────────────────────────────────────────────────────────────────────
# CatBoost
# ──────────────────────────────────────────────────────────────────────────────

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    init_test: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
    """Train CatBoost with log-odds baseline and return test probabilities."""
    model = CatBoostClassifier(
        iterations=cfg["iterations"],
        learning_rate=cfg["learning_rate"],
        depth=cfg["depth"],
        logging_level="Silent",
        random_seed=seed,
    )
    model.fit(Pool(X_train, y_train, baseline=init_train))
    raw = model.predict(
        Pool(X_test, y_test, baseline=init_test),
        prediction_type="RawFormulaVal",
    )
    return expit(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Robustness seed loop
# ──────────────────────────────────────────────────────────────────────────────

def run_robustness_seeds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    init_test: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, np.ndarray]]:
    """Run all six models across multiple random seeds.

    Parameters
    ----------
    cfg : dict
        Full config dict.

    Returns
    -------
    results : dict
        ``{model_name: {"pr": [...], "roc": [...]}}`` — one entry per seed.
    last_seed_preds : dict
        ``{model_name: np.ndarray}`` — test predictions from the final seed,
        used for regime evaluation and pyCSEP spatial testing.
    """
    model_names = [
        "Bi-LSTM", "Neuro-Fuzzy", "LightGBM",
        "CatBoost", "LGBM-CatBoost", "LSTM-CatBoost",
    ]
    results: dict[str, dict[str, list[float]]] = {n: {"pr": [], "roc": []} for n in model_names}
    last_seed_preds: dict[str, np.ndarray] = {}

    X_tr_arr = X_train.values
    X_te_arr = X_test.values
    mcfg     = cfg["model"]

    for seed in mcfg["seeds"]:
        logger.info("── Seed %d ──────────────────────────────────────────────", seed)
        np.random.seed(seed)

        # 1. Bi-LSTM
        bilstm_model, lstm_train_preds = train_bilstm(
            X_tr_arr, y_train.values, mcfg["bilstm"], device, seed
        )
        lstm_preds = predict_bilstm(bilstm_model, X_te_arr, device)

        # 2. Neuro-Fuzzy
        fuzzy_preds = train_neuro_fuzzy(
            X_tr_arr, y_train.values, X_te_arr, mcfg["neuro_fuzzy"], device, seed
        )

        # 3. LightGBM
        lgbm_preds = train_lgbm(
            X_train, y_train, init_train,
            X_test, init_test,
            mcfg["lgbm"], seed,
        )

        # 4. CatBoost
        cat_preds = train_catboost(
            X_train, y_train, init_train,
            X_test, y_test, init_test,
            mcfg["catboost"], seed,
        )

        # 5. LGBM-CatBoost blend
        lgbm_cat_preds = lgbm_catboost_blend(
            lgbm_preds, cat_preds,
            lgbm_weight=mcfg["ensemble"]["lgbm_weight"],
        )

        # 6. LSTM-CatBoost meta-learner
        lstm_cat_preds = lstm_catboost_meta(
            X_train, y_train, init_train,
            X_test, y_test, init_test,
            lstm_train_preds, lstm_preds,
            mcfg["catboost"], seed,
        )

        seed_preds = {
            "Bi-LSTM":       lstm_preds,
            "Neuro-Fuzzy":   fuzzy_preds,
            "LightGBM":      lgbm_preds,
            "CatBoost":      cat_preds,
            "LGBM-CatBoost": lgbm_cat_preds,
            "LSTM-CatBoost": lstm_cat_preds,
        }
        for name, preds in seed_preds.items():
            results[name]["pr"].append(average_precision_score(y_test, preds))
            results[name]["roc"].append(roc_auc_score(y_test, preds))

        last_seed_preds = seed_preds
        logger.info(
            "Seed %d done — LSTM-CatBoost PR-AUC=%.4f",
            seed, results["LSTM-CatBoost"]["pr"][-1],
        )

    return results, last_seed_preds
