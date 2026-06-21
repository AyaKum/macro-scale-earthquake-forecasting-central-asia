"""
scripts/run_pipeline.py
------------------------
Full pipeline entry point.  Reads config, runs data acquisition → feature
engineering → training → evaluation → figures.

Usage
-----
    # From the repo root:
    python scripts/run_pipeline.py --config config/config.yaml

    # Or install in editable mode first (recommended):
    pip install -e .
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure repo root is on sys.path when running this script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from src.data.acquisition import fetch_catalog
from src.data.preprocessing import (
    build_weekly_dataframe,
    compute_target,
    compute_temporal_features,
    merge_tectonic_context,
)
from src.evaluation.metrics import precision_at_k, summarize_seed_results
from src.evaluation.regime_eval import evaluate_by_regime
from src.evaluation.spatial_test import run_binary_spatial_test
from src.features.normalization import (
    FEATURE_COLS,
    compute_log_odds_baseline,
    regime_zscore_normalize,
)
from src.features.spatial_grid import assign_regimes, build_grid
from src.features.tectonic import compute_fault_features, load_gem_faults
from src.models.train import run_robustness_seeds
from src.visualization.plots import plot_regime_comparison, plot_spatial_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-35s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def main(cfg: dict) -> None:
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 1. Data acquisition ────────────────────────────────────────────────────
    df_raw = fetch_catalog(cfg["data"])

    # ── 2. Spatial grid + tectonic context ────────────────────────────────────
    df_raw, gdf_cells = build_grid(df_raw, cfg["grid"])
    gdf_cells = assign_regimes(gdf_cells, cfg["paths"]["emca_shapefile"])
    faults    = load_gem_faults(cfg)
    gdf_cells = compute_fault_features(gdf_cells, faults)

    # ── 3. Weekly panel + target + temporal features ───────────────────────────
    df_full = build_weekly_dataframe(df_raw, gdf_cells)
    df_full = compute_target(df_full)
    df_full = compute_temporal_features(df_full)
    df_full = merge_tectonic_context(df_full, gdf_cells)

    # ── 4. Normalisation ───────────────────────────────────────────────────────
    train_mask = df_full["year_week"] < cfg["evaluation"]["test_start"]
    test_mask  = df_full["year_week"] >= cfg["evaluation"]["test_start"]

    df_full = compute_log_odds_baseline(df_full, train_mask)
    df_full = regime_zscore_normalize(df_full, train_mask)

    X_train    = df_full.loc[train_mask, FEATURE_COLS]
    y_train    = df_full.loc[train_mask, "target_4w_bin"]
    init_train = df_full.loc[train_mask, "log_odds_base"].values

    X_test    = df_full.loc[test_mask, FEATURE_COLS]
    y_test    = df_full.loc[test_mask, "target_4w_bin"]
    init_test = df_full.loc[test_mask, "log_odds_base"].values

    # ── 5. Training + robustness evaluation ────────────────────────────────────
    results, last_preds = run_robustness_seeds(
        X_train, y_train, init_train,
        X_test, y_test, init_test,
        cfg, device,
    )
    summary = summarize_seed_results(results)
    out = cfg["paths"]["output_dir"]
    summary.to_csv(os.path.join(out, "metrics_summary.csv"))

    # Precision@1% for the champion model on the final seed
    pa1 = precision_at_k(
        y_test.values, last_preds["LSTM-CatBoost"],
        k=cfg["evaluation"]["precision_at_k"],
    )
    logger.info("LSTM-CatBoost  Precision@1%% = %.4f", pa1)

    # ── 6. Regime evaluation ───────────────────────────────────────────────────
    regimes_test = df_full.loc[test_mask, "tect_reg"]
    regime_df    = evaluate_by_regime(y_test, last_preds, regimes_test)
    regime_df.to_csv(os.path.join(out, "regime_results.csv"), index=False)
    plot_regime_comparison(
        regime_df,
        models=["Bi-LSTM", "CatBoost", "LSTM-CatBoost"],
        output_path=os.path.join(out, "regime_comparison.png"),
    )

    # ── 7. pyCSEP binary spatial test ──────────────────────────────────────────
    spatial_df = run_binary_spatial_test(
        df_full, test_mask, last_preds,
        num_simulations=cfg["evaluation"]["csep_simulations"],
        seed=cfg["evaluation"]["csep_seed"],
    )
    spatial_df.to_csv(os.path.join(out, "spatial_test.csv"), index=False)
    plot_spatial_test(
        spatial_df,
        output_path=os.path.join(out, "csep_binary_spatial_test.png"),
    )

    logger.info("Pipeline complete. All outputs in: %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Central Asia earthquake forecasting pipeline.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
