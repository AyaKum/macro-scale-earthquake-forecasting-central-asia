"""
src/visualization/plots.py
---------------------------
Matplotlib figures for the spatial test and regime comparison.
"""
from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_spatial_test(
    results_df: pd.DataFrame,
    output_path: str | None = "outputs/csep_binary_spatial_test.png",
    dpi: int = 150,
) -> plt.Figure:
    """Horizontal bar chart of pyCSEP δ-scores per model.

    Bars are colour-coded: steel-blue for consistent (δ > 0.05),
    salmon for inconsistent.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`src.evaluation.spatial_test.run_binary_spatial_test`.
    output_path : str | None
        Save path. ``None`` → display only.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    names  = results_df["Model"].tolist()
    scores = results_df["delta"].tolist()
    colors = ["steelblue" if s > 0.05 else "salmon" for s in scores]

    bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0.05, color="crimson", linewidth=1.8, linestyle="--", label="α = 0.05")
    ax.axvline(0.50, color="gray",    linewidth=1.0, linestyle=":",  label="δ = 0.50")

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=9,
        )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("δ-score  (quantile)", fontsize=11)
    ax.set_title(
        "pyCSEP Binary Spatial Test — Central Asia M≥3.0  (2019–2024)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved: %s", output_path)
    return fig


def plot_regime_comparison(
    regime_df: pd.DataFrame,
    models: list[str] | None = None,
    output_path: str | None = "outputs/regime_comparison.png",
    dpi: int = 150,
) -> plt.Figure:
    """Grouped bar chart of PR-AUC disaggregated by tectonic regime and model.

    Parameters
    ----------
    regime_df : pd.DataFrame
        Output of :func:`src.evaluation.regime_eval.evaluate_by_regime`.
    models : list[str] | None
        Subset of models to plot. ``None`` → all.
    output_path : str | None
        Save path. ``None`` → display only.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if models is None:
        models = regime_df["Model"].unique().tolist()

    regimes = sorted(regime_df["Regime"].unique())
    x       = np.arange(len(regimes))
    width   = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        subset = regime_df[regime_df["Model"] == model].set_index("Regime")
        vals   = [
            float(subset.loc[r, "PR-AUC"]) if r in subset.index else 0.0
            for r in regimes
        ]
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([r.replace(" ", "\n") for r in regimes], fontsize=9)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Spatial Generalisability by Tectonic Regime")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved: %s", output_path)
    return fig
