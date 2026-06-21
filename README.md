# Macro-Scale Earthquake Forecasting Under Class Imbalance in Central Asia

> **Paper** for *Journal of Big Data*, Springer (2025, under review)

## Overview

Short-term probabilistic earthquake forecasting (M≥3.0, 4-week horizon) across Central Asia
on a 1°×1° spatial grid spanning 1973–2024. The pipeline addresses severe class imbalance
(~4.5% positive prevalence) through regime-aware Z-score normalisation and per-cell log-odds
initialisation, and evaluates six models: Bi-LSTM, Neuro-Fuzzy, LightGBM, CatBoost,
LGBM-CatBoost ensemble, and LSTM-CatBoost meta-learner. Spatial consistency is assessed
using the pyCSEP binary spatial test (Bayona et al., 2022).

## Data

| Source | Access | Destination |
|---|---|---|
| USGS FDSN (~110k events, M≥3.0, 1973–2024) | Auto-downloaded | — |
| GEM Global Active Faults | Auto-downloaded | `data/gem_active_faults.geojson` |
| EMCA Seismic Zones v1.0 | **Manual** — request from [EMCA portal](https://www.seismicportal.eu/) | `data/EMCA_seismozonesv1.0_shp/` |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/earthquake-forecasting-central-asia.git
cd earthquake-forecasting-central-asia
pip install -r requirements.txt
```

## Usage

### Full pipeline

```bash
python scripts/run_pipeline.py --config config/config.yaml
```

Outputs are written to `outputs/`.

### Changing settings

All hyperparameters, date ranges, file paths, and random seeds live in
`config/config.yaml` — no need to touch the source code.

## Repository Structure

```
earthquake-forecasting-central-asia/
├── config/
│   └── config.yaml              # all hyperparameters and paths
├── src/
│   ├── data/
│   │   ├── acquisition.py       # USGS FDSN download
│   │   └── preprocessing.py     # weekly panel, target, temporal features
│   ├── features/
│   │   ├── spatial_grid.py      # 1° grid construction + EMCA regime assignment
│   │   ├── tectonic.py          # GEM Active Faults download + fault features
│   │   └── normalization.py     # log-odds baseline + regime Z-score
│   ├── models/
│   │   ├── bilstm.py            # BiLSTMModel
│   │   ├── neuro_fuzzy.py       # NeuroFuzzyLayer (ANFIS-style)
│   │   ├── ensemble.py          # LGBM-CatBoost blend + LSTM-CatBoost meta-learner
│   │   └── train.py             # per-model training functions + seed loop
│   ├── evaluation/
│   │   ├── metrics.py           # PR-AUC, ROC-AUC, Precision@k, summary table
│   │   ├── regime_eval.py       # per-regime disaggregation
│   │   └── spatial_test.py      # pyCSEP binary spatial test (Bayona et al. 2022)
│   └── visualization/
│       └── plots.py             # spatial test bar chart, regime comparison
├── scripts/
│   └── run_pipeline.py          # entry point
├── requirements.txt
├── CITATION.cff
└── .gitignore
```

## Citation

Will add after publication.
```
