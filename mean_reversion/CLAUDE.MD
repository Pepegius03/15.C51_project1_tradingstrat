# Goal
Mean-reversion trading strategy across DJ30 stocks, daily data 2014–2026.

# Splits
- Train: 2014–2019 | Val: 2020–2022 | Test: 2023–2025

# Pipeline
| File | Role |
|---|---|
| `data_download.py` | yfinance download for all DJ30 ever-members → `data/prices.parquet`, `data/returns.parquet` |
| `data_prep.py` | 252-day rolling beta, CAPM residuals, membership mask, cross-sectional z-scores, 8 features, next-day targets |
| `model.py` | Train RF + LogReg; select best by val AUC; cache to `models/best_model.pkl` |
| `portfolio.py` | Rank-weighted sparse long/short; filters: `|z|>1.0` and confidence>0.6 |
| `backtest.py` | Sharpe, max drawdown, turnover, equity curve plots |
| `main.py` | Orchestrates full pipeline; `--refresh` clears cache |

# Features (8)
`z_t`, `z_t1`, `z_t5`, `z_roll5_mean`, `z_roll5_std`, `cum_resid_5d`, `beta`, `realized_vol_20d`

# Target
Binary: `1` if next-day excess return > 0, else `0`

# Run
```bash
python main.py           # uses cache
python main.py --refresh # rerun from scratch
``` 