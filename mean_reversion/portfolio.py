"""
Construct a rank-weighted, dollar-neutral long/short portfolio.

Signal logic:
  long_score  = P(revert up) × max(-z, 0)   [depressed stock + high confidence]
  short_score = P(revert dn) × max( z, 0)   [elevated stock + high confidence]

Sparsity filters (only trade the most significant positions):
  |z| > Z_THRESHOLD  AND  confidence > CONF_THRESHOLD
"""

import numpy as np
import pandas as pd
from pathlib import Path
from model import predict_proba, FEATURE_COLS

DATA_DIR = Path(__file__).parent / "data"

Z_THRESHOLD   = 1.0   # minimum |z-score| to trade
CONF_THRESHOLD = 0.6  # minimum confidence (prob away from 0.5)


def build_weights(features: pd.DataFrame, zscores: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    cache = DATA_DIR / "weights.pkl"
    if cache.exists():
        print("Loading cached weights...")
        return pd.read_pickle(cache)

    print("Building portfolio weights...")
    dates   = features.index.get_level_values("date").unique().sort_values()
    tickers = zscores.columns.tolist()

    weights_list = []

    for date in dates:
        if date not in features.index.get_level_values("date"):
            continue

        day_feat = features.xs(date, level="date")
        day_z    = zscores.loc[date, day_feat.index] if date in zscores.index else pd.Series(dtype=float)

        if day_feat.empty or day_z.dropna().empty:
            weights_list.append(pd.Series(0.0, index=tickers, name=date))
            continue

        # Reversion probability
        proba = predict_proba(bundle, day_feat[FEATURE_COLS])
        prob_s = pd.Series(proba, index=day_feat.index)

        # Sparsity filter
        abs_z = day_z.abs()
        confidence = (prob_s - 0.5).abs()
        mask = (abs_z > Z_THRESHOLD) & (confidence > CONF_THRESHOLD - 0.5)

        active_tickers = mask[mask].index
        if active_tickers.empty:
            weights_list.append(pd.Series(0.0, index=tickers, name=date))
            continue

        z_active    = day_z[active_tickers]
        prob_active = prob_s[active_tickers]

        long_score  = prob_active * (-z_active).clip(lower=0)
        short_score = (1 - prob_active) * z_active.clip(lower=0)

        w = pd.Series(0.0, index=tickers)

        if long_score.sum() > 0:
            longs = long_score[long_score > 0]
            w[longs.index] = longs / longs.sum()

        if short_score.sum() > 0:
            shorts = short_score[short_score > 0]
            w[shorts.index] -= shorts / shorts.sum()

        weights_list.append(w.rename(date))

    weights = pd.DataFrame(weights_list)
    weights.index.name = "date"
    weights.to_pickle(cache)
    print(f"Weights built: {weights.shape}")
    return weights


if __name__ == "__main__":
    import pickle
    features = pd.read_pickle(DATA_DIR / "features.pkl")
    zscores  = pd.read_pickle(DATA_DIR / "zscores.pkl")
    with open(Path(__file__).parent / "models" / "best_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    weights = build_weights(features, zscores, bundle)
    print(weights.tail())
    print("Net exposure check (should be ~0):", weights.sum(axis=1).describe())
