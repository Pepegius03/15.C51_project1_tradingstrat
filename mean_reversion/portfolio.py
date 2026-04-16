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

Z_THRESHOLD    = 1.0   # stock must be depressed: z < -Z_THRESHOLD
PROB_THRESHOLD = 0.55  # model must predict upward reversion with this confidence


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

        # Sparsity filter: only depressed stocks where model predicts upward reversion
        mask = (day_z < -Z_THRESHOLD) & (prob_s > PROB_THRESHOLD) 

        active_tickers = mask[mask].index
        if active_tickers.empty:
            weights_list.append(pd.Series(0.0, index=tickers, name=date))
            continue

        z_active    = day_z[active_tickers]
        prob_active = prob_s[active_tickers]

        long_score = prob_active * (-z_active)  # both factors positive by construction

        w = pd.Series(0.0, index=tickers)
        w[long_score.index] = long_score / long_score.sum()

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
    print("Daily gross exposure (should be ~1):", weights.sum(axis=1).describe())
