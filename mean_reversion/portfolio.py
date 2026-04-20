"""
Long-only, equal-weight portfolio — no ML model.

Rule: every REBAL_FREQ trading days, select the top-N stocks whose
cross-sectional z-score < -Z_THRESHOLD (most depressed), assign equal
weight 1/N.  Weights are held constant between rebalance dates so
daily returns compound naturally via weights[t] · returns[t+1].
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

Z_THRESHOLD  = 1.0   # entry: z must be below this (depressed stocks)
TOP_N        = 3     # number of stocks to hold
REBAL_FREQ   = 5     # trading days between rebalances (≈ weekly)


def build_weights(zscores: pd.DataFrame) -> pd.DataFrame:
    cache = DATA_DIR / "weights.pkl"
    if cache.exists():
        print("Loading cached weights...")
        return pd.read_pickle(cache)

    print("Building portfolio weights (weekly rebalance, top-3 z-score)...")
    tickers = zscores.columns.tolist()
    dates   = zscores.index

    current_w    = pd.Series(0.0, index=tickers)
    weights_list = []

    for i, date in enumerate(dates):
        if i % REBAL_FREQ == 0:
            day_z    = zscores.loc[date].dropna()
            eligible = day_z[day_z < -Z_THRESHOLD]

            if not eligible.empty:
                # pick the N most depressed stocks (smallest z)
                top = eligible.nsmallest(TOP_N).index
                current_w = pd.Series(0.0, index=tickers)
                current_w[top] = 1.0 / len(top)
            else:
                current_w = pd.Series(0.0, index=tickers)

        weights_list.append(current_w.rename(date))

    weights = pd.DataFrame(weights_list)
    weights.index.name = "date"
    weights.to_pickle(cache)
    print(f"Weights built: {weights.shape}")
    return weights


if __name__ == "__main__":
    zscores = pd.read_pickle(DATA_DIR / "zscores.pkl")
    weights = build_weights(zscores)
    print(weights.tail())
    print("Daily gross exposure:", weights.sum(axis=1).describe())
