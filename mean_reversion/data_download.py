"""
Download adjusted close prices for all DJ30 ever-members (2014-01-01 to 2026-01-01)
and save prices + log returns as pickle files.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from dj30_changes import all_ever_members

START = "2013-01-01"   # extra year for 252-day rolling window burn-in
END   = "2026-01-01"
DATA_DIR = Path(__file__).parent / "data"

EXCLUDED_TICKERS = {"WBA", "UTX"}  # not available on yfinance
TICKERS = [t for t in all_ever_members if t not in EXCLUDED_TICKERS]


def download_prices() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "prices.pkl"
    if cache.exists():
        print("Loading cached prices...")
        return pd.read_pickle(cache)

    print(f"Downloading {len(TICKERS)} tickers from {START} to {END}...")
    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=True,
    )
    # yfinance returns MultiIndex columns (field, ticker); keep Close
    prices = raw["Close"].dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    prices.to_pickle(cache)
    print(f"Saved prices: {prices.shape}")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    cache = DATA_DIR / "returns.pkl"
    if cache.exists():
        print("Loading cached returns...")
        return pd.read_pickle(cache)

    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    returns.to_pickle(cache)
    print(f"Saved returns: {returns.shape}")
    return returns


if __name__ == "__main__":
    prices = download_prices()
    returns = compute_returns(prices)
    print(prices.tail())
    print(returns.tail())
