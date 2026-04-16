"""
Full pipeline orchestrator.

Usage:
    python main.py                  # run everything, cache intermediate results
    python main.py --refresh        # delete cache and rerun from scratch
"""

import argparse
import shutil
from pathlib import Path

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


def main(refresh: bool = False):
    if refresh:
        print("Clearing cache...")
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        shutil.rmtree(MODEL_DIR, ignore_errors=True)

    # ── 1. Download ──────────────────────────────────────────────
    from data_download import download_prices, compute_returns
    prices  = download_prices()
    returns = compute_returns(prices)

    # ── 2. Data prep ────────────────────────────────────────────
    from data_prep import load_ff_factors, build_features
    ff = load_ff_factors()
    features, targets, zscores = build_features(returns, ff)

    # ── 3. Model ────────────────────────────────────────────────
    from model import train
    bundle = train(features, targets)

    # ── 4. Portfolio ─────────────────────────────────────────────
    from portfolio import build_weights
    weights = build_weights(features, zscores, bundle)

    # ── 5. Backtest ──────────────────────────────────────────────
    from backtest import run_backtest
    pnl = run_backtest(weights, returns)

    return pnl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Clear cache and rerun from scratch")
    args = parser.parse_args()
    main(refresh=args.refresh)
