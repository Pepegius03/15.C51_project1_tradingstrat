"""
Backtest: compute portfolio returns from weights and next-day log returns.
Reports annualized Sharpe, max drawdown, turnover; plots equity curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR  = Path(__file__).parent / "figures"

TRAIN_END = "2019-12-31"
VAL_END   = "2022-12-31"
TEST_END  = "2025-12-31"
PERIODS = {
    "Train (2014–2019)": (None, TRAIN_END),
    "Val   (2020–2022)": (TRAIN_END, VAL_END),
    "Test  (2023–2025)": (VAL_END, TEST_END),
}


def portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """weights[t] · returns[t+1]"""
    fwd_ret = returns.shift(-1)
    common_dates = weights.index.intersection(fwd_ret.index)
    w = weights.loc[common_dates]
    r = fwd_ret.loc[common_dates].reindex(columns=w.columns, fill_value=0.0)
    pnl = (w * r).sum(axis=1)
    return pnl


def sharpe(pnl: pd.Series, ann: int = 252) -> float:
    if pnl.std() == 0:
        return np.nan
    return pnl.mean() / pnl.std() * np.sqrt(ann)


def max_drawdown(pnl: pd.Series) -> float:
    cum = pnl.cumsum()
    roll_max = cum.cummax()
    dd = cum - roll_max
    return dd.min()


def turnover(weights: pd.DataFrame) -> float:
    """Average daily one-way turnover."""
    diff = weights.diff().abs().sum(axis=1)
    return diff.mean()


def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    pnl = portfolio_returns(weights, returns)

    FIG_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 55)
    print(f"{'Period':<25} {'Sharpe':>8} {'MaxDD':>10} {'TurnoverD':>10}")
    print("=" * 55)

    fig, axes = plt.subplots(len(PERIODS), 1, figsize=(12, 9), sharex=False)
    fig.suptitle("Mean-Reversion Strategy — Equity Curves", fontsize=13)

    for ax, (label, (start, end)) in zip(axes, PERIODS.items()):
        mask = pd.Series(True, index=pnl.index)
        if start:
            mask &= pnl.index > start
        if end:
            mask &= pnl.index <= end

        sub_pnl = pnl[mask]
        sub_w   = weights.loc[weights.index.isin(sub_pnl.index)]

        sr  = sharpe(sub_pnl)
        mdd = max_drawdown(sub_pnl)
        to  = turnover(sub_w)

        print(f"{label:<25} {sr:>8.3f} {mdd:>10.4f} {to:>10.4f}")

        cum = sub_pnl.cumsum()
        ax.plot(cum.index, cum.values, linewidth=1.2)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(f"{label}  |  Sharpe={sr:.2f}  MaxDD={mdd:.4f}", fontsize=10)
        ax.set_ylabel("Cum. log return")

    print("=" * 55)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "equity_curves.png", dpi=150)
    plt.show()
    print(f"\nPlot saved to {FIG_DIR / 'equity_curves.png'}")
    return pnl


if __name__ == "__main__":
    from data_download import compute_returns, download_prices
    prices  = download_prices()
    returns = compute_returns(prices)
    weights = pd.read_pickle(DATA_DIR / "weights.pkl")
    run_backtest(weights, returns)
