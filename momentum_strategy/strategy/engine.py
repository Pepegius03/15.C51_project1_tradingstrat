"""
Strategy engine: beta-neutral long/short momentum backtest.

  1. Signal: dual-horizon (12-1 + 6-1 month) cross-sectional rank with
     consistency filter (% of positive months) on longs; raw rank on shorts.
  2. Long book:  top N stocks by combined signal, inv-vol weighted, sums to +1.
  3. Short book: bottom N stocks by raw signal, inv-vol weighted, sums to -0.5.
  4. DJI index hedge: constant -0.5 notional short on the DJI to cancel
     the residual net exposure from the asymmetric books.
  5. Net notional = 0, beta ≈ 0.
  6. Transaction costs applied to both stock legs on rebalance days.
"""

import numpy as np
import pandas as pd

from momentum_strategy.config import VOL_WINDOW, CONSISTENCY_WEIGHT


def momentum_signal(prices_df: pd.DataFrame,
                    formation: int,
                    skip: int) -> pd.DataFrame:
    """Cross-sectional momentum: TRI[t-skip] / TRI[t-skip-formation] - 1."""
    return prices_df.shift(skip) / prices_df.shift(skip + formation) - 1


def backtest_momentum(prices_df: pd.DataFrame,
                      rf_daily: pd.Series,
                      dji_prices: pd.Series = None,
                      formation: int = 252,
                      formation_short: int = 126,
                      skip: int = 21,
                      holding: int = 21,
                      n_long: int = 5,
                      n_short: int = 5,
                      short_notional: float = 0.5,
                      tc: float = 0.001,
                      consistency_weight: float = 0.20):
    """
    Monthly beta-neutral long/short momentum backtest.

    Longs the top n_long DJIA stocks (+1 notional) and shorts the bottom
    n_short stocks (-short_notional). A constant DJI index short of
    -short_notional cancels the residual net exposure, making beta ≈ 0.

    Parameters
    ----------
    prices_df       : TRI masked to DJIA membership (date × ticker)
    rf_daily        : daily T-bill rate aligned to prices_df
    dji_prices      : ^DJI index levels for the index hedge
    formation       : lookback for long-horizon signal (trading days)
    formation_short : lookback for short-horizon signal (trading days)
    skip            : short-term reversal skip (trading days)
    holding         : rebalance frequency (trading days)
    n_long          : number of long positions
    n_short         : number of short positions
    short_notional  : notional weight of the short stock book (and DJI hedge)
    tc              : one-way transaction cost (fraction)
    consistency_weight: blend weight for momentum consistency score

    Returns
    -------
    port_returns : pd.Series   daily strategy returns
    rebal_log    : list[dict]  per-rebalance metadata
    """
    sig_12    = momentum_signal(prices_df, formation, skip)
    sig_6     = momentum_signal(prices_df, formation_short, skip)
    daily_ret = prices_df.pct_change()

    monthly_ret = daily_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    all_dates = prices_df.index
    min_obs   = formation + skip

    dji_ret = dji_prices.pct_change() if dji_prices is not None else None

    rebal_idx   = list(range(min_obs, len(all_dates), holding))
    rebal_dates = [all_dates[i] for i in rebal_idx]

    port_returns: pd.Series = pd.Series(index=all_dates, dtype=float)
    rebal_log:    list      = []

    prev_weights: dict = {}

    for j in range(len(rebal_dates) - 1):
        t0 = rebal_dates[j]
        t1 = rebal_dates[j + 1]

        # ── Combined cross-sectional signal ───────────────────────
        sl = sig_12.loc[t0].dropna()
        ss = sig_6.loc[t0].dropna()
        common = sl.index.intersection(ss.index)

        if len(common) < n_long + n_short + 2:
            continue

        rank_12  = sl[common].rank(pct=True)
        rank_6   = ss[common].rank(pct=True)
        raw_rank = 0.5 * rank_12 + 0.5 * rank_6

        recent_monthly = monthly_ret[monthly_ret.index < t0].tail(12)
        if len(recent_monthly) >= 6:
            consistency = (recent_monthly[common] > 0).mean().fillna(0.5)
            rank_cons   = consistency.rank(pct=True)
            alpha       = consistency_weight
            combined    = (1 - alpha) * raw_rank + alpha * rank_cons
        else:
            combined = raw_rank

        combined     = combined.sort_values(ascending=False)
        long_stocks  = combined.head(n_long).index.tolist()
        # shorts use raw momentum only — consistency filter is designed for uptrends
        # and pushes mean-reverting blue chips out of the short book
        short_stocks = raw_rank.sort_values(ascending=True).head(n_short).index.tolist()

        # ── Volatility-scaled weights ─────────────────────────────
        hist_ret = daily_ret.loc[:t0].tail(VOL_WINDOW)
        vols     = hist_ret.std()

        def inv_vol_weights(stocks: list) -> dict:
            raw = {s: 1.0 / vols[s] if (s in vols and pd.notna(vols[s]) and vols[s] > 0)
                   else 1.0
                   for s in stocks}
            total = sum(raw.values())
            return {s: w / total for s, w in raw.items()}

        long_w  = inv_vol_weights(long_stocks)                                          # sum to +1
        short_w = {s: -short_notional * w for s, w in inv_vol_weights(short_stocks).items()}  # sum to -short_notional
        cur_weights = {**long_w, **short_w}

        # ── Transaction costs (both sides) ────────────────────────
        all_s   = set(cur_weights) | set(prev_weights)
        tc_cost = tc * sum(
            abs(cur_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
            for s in all_s
        )

        rebal_log.append({
            'date'   : t0,
            'longs'  : long_stocks,
            'shorts' : short_stocks,
            'tc_cost': tc_cost,
        })

        # ── Holding-period returns ────────────────────────────────
        hold_dates = all_dates[(all_dates > t0) & (all_dates <= t1)]
        first_day  = True

        for d in hold_dates:
            if d not in daily_ret.index:
                continue

            day_ret = daily_ret.loc[d]
            pnl = sum(
                cur_weights[s] * day_ret.get(s, np.nan)
                for s in cur_weights
                if not np.isnan(day_ret.get(s, np.nan))
            )

            # DJI index short hedge: -short_notional × r_dji cancels net long exposure
            if dji_ret is not None and d in dji_ret.index:
                dji_day = dji_ret.loc[d]
                if pd.notna(dji_day):
                    pnl -= short_notional * dji_day

            if first_day:
                pnl      -= tc_cost
                first_day  = False

            port_returns[d] = pnl

        prev_weights = cur_weights

    return port_returns.dropna(), rebal_log
