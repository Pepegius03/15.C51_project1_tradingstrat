"""
Multi-Factor Market-Neutral Long/Short backtest engine.

Strategy overview
-----------------
Four cross-sectional factors are computed monthly within the active DJIA
universe, z-score normalised, winsorized, and blended into a composite score:

    1. Momentum    (12-1 month TRI return, blended 50/50 with 6-1 month)
    2. Value       (book-to-market: ceqq / market-cap, higher = cheaper)
    3. Quality     (ROE: trailing-12-month net income / avg book equity)
    4. Low-Vol     (63-day realized vol, inverted — lower vol → higher score)

Portfolio construction
----------------------
- Long  top    N stocks by composite score  (inv-vol weighted, sums to +1)
- Short bottom N stocks by composite score  (inv-vol weighted, sums to -1)
- Net notional = 0 → strictly dollar-neutral, no index hedge required
- Soft sector tilt: no sector may contribute more than MAX_SECTOR_TILT stocks
  to either the long or short book (marginal offenders replaced with the
  next-best stock from an under-represented sector)
- Rebalance on calendar month-end trading dates
- Transaction costs: 10 bps one-way on net weight changes
"""

import numpy as np
import pandas as pd

from momentum_strategy.strategy.engine import momentum_signal
from momentum_strategy.config import (
    FORMATION_DAYS, FORMATION_SHORT, SKIP_DAYS,
    LOW_VOL_DAYS, WINSOR_CLIP, FACTOR_WEIGHTS,
    N_LONG_MF, N_SHORT_MF, TC,
    SECTOR_MAP, MAX_SECTOR_TILT,
)


# ── Factor helpers ────────────────────────────────────────────────────────────

def _normalize(s: pd.Series, winsor_clip: float = WINSOR_CLIP) -> pd.Series:
    """Winsorize then z-score a cross-sectional factor series."""
    s = s.dropna()
    if len(s) < 3:
        return pd.Series(dtype=float)
    lo = s.quantile(winsor_clip)
    hi = s.quantile(1.0 - winsor_clip)
    s  = s.clip(lower=lo, upper=hi)
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-10:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def _factor_momentum(tri_df: pd.DataFrame, date) -> pd.Series:
    """Blend of 12-1 and 6-1 month momentum signals at a given date."""
    sig_12 = momentum_signal(tri_df, FORMATION_DAYS,  SKIP_DAYS).loc[date]
    sig_6  = momentum_signal(tri_df, FORMATION_SHORT, SKIP_DAYS).loc[date]
    common = sig_12.dropna().index.intersection(sig_6.dropna().index)
    return (0.5 * sig_12[common] + 0.5 * sig_6[common])


def _factor_value(
    ceqq_panel:  pd.DataFrame,
    cshoq_panel: pd.DataFrame,
    prices_df:   pd.DataFrame,
    date,
) -> pd.Series:
    """
    Book-to-market ratio.

    B/M = ceqq / (cshoq * 1000 * |price|)

    Both ceqq and cshoq come from Compustat (already point-in-time aligned).
    Higher B/M → cheaper → positive signal.
    """
    book_eq  = ceqq_panel.loc[date]
    shares   = cshoq_panel.loc[date]         # thousands of shares
    last_prc = prices_df.loc[:date].iloc[-1] # most recent available price

    mkt_cap = shares * 1_000.0 * last_prc.abs()
    bm = book_eq / mkt_cap

    # Exclude negative book equity (distressed / financial anomalies)
    bm = bm[bm > 0]
    return bm


def _factor_quality(
    ceqq_panel: pd.DataFrame,
    niq_panel:  pd.DataFrame,
    date,
    ttm_quarters: int = 4,
) -> pd.Series:
    """
    Return-on-Equity: trailing-12-month net income / average book equity.

    Uses the point-in-time quarterly panels — the last ttm_quarters rows
    before (and including) date are available data only.
    """
    # TTM net income: sum of last 4 available quarterly observations
    ni_slice   = niq_panel.loc[:date].tail(ttm_quarters)
    ni_ttm     = ni_slice.sum(min_count=ttm_quarters)   # NaN if fewer than 4 obs

    # Average book equity: mean of last 2 available quarterly observations
    ceqq_slice = ceqq_panel.loc[:date].tail(2)
    avg_ceqq   = ceqq_slice.mean()

    roe = ni_ttm / avg_ceqq.abs()
    # Exclude where avg book equity is near zero or where TTM NI is unavailable
    roe = roe.replace([np.inf, -np.inf], np.nan).dropna()
    return roe


def _factor_low_vol(
    daily_returns: pd.DataFrame,
    date,
    window: int = LOW_VOL_DAYS,
) -> pd.Series:
    """
    Negative trailing realized volatility.
    Lower vol stocks receive a higher (less negative) score.
    """
    hist_ret = daily_returns.loc[:date].tail(window)
    vol      = hist_ret.std()
    return -vol   # invert so that low vol → high score


# ── Sector tilt enforcement ───────────────────────────────────────────────────

def _enforce_sector_tilt(
    composite:        pd.Series,
    initial_stocks:   list,
    all_available:    pd.Series,
    sector_map:       dict,
    max_tilt:         int,
    book_size:        int,
    prefer_high:      bool,
) -> list:
    """
    Replace overweight-sector stocks with next-best alternatives.

    Parameters
    ----------
    composite      : full composite-score series (sorted best→worst or worst→best)
    initial_stocks : initial selection (top or bottom N)
    all_available  : composite series restricted to eligible stocks
    sector_map     : ticker → GICS sector string
    max_tilt       : max allowed stocks from any one sector in this book
    book_size      : target number of stocks (N_LONG or N_SHORT)
    prefer_high    : True for the long book (want high composite), False for short
    """
    selected = list(initial_stocks)
    used     = set(selected)

    def sector_counts(stocks):
        counts = {}
        for s in stocks:
            sec = sector_map.get(s, 'Unknown')
            counts[sec] = counts.get(sec, 0) + 1
        return counts

    # Build a ranked list of candidates outside the initial selection
    if prefer_high:
        candidates = [t for t in composite.sort_values(ascending=False).index
                      if t not in used]
    else:
        candidates = [t for t in composite.sort_values(ascending=True).index
                      if t not in used]

    changed = True
    while changed:
        changed = False
        counts  = sector_counts(selected)
        for sec, cnt in counts.items():
            if cnt > max_tilt:
                # Find the marginal stock from this sector to remove
                # (the worst-ranked one in the book)
                sec_stocks = [s for s in selected if sector_map.get(s, 'Unknown') == sec]
                if prefer_high:
                    # Remove the stock with the lowest composite score in the book
                    to_remove = min(sec_stocks, key=lambda s: composite.get(s, 0))
                else:
                    # Remove the stock with the highest composite score in the book
                    to_remove = max(sec_stocks, key=lambda s: composite.get(s, 0))

                # Find the best replacement from a different / non-over-tilted sector
                for cand in candidates:
                    cand_sec    = sector_map.get(cand, 'Unknown')
                    cand_counts = sector_counts(selected)
                    if cand_counts.get(cand_sec, 0) < max_tilt:
                        selected.remove(to_remove)
                        selected.append(cand)
                        candidates.remove(cand)
                        candidates.insert(0, to_remove)  # to_remove goes back to pool
                        used = set(selected)
                        changed = True
                        break
                if changed:
                    break

    return selected[:book_size]


# ── Inverse-vol position weights ──────────────────────────────────────────────

def _inv_vol_weights(stocks: list, vols: pd.Series) -> dict:
    """Inverse-volatility weights that sum to 1.0."""
    raw   = {s: 1.0 / max(float(vols.get(s, 1.0)), 1e-8) for s in stocks}
    total = sum(raw.values())
    return {s: w / total for s, w in raw.items()}


# ── Main backtest ─────────────────────────────────────────────────────────────

def backtest_multifactor(
    tri_df:        pd.DataFrame,
    returns_df:    pd.DataFrame,
    rf_daily:      pd.Series,
    ceqq_panel:    pd.DataFrame,
    niq_panel:     pd.DataFrame,
    cshoq_panel:   pd.DataFrame,
    prices_df:     pd.DataFrame,
    factor_weights: dict  = None,
    n_long:         int   = N_LONG_MF,
    n_short:        int   = N_SHORT_MF,
    tc:             float = TC,
    sector_map:     dict  = None,
    max_sector_tilt: int  = MAX_SECTOR_TILT,
) -> tuple:
    """
    Multi-factor market-neutral long/short backtest.

    Parameters
    ----------
    tri_df         : Total Return Index, date × ticker (masked to DJIA membership)
    returns_df     : daily returns, date × ticker (masked)
    rf_daily       : daily T-bill rate aligned to returns_df
    ceqq_panel     : point-in-time book equity (date × ticker)
    niq_panel      : point-in-time quarterly net income (date × ticker)
    cshoq_panel    : point-in-time shares outstanding in 000s (date × ticker)
    prices_df      : closing prices, date × ticker (for market-cap denominator)
    factor_weights : dict mapping factor names to blend weights
    n_long/n_short : number of longs / shorts per rebalance
    tc             : one-way transaction cost (fraction)
    sector_map     : ticker → GICS sector (used for soft sector-tilt constraint)
    max_sector_tilt: max stocks from one sector allowed in either book

    Returns
    -------
    port_returns : pd.Series  daily net strategy returns
    rebal_log    : list[dict] per-rebalance metadata
    """
    if factor_weights is None:
        factor_weights = FACTOR_WEIGHTS
    if sector_map is None:
        sector_map = SECTOR_MAP

    # Calendar month-end rebalance dates (cleaner than fixed 21-day drift)
    min_obs     = FORMATION_DAYS + SKIP_DAYS
    all_dates   = returns_df.index
    rebal_dates = (
        returns_df.resample('ME').last().index
        .intersection(all_dates)
    )
    rebal_dates = rebal_dates[rebal_dates >= all_dates[min_obs]]

    port_returns: pd.Series = pd.Series(index=all_dates, dtype=float)
    rebal_log:    list      = []
    prev_weights: dict      = {}

    for j in range(len(rebal_dates) - 1):
        t0 = rebal_dates[j]
        t1 = rebal_dates[j + 1]

        # ── Compute each factor ───────────────────────────────────
        f_mom   = _factor_momentum(tri_df, t0)
        f_val   = _factor_value(ceqq_panel, cshoq_panel, prices_df, t0)
        f_qual  = _factor_quality(ceqq_panel, niq_panel, t0)
        f_lvol  = _factor_low_vol(returns_df, t0)

        # ── Common universe: stocks with all 4 factor scores ─────
        common = (f_mom.dropna().index
                  .intersection(f_val.dropna().index)
                  .intersection(f_qual.dropna().index)
                  .intersection(f_lvol.dropna().index))

        if len(common) < n_long + n_short + 2:
            continue

        # ── Normalize and combine ─────────────────────────────────
        z_mom  = _normalize(f_mom[common])
        z_val  = _normalize(f_val[common])
        z_qual = _normalize(f_qual[common])
        z_lvol = _normalize(f_lvol[common])

        composite = (
            factor_weights.get('momentum', 0.30) * z_mom  +
            factor_weights.get('value',    0.25) * z_val  +
            factor_weights.get('quality',  0.25) * z_qual +
            factor_weights.get('low_vol',  0.20) * z_lvol
        ).dropna().sort_values(ascending=False)

        if len(composite) < n_long + n_short + 2:
            continue

        # ── Initial long/short selection ──────────────────────────
        init_longs  = composite.head(n_long).index.tolist()
        init_shorts = composite.tail(n_short).index.tolist()

        # ── Soft sector-tilt constraint ───────────────────────────
        long_stocks  = _enforce_sector_tilt(
            composite, init_longs, composite, sector_map,
            max_tilt=max_sector_tilt, book_size=n_long, prefer_high=True,
        )
        short_stocks = _enforce_sector_tilt(
            composite, init_shorts, composite, sector_map,
            max_tilt=max_sector_tilt, book_size=n_short, prefer_high=False,
        )

        # ── Volatility-scaled weights (inv-vol, symmetric ±1) ────
        hist_vols = returns_df.loc[:t0].tail(LOW_VOL_DAYS).std()

        long_w  = _inv_vol_weights(long_stocks,  hist_vols)        # sums to +1
        short_w = {s: -w for s, w in
                   _inv_vol_weights(short_stocks, hist_vols).items()}  # sums to -1
        cur_weights = {**long_w, **short_w}

        # ── Transaction costs (on net weight change) ──────────────
        all_s   = set(cur_weights) | set(prev_weights)
        tc_cost = tc * sum(
            abs(cur_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
            for s in all_s
        )

        rebal_log.append({
            'date'            : t0,
            'longs'           : long_stocks,
            'shorts'          : short_stocks,
            'tc_cost'         : tc_cost,
            'n_common'        : len(common),
            'factor_scores'   : {
                t: {
                    'momentum': float(z_mom.get(t, np.nan)),
                    'value'   : float(z_val.get(t, np.nan)),
                    'quality' : float(z_qual.get(t, np.nan)),
                    'low_vol' : float(z_lvol.get(t, np.nan)),
                    'composite': float(composite.get(t, np.nan)),
                }
                for t in long_stocks + short_stocks
            },
        })

        # ── Daily P&L over holding period ────────────────────────
        hold_dates = all_dates[(all_dates > t0) & (all_dates <= t1)]
        first_day  = True

        for d in hold_dates:
            if d not in returns_df.index:
                continue
            day_ret = returns_df.loc[d]
            pnl = sum(
                cur_weights[s] * day_ret.get(s, np.nan)
                for s in cur_weights
                if pd.notna(day_ret.get(s, np.nan))
            )
            if first_day:
                pnl      -= tc_cost
                first_day  = False
            port_returns[d] = pnl

        prev_weights = cur_weights

    return port_returns.dropna(), rebal_log
