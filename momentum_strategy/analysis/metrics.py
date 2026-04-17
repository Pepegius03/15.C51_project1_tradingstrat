"""
Performance statistics: risk/return metrics, console printing, annual table.
"""

import numpy as np
import pandas as pd
from scipy import stats


def performance_stats(returns: pd.Series,
                      rf: pd.Series = None,
                      label: str = "Strategy",
                      freq: int = 252) -> dict:
    """Full suite of risk / return statistics."""
    r    = returns.dropna()
    if len(r) == 0:
        raise ValueError(f"'{label}' return series is empty — check data alignment.")
    rf_s = rf.reindex(r.index).fillna(0) if rf is not None else pd.Series(0, index=r.index)
    excess = r - rf_s

    n       = len(r)
    ann_ret = (1 + r).prod() ** (freq / n) - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe  = excess.mean() / r.std() * np.sqrt(freq) if r.std() > 0 else np.nan

    neg_r        = r[r < 0]
    downside_vol = neg_r.std() * np.sqrt(freq) if len(neg_r) > 1 else np.nan
    sortino      = ann_ret / downside_vol if downside_vol and downside_vol > 0 else np.nan

    cum      = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / roll_max
    max_dd   = dd.min()
    calmar   = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    monthly  = r.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly > 0).mean()

    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=True))

    var_95  = float(np.percentile(r, 5))
    cvar_95 = float(r[r <= var_95].mean())

    total_ret = float((1 + r).prod() - 1)

    return {
        'label'      : label,
        'ann_ret'    : ann_ret,
        'ann_vol'    : ann_vol,
        'sharpe'     : sharpe,
        'sortino'    : sortino,
        'calmar'     : calmar,
        'max_dd'     : max_dd,
        'win_rate'   : win_rate,
        'skew'       : skew,
        'kurt'       : kurt,
        'var_95'     : var_95,
        'cvar_95'    : cvar_95,
        'total_ret'  : total_ret,
        'best_month' : float(monthly.max()),
        'worst_month': float(monthly.min()),
        'cum'        : cum,
        'drawdown'   : dd,
        'monthly'    : monthly,
        'n_days'     : n,
    }


def print_stats(d: dict) -> None:
    print(f"  {'─'*44}")
    print(f"  {d['label']:^44}")
    print(f"  {'─'*44}")
    print(f"  {'Annualised Return':<28}: {d['ann_ret']*100:>+8.2f}%")
    print(f"  {'Annualised Volatility':<28}: {d['ann_vol']*100:>8.2f}%")
    print(f"  {'Sharpe Ratio':<28}: {d['sharpe']:>8.3f}")
    print(f"  {'Sortino Ratio':<28}: {d['sortino']:>8.3f}")
    print(f"  {'Calmar Ratio':<28}: {d['calmar']:>8.3f}")
    print(f"  {'Max Drawdown':<28}: {d['max_dd']*100:>8.2f}%")
    print(f"  {'Monthly Win Rate':<28}: {d['win_rate']*100:>8.1f}%")
    print(f"  {'Skewness':<28}: {d['skew']:>8.3f}")
    print(f"  {'Excess Kurtosis':<28}: {d['kurt']:>8.3f}")
    print(f"  {'Daily VaR (95%)':<28}: {d['var_95']*100:>8.3f}%")
    print(f"  {'Daily CVaR (95%)':<28}: {d['cvar_95']*100:>8.3f}%")
    print(f"  {'Best Month':<28}: {d['best_month']*100:>+8.2f}%")
    print(f"  {'Worst Month':<28}: {d['worst_month']*100:>+8.2f}%")
    print(f"  {'Total Return':<28}: {d['total_ret']*100:>+8.2f}%")
    print()


def print_annual_table(annual_mom: pd.Series,
                       annual_dji: pd.Series) -> None:
    print("  Annual Returns")
    print(f"  {'─'*36}")
    print(f"  {'Year':<8} {'Momentum':>10} {'DJI':>10}  {'Spread':>8}")
    print(f"  {'─'*36}")
    for yr in annual_mom.index:
        m   = annual_mom.loc[yr]
        b   = annual_dji.loc[yr] if yr in annual_dji.index else float('nan')
        spr = m - b if not (m != m or b != b) else float('nan')
        print(f"  {yr.year:<8} {m*100:>+10.2f}% {b*100:>+9.2f}%  {spr*100:>+7.2f}%")
    print()
