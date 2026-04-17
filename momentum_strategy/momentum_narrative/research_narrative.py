"""
Research narrative: Multi-Factor Beta-Neutral → Long-Only Momentum

Generates three publication-quality figures that tell the story of how
and why we moved from a sophisticated multi-factor L/S strategy to a
simpler long-only momentum approach.

Usage:
    python research_narrative.py

Requires the cached data (run `python -m momentum_strategy` once first
so the WRDS data pickle is populated).
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import momentum_strategy.config as cfg
from momentum_strategy.strategy.engine import backtest_momentum
from momentum_strategy.strategy.mf_engine import backtest_multifactor

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(cfg.BASE, 'momentum_strategy', 'momentum_narrative')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE   = '#2563EB'
ORANGE = '#F59E0B'
RED    = '#DC2626'
GREEN  = '#16A34A'
GRAY   = '#6B7280'
LIGHT  = '#F1F5F9'

plt.rcParams.update({
    'figure.dpi'       : 150,
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.25,
    'grid.linestyle'   : '--',
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_data() -> dict:
    path = os.path.join(
        cfg.CACHE_DIR,
        f'data_{cfg.DOWNLOAD_START}_{cfg.ANALYSIS_END}_v2.pkl'.replace('-', ''),
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            "Run `python -m momentum_strategy` first to populate the cache."
        )
    print(f"  Loading cached data …")
    with open(path, 'rb') as f:
        return pickle.load(f)


def _eq(ret: pd.Series) -> pd.Series:
    return (1 + ret).cumprod()


def _sharpe(ret: pd.Series, rf: pd.Series) -> float:
    excess = ret - rf.reindex(ret.index).fillna(0)
    return float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 1e-10 else 0.0


def _ann_ret(ret: pd.Series) -> float:
    return float((1 + ret).prod() ** (252 / len(ret)) - 1)



# ── Run both backtests ────────────────────────────────────────────────────────

def _run_backtests(data: dict):
    rt = data['returns_train']
    tri = data['tri_train']
    tbill = data['tbill_train']

    print("  Running multi-factor L/S backtest …")
    mf_ret, mf_log = backtest_multifactor(
        tri_df      = tri,
        returns_df  = rt,
        rf_daily    = tbill,
        ceqq_panel  = data['ceqq_panel'].reindex(rt.index),
        niq_panel   = data['niq_panel'].reindex(rt.index),
        cshoq_panel = data['cshoq_panel'].reindex(rt.index),
        prices_df   = data['prices_train'],
        factor_weights = {'momentum': 0.30, 'value': 0.25, 'quality': 0.25, 'low_vol': 0.20},
        n_long=6, n_short=6, tc=cfg.TC,
        sector_map=cfg.SECTOR_MAP,
    )

    print("  Running long-only momentum backtest …")
    lo_ret, _ = backtest_momentum(
        tri, tbill,
        dji_prices         = None,
        formation          = cfg.FORMATION_DAYS,
        formation_short    = cfg.FORMATION_SHORT,
        skip               = cfg.SKIP_DAYS,
        holding            = cfg.HOLDING_DAYS,
        n_long             = cfg.N_LONG,
        n_short            = 0,
        short_notional     = 0.0,
        tc                 = cfg.TC,
        consistency_weight = cfg.CONSISTENCY_WEIGHT,
    )

    dji_ret = data['dji_train'].pct_change().dropna()

    # Align to common trading dates
    idx = mf_ret.index.intersection(lo_ret.index).intersection(dji_ret.index)
    return mf_ret[idx], lo_ret[idx], dji_ret[idx], mf_log


# ── Figure 1: The Journey ─────────────────────────────────────────────────────

def fig_journey(mf_ret, lo_ret, dji_ret):
    """Three equity curves on one panel — the full research arc."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    mf_eq  = _eq(mf_ret)
    lo_eq  = _eq(lo_ret)
    dji_eq = _eq(dji_ret)

    dji_eq.plot(ax=ax, color=ORANGE, lw=1.8, linestyle='--',
                label='DJI Benchmark', zorder=2, alpha=0.9)
    mf_eq.plot(ax=ax,  color=RED,    lw=2.2,
               label='Step 1 — Multi-Factor L/S (Market-Neutral)', zorder=3)
    lo_eq.plot(ax=ax,  color=BLUE,   lw=2.2,
               label=f'Step 2 — Long-Only Momentum (Top {cfg.N_LONG}, Final)', zorder=4)

    ax.axhline(1.0, color=GRAY, linestyle=':', lw=0.8, alpha=0.6)

    # Shade the multi-factor "underperformance zone"
    common_idx = mf_eq.index.intersection(dji_eq.index)
    ax.fill_between(common_idx, mf_eq[common_idx], 1.0,
                    where=(mf_eq[common_idx] < 1.0),
                    color=RED, alpha=0.07, label='MF below par')

    # Annotation: pivot point where we gave up on multi-factor
    pivot = mf_eq.index[mf_eq.index.searchsorted(pd.Timestamp('2019-06-01'))]
    ax.annotate(
        'Persistent losses;\ndecide to simplify',
        xy=(pivot, float(mf_eq.loc[pivot])),
        xytext=(pd.Timestamp('2017-09-01'), float(mf_eq.min()) - 0.04),
        fontsize=8.5, color=RED, ha='center',
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.2),
    )

    # Annotation: 2017-2018 drawdown
    pt_2018 = mf_eq.index[mf_eq.index.searchsorted(pd.Timestamp('2018-01-01'))]
    ax.annotate(
        'Early drawdown\n2017–2018',
        xy=(pt_2018, float(mf_eq.loc[pt_2018])),
        xytext=(pd.Timestamp('2016-03-01'), float(mf_eq.loc[pt_2018]) + 0.12),
        fontsize=8.5, color=RED, ha='center',
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.2),
    )

    # Annotation: 2021-2022 drawdown
    pt_2022 = mf_eq.index[mf_eq.index.searchsorted(pd.Timestamp('2021-09-01'))]
    ax.annotate(
        'Renewed losses\n2021–2022',
        xy=(pt_2022, float(mf_eq.loc[pt_2022])),
        xytext=(pd.Timestamp('2020-03-01'), float(mf_eq.loc[pt_2022]) - 0.12),
        fontsize=8.5, color=RED, ha='center',
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.2),
    )

    ax.set_title(
        'Research Journey: From Sophisticated to Simple\n'
        'Train Set: January 2015 – December 2021',
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.set_ylabel('Growth of $1', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.2f}'))
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    fig.tight_layout()
    out = f'{OUT_DIR}/fig1_journey.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ── Figure 3: Final Strategy ──────────────────────────────────────────────────

def fig_final_strategy(lo_ret, dji_ret):
    """Equity curve for the long-only momentum strategy."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    lo_eq  = _eq(lo_ret)
    dji_eq = _eq(dji_ret)

    dji_eq.plot(ax=ax, color=ORANGE, lw=1.8, linestyle='--',
                label='DJI Benchmark', alpha=0.9)
    lo_eq.plot(ax=ax,  color=BLUE,   lw=2.2,
               label=f'Long-Only Momentum (Top {cfg.N_LONG})')

    common = lo_eq.index.intersection(dji_eq.index)
    ax.fill_between(common, lo_eq[common], dji_eq[common],
                    where=(lo_eq[common] >= dji_eq[common]),
                    color=BLUE, alpha=0.08, label='Outperformance')
    ax.fill_between(common, lo_eq[common], dji_eq[common],
                    where=(lo_eq[common] < dji_eq[common]),
                    color=ORANGE, alpha=0.08, label='Underperformance')

    ax.axhline(1.0, color=GRAY, linestyle=':', lw=0.8)
    ax.set_ylabel('Growth of $1', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.2f}'))
    ax.set_title(
        f'Final Strategy: Long-Only DJIA Momentum (Top {cfg.N_LONG})\n'
        'Train Set: January 2015 – December 2021',
        fontsize=11, fontweight='bold', pad=12,
    )
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = f'{OUT_DIR}/fig3_final_strategy.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('━' * 60)
    print('  Research Narrative: Multi-Factor → Long-Only Momentum')
    print('━' * 60)

    data = _load_data()
    mf_ret, lo_ret, dji_ret, _ = _run_backtests(data)
    tbill = data['tbill_train']

    print()
    print('  Generating figures …')
    fig_journey(mf_ret, lo_ret, dji_ret)
    fig_final_strategy(lo_ret, dji_ret)

    print()
    print(f'  ✓ All figures saved to {OUT_DIR}/')
    print()

    # Quick summary stats
    print('  Summary')
    print('  ' + '─' * 44)
    for label, ret in [('Multi-Factor L/S', mf_ret), (f'Long-Only Mom. (Top {cfg.N_LONG})', lo_ret), ('DJI Benchmark', dji_ret)]:
        print(f'  {label:<28}  Sharpe {_sharpe(ret, tbill):+.2f}  '
              f'Ann {_ann_ret(ret)*100:+.1f}%  '
              f'Total {((1+ret).prod()-1)*100:+.1f}%')


if __name__ == '__main__':
    main()
