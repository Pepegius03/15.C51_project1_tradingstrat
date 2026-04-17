"""
Publication-quality figures for the momentum strategy backtest.
All figures are saved to FIG_DIR defined in config.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import warnings
from scipy import stats

from momentum_strategy.config import FIG_DIR as _DEFAULT_FIG_DIR

warnings.filterwarnings('ignore')

_active_fig_dir = _DEFAULT_FIG_DIR   # overridden by generate_all_figures when needed

plt.rcParams.update({
    'figure.dpi'       : 150,
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

BLUE   = '#2563EB'
ORANGE = '#F59E0B'
RED    = '#DC2626'
GREEN  = '#16A34A'
GRAY   = '#6B7280'


def generate_all_figures(s_mom: dict,
                         s_dji: dict,
                         port_ret_aligned,
                         dji_ret,
                         annual_mom,
                         annual_dji,
                         beta_vs_dji: float,
                         alpha_ann: float,
                         ols,
                         output_dir: str = None) -> None:
    """Generate and save all 8 strategy figures."""
    global _active_fig_dir
    import os
    if output_dir is not None:
        fig_dir = os.path.join(output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        _active_fig_dir = fig_dir
    else:
        _active_fig_dir = _DEFAULT_FIG_DIR

    _fig1_cumulative(s_mom, s_dji)
    _fig2_drawdown(s_mom)
    _fig3_rolling_sharpe(s_mom, port_ret_aligned)
    _fig4_monthly_heatmap(s_mom)
    _fig5_annual_returns(annual_mom, annual_dji)
    _fig6_distributions(s_mom, port_ret_aligned)
    _fig7_scatter(dji_ret, port_ret_aligned, ols, beta_vs_dji, alpha_ann)
    _fig8_equity_drawdown(s_mom)
    print(f"  All figures saved to {_active_fig_dir}/")


def _fig1_cumulative(s_mom, s_dji):
    fig, ax = plt.subplots(figsize=(12, 5))
    s_mom['cum'].plot(ax=ax, color=BLUE,   lw=2.0, label='Momentum L/S Dollar-Neutral')
    s_dji['cum'].plot(ax=ax, color=ORANGE, lw=1.8, linestyle='--', label='DJI Benchmark')
    ax.axhline(1, color=GRAY, linestyle=':', lw=0.8, alpha=0.7)
    ax.set_title('Cumulative Returns — Momentum L/S Dollar-Neutral vs. DJI\n'
                 'Train Set: January 2015 – December 2021',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Growth of $1', fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.2f}'))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'{_active_fig_dir}/fig1_cumulative_returns.png', bbox_inches='tight')
    plt.close()


def _fig2_drawdown(s_mom):
    fig, ax = plt.subplots(figsize=(12, 4))
    s_mom['drawdown'].plot(ax=ax, color=RED, lw=1.5)
    ax.fill_between(s_mom['drawdown'].index, s_mom['drawdown'].values, 0,
                    alpha=0.25, color=RED)
    ax.set_title('Strategy Drawdown — Train Set: 2015–2021',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Drawdown', fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'{_active_fig_dir}/fig2_drawdown.png', bbox_inches='tight')
    plt.close()


def _fig3_rolling_sharpe(s_mom, port_ret_aligned):
    rolling_sharpe = port_ret_aligned.rolling(252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan,
        raw=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    rolling_sharpe.plot(ax=ax, color=BLUE, lw=1.5, label='Rolling 1-yr Sharpe')
    ax.axhline(0, color=GRAY, linestyle='--', lw=0.8)
    ax.axhline(s_mom['sharpe'], color=GREEN, linestyle='--', lw=1.2,
               label=f"Full-period Sharpe = {s_mom['sharpe']:.2f}")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=(rolling_sharpe.values > 0), alpha=0.2, color=GREEN)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=(rolling_sharpe.values <= 0), alpha=0.2, color=RED)
    ax.set_title('Rolling 12-Month Sharpe Ratio — Train Set: 2015–2021',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'{_active_fig_dir}/fig3_rolling_sharpe.png', bbox_inches='tight')
    plt.close()


def _fig4_monthly_heatmap(s_mom):
    import pandas as pd
    mo_ret = s_mom['monthly']
    mo_df  = pd.DataFrame({
        'Year' : mo_ret.index.year,
        'Month': mo_ret.index.month,
        'Ret'  : mo_ret.values
    }).pivot(index='Year', columns='Month', values='Ret')
    mo_df.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                     'Jul','Aug','Sep','Oct','Nov','Dec']
    abs_max = np.nanmax(np.abs(mo_df.values))
    norm    = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(mo_df.values, cmap='RdYlGn', norm=norm, aspect='auto')
    ax.set_xticks(range(12)); ax.set_xticklabels(mo_df.columns, fontsize=10)
    ax.set_yticks(range(len(mo_df))); ax.set_yticklabels(mo_df.index, fontsize=10)
    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('Monthly Return', fontsize=9)
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    for i in range(len(mo_df)):
        for j in range(12):
            v = mo_df.iloc[i, j]
            if not np.isnan(v):
                color = 'white' if abs(v) > abs_max * 0.5 else 'black'
                ax.text(j, i, f'{v*100:.1f}%', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
    ax.set_title('Monthly Returns Heatmap — Momentum L/S Dollar-Neutral (Train Set: 2015–2021)',
                 fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(f'{_active_fig_dir}/fig4_monthly_heatmap.png', bbox_inches='tight')
    plt.close()


def _fig5_annual_returns(annual_mom, annual_dji):
    years = annual_mom.index.year
    x, w  = np.arange(len(years)), 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_colors = [GREEN if v > 0 else RED for v in annual_mom.values]
    b1 = ax.bar(x - w/2, annual_mom.values*100, w, color=bar_colors,
                alpha=0.85, label='Momentum L/S Dollar-Neutral', edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, annual_dji.values*100, w, color=ORANGE,
           alpha=0.75, label='DJI Benchmark', edgecolor='white', linewidth=0.5)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h + (0.4 if h >= 0 else -1.2),
                f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(years, fontsize=10)
    ax.set_ylabel('Annual Return (%)', fontsize=11)
    ax.set_title('Annual Returns — Momentum Strategy vs. DJI (Train Set: 2015–2021)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f'{_active_fig_dir}/fig5_annual_returns.png', bbox_inches='tight')
    plt.close()


def _fig6_distributions(s_mom, port_ret_aligned):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ret, title_sfx in zip(axes,
            [port_ret_aligned, s_mom['monthly']],
            ['Daily Returns', 'Monthly Returns']):
        data = ret.values * 100
        ax.hist(data, bins=50, color=BLUE, alpha=0.65, edgecolor='white',
                density=True, label='Empirical')
        mu_fit, sig_fit = np.mean(data), np.std(data)
        x_fit = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_fit, stats.norm.pdf(x_fit, mu_fit, sig_fit),
                color=RED, lw=2, label='Normal fit')
        if title_sfx == 'Daily Returns':
            var_line = s_mom['var_95'] * 100
            ax.axvline(var_line, color=ORANGE, lw=1.8, linestyle='--',
                       label=f'VaR 95%: {var_line:.2f}%')
        ax.set_title(f'{title_sfx} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{title_sfx} (%)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        sk = stats.skew(ret.values)
        ku = stats.kurtosis(ret.values, fisher=True)
        ax.text(0.97, 0.95, f'Skew = {sk:.2f}\nKurt = {ku:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.suptitle('Return Distributions — Momentum L/S Dollar-Neutral (Train Set: 2015–2021)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{_active_fig_dir}/fig6_distributions.png', bbox_inches='tight')
    plt.close()


def _fig7_scatter(dji_ret, port_ret_aligned, ols, beta_vs_dji, alpha_ann):
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(dji_ret*100, port_ret_aligned*100,
               alpha=0.25, s=12, color=BLUE, edgecolors='none')
    xl = np.linspace(dji_ret.min()*100, dji_ret.max()*100, 200)
    ax.plot(xl, ols.intercept*100 + ols.slope*xl, color=RED, lw=2,
            label=f'β = {beta_vs_dji:.2f},  α = {alpha_ann*100:+.1f}%/yr')
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.axvline(0, color=GRAY, lw=0.7)
    ax.set_xlabel('DJI Daily Return (%)', fontsize=11)
    ax.set_ylabel('Strategy Daily Return (%)', fontsize=11)
    ax.set_title('Strategy Returns vs. DJI (Train Set)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f'{_active_fig_dir}/fig7_scatter_beta.png', bbox_inches='tight')
    plt.close()


def _fig8_equity_drawdown(s_mom):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    ax_cum, ax_dd = axes
    s_mom['cum'].plot(ax=ax_cum, color=BLUE, lw=2)
    ax_cum.set_ylabel('Cumulative Return ($)', fontsize=10)
    ax_cum.set_title('Equity Curve and Drawdown — Train Set: 2015–2021',
                     fontsize=13, fontweight='bold')
    s_mom['drawdown'].plot(ax=ax_dd, color=RED, lw=1.3)
    ax_dd.fill_between(s_mom['drawdown'].index, s_mom['drawdown'].values, 0,
                       alpha=0.25, color=RED)
    ax_dd.set_ylabel('Drawdown', fontsize=10)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax_dd.set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'{_active_fig_dir}/fig8_equity_and_drawdown.png', bbox_inches='tight')
    plt.close()
