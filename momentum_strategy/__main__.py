#!/usr/bin/env python3
"""
Entry point: python -m momentum_strategy

Dispatches between the original momentum strategy and the new multi-factor
market-neutral strategy based on config.STRATEGY_NAME.
"""

import os
import json
import numpy as np
from scipy import stats
import pandas as pd

import momentum_strategy.config as cfg
from momentum_strategy.data.loader import load_data
from momentum_strategy.analysis.metrics import (
    performance_stats, print_stats, print_annual_table,
)
from momentum_strategy.visualization.plots import generate_all_figures


def _run_momentum(data: dict) -> tuple:
    from momentum_strategy.strategy.engine import backtest_momentum
    long_only = (cfg.N_SHORT == 0)
    port_ret, rebal_log = backtest_momentum(
        data['tri_train'], data['tbill_train'],
        dji_prices         = None if long_only else data['dji_train'],
        formation          = cfg.FORMATION_DAYS,
        formation_short    = cfg.FORMATION_SHORT,
        skip               = cfg.SKIP_DAYS,
        holding            = cfg.HOLDING_DAYS,
        n_long             = cfg.N_LONG,
        n_short            = cfg.N_SHORT,
        short_notional     = 0.0 if long_only else 0.5,
        tc                 = cfg.TC,
        consistency_weight = cfg.CONSISTENCY_WEIGHT,
    )
    label = "Momentum Long-Only" if long_only else "Momentum L/S Dollar-Neutral"
    return port_ret, rebal_log, cfg.OUTPUT_DIR, label


def _run_multifactor(data: dict) -> tuple:
    from momentum_strategy.strategy.mf_engine import backtest_multifactor
    port_ret, rebal_log = backtest_multifactor(
        tri_df        = data['tri_train'],
        returns_df    = data['returns_train'],
        rf_daily      = data['tbill_train'],
        ceqq_panel    = data['ceqq_panel'].reindex(data['returns_train'].index),
        niq_panel     = data['niq_panel'].reindex(data['returns_train'].index),
        cshoq_panel   = data['cshoq_panel'].reindex(data['returns_train'].index),
        prices_df     = data['prices_train'],
        factor_weights = cfg.FACTOR_WEIGHTS,
        n_long        = cfg.N_LONG_MF,
        n_short       = cfg.N_SHORT_MF,
        tc            = cfg.TC,
        sector_map    = cfg.SECTOR_MAP,
        max_sector_tilt = cfg.MAX_SECTOR_TILT,
    )
    return port_ret, rebal_log, cfg.OUTPUT_MF_DIR, "Multi-Factor L/S Market-Neutral"


def run():
    # ── 1. Load data ──────────────────────────────────────────────
    data = load_data()
    
    # ── 2. Handle OOS Testing Window ──────────────────────────────
    mode = getattr(cfg, 'BACKTEST_MODE', 'train')
    
    if mode == 'test':
        print(f"  Mode: OUT-OF-SAMPLE ({cfg.TEST_START} to {cfg.ANALYSIS_END})")
        
        # Pad the start date by 1 year to feed the 252-day lookback formation window.
        # This ensures the first actual trading day lands right at TEST_START.
        pad_start = str(int(cfg.TEST_START[:4]) - 1) + cfg.TEST_START[4:]
        
        # Overwrite the 'train' keys with our test slice so downstream functions work seamlessly
        data['tri_train']     = data['tri_prices'].loc[pad_start:]
        data['returns_train'] = data['returns'].loc[pad_start:]
        data['tbill_train']   = data['tbill_daily'].loc[pad_start:]
        data['dji_train']     = data['dji'].loc[pad_start:]
        data['prices_train']  = data['prices'].loc[pad_start:]
        
        # Append '_oos' to output directories to prevent overwriting your train results
        cfg.OUTPUT_DIR    = f"{cfg.OUTPUT_DIR}_oos"
        cfg.OUTPUT_MF_DIR = f"{cfg.OUTPUT_MF_DIR}_oos"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cfg.OUTPUT_MF_DIR, exist_ok=True)
    else:
        print(f"  Mode: TRAIN-SET ({cfg.ANALYSIS_START} to {cfg.TRAIN_END})")

    # ── 3. Dispatch to strategy ───────────────────────────────────
    print(f"  Strategy mode: {cfg.STRATEGY_NAME}")
    print("  Running backtest …")

    if cfg.STRATEGY_NAME == 'multi_factor':
        port_ret, rebal_log, output_dir, strat_label = _run_multifactor(data)
    else:
        port_ret, rebal_log, output_dir, strat_label = _run_momentum(data)

    print(f"  Backtest complete: {len(port_ret)} daily observations, "
          f"{len(rebal_log)} rebalances")
    print()

    # ── 4. Align benchmark ────────────────────────────────────────
    dji_train        = data['dji_train']
    dji_ret          = dji_train.pct_change().dropna().reindex(port_ret.index).dropna()
    port_ret_aligned = port_ret.reindex(dji_ret.index).dropna()
    tbill_train      = data['tbill_train']

    # ── 5. Performance statistics ─────────────────────────────────
    s_strat = performance_stats(port_ret_aligned, tbill_train, strat_label)
    s_dji   = performance_stats(dji_ret,          tbill_train, "DJI Benchmark")

    ols         = stats.linregress(dji_ret.values, port_ret_aligned.values)
    beta_vs_dji = ols.slope
    alpha_ann   = ols.intercept * 252
    corr_dji    = port_ret_aligned.corr(dji_ret)

    print_stats(s_strat)
    print_stats(s_dji)
    print(f"  {'Market Exposure (vs DJI)'}")
    print(f"  {'─'*44}")
    print(f"  {'Beta':<28}: {beta_vs_dji:>8.3f}")
    print(f"  {'Alpha (annualised)':<28}: {alpha_ann*100:>+8.2f}%")
    print(f"  {'Correlation':<28}: {corr_dji:>8.3f}")
    print(f"  {'R-squared':<28}: {ols.rvalue**2:>8.3f}")
    print()

    annual_strat = port_ret_aligned.resample('YE').apply(lambda x: (1+x).prod()-1)
    annual_dji   = dji_ret.resample('YE').apply(lambda x: (1+x).prod()-1)
    print_annual_table(annual_strat, annual_dji)

    # ── 6. Figures ────────────────────────────────────────────────
    print("  Generating figures …")
    generate_all_figures(
        s_strat, s_dji,
        port_ret_aligned, dji_ret,
        annual_strat, annual_dji,
        beta_vs_dji, alpha_ann, ols,
        output_dir=output_dir,
    )

    # ── 7. Save JSON results ──────────────────────────────────────
    results = {
        'strategy': {k: float(v) for k, v in s_strat.items()
                     if isinstance(v, (int, float, np.floating))},
        'benchmark': {k: float(v) for k, v in s_dji.items()
                       if isinstance(v, (int, float, np.floating))},
        'market_exposure': {
            'beta_vs_dji': float(beta_vs_dji),
            'alpha_ann'  : float(alpha_ann),
            'corr_dji'   : float(corr_dji),
            'r_squared'  : float(ols.rvalue**2),
        },
        'annual': {
            str(yr.year): {
                'strategy': float(annual_strat.get(yr, float('nan'))),
                'dji'     : float(annual_dji.get(yr, float('nan'))),
            }
            for yr in annual_strat.index
        },
    }

    out_path = f'{output_dir}/backtest_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Statistics saved to {out_path}")
    print()
    print("  Done.")


if __name__ == '__main__':
    run()
