"""
Data acquisition: DJIA constituents (from Wikipedia change history),
CRSP daily returns, Fama-French risk-free rate, and CRSP value-weighted market
index (benchmark) — all sourced from WRDS. VIX via Yahoo Finance.
"""

import os
import pickle
import time
import wrds
import yfinance as yf
import pandas as pd
import numpy as np

from momentum_strategy.config import (
    DOWNLOAD_START, ANALYSIS_START, TRAIN_END, ANALYSIS_END, CACHE_DIR,
)
from momentum_strategy.data.compustat import load_fundamentals

# ── Point-in-time DJIA membership spells ─────────────────────────────────────
# Derived from Wikipedia: "Historical components of the Dow Jones Industrial Average"
# Each tuple: (ticker, membership_start, membership_end)
# Clipped to our download window 2013-01-01 – 2026-04-11.
DJIA_SPELLS = [
    # Continuous members throughout the full download window
    ('MMM',  '2013-01-01', '2026-04-11'),
    ('AXP',  '2013-01-01', '2026-04-11'),
    ('BA',   '2013-01-01', '2026-04-11'),
    ('CAT',  '2013-01-01', '2026-04-11'),
    ('CVX',  '2013-01-01', '2026-04-11'),
    ('CSCO', '2013-01-01', '2026-04-11'),
    ('KO',   '2013-01-01', '2026-04-11'),
    ('DIS',  '2013-01-01', '2026-04-11'),
    ('HD',   '2013-01-01', '2026-04-11'),
    ('IBM',  '2013-01-01', '2026-04-11'),
    ('JNJ',  '2013-01-01', '2026-04-11'),
    ('JPM',  '2013-01-01', '2026-04-11'),
    ('MCD',  '2013-01-01', '2026-04-11'),
    ('MRK',  '2013-01-01', '2026-04-11'),
    ('MSFT', '2013-01-01', '2026-04-11'),
    ('PG',   '2013-01-01', '2026-04-11'),
    ('TRV',  '2013-01-01', '2026-04-11'),
    ('UNH',  '2013-01-01', '2026-04-11'),
    ('VZ',   '2013-01-01', '2026-04-11'),
    ('WMT',  '2013-01-01', '2026-04-11'),
    # Removed 2013-09-23: replaced by GS, NKE, V
    ('AA',   '2013-01-01', '2013-09-22'),
    ('BAC',  '2013-01-01', '2013-09-22'),
    ('HPQ',  '2013-01-01', '2013-09-22'),
    # Added 2013-09-23
    ('GS',   '2013-09-23', '2026-04-11'),
    ('NKE',  '2013-09-23', '2026-04-11'),
    ('V',    '2013-09-23', '2026-04-11'),
    # AT&T removed 2015-03-19 (replaced by AAPL)
    ('T',    '2013-01-01', '2015-03-18'),
    ('AAPL', '2015-03-19', '2026-04-11'),
    # DuPont removed 2017-09-01; DowDuPont (DWDP) added; replaced by DOW 2019-04-02
    ('DD',   '2013-01-01', '2017-08-31'),
    ('DWDP', '2017-09-01', '2019-04-01'),
    ('DOW',  '2019-04-02', '2024-11-07'),
    # GE removed 2018-06-26 (replaced by WBA); WBA removed 2024-02-26 (replaced by AMZN)
    ('GE',   '2013-01-01', '2018-06-25'),
    ('WBA',  '2018-06-26', '2024-02-25'),
    ('AMZN', '2024-02-26', '2026-04-11'),
    # UTX removed 2020-04-06 (became RTX); RTX removed 2020-08-31
    ('UTX',  '2013-01-01', '2020-04-05'),
    ('RTX',  '2020-04-06', '2020-08-30'),
    # XOM and PFE removed 2020-08-31
    ('XOM',  '2013-01-01', '2020-08-30'),
    ('PFE',  '2013-01-01', '2020-08-30'),
    # Added 2020-08-31
    ('AMGN', '2020-08-31', '2026-04-11'),
    ('HON',  '2020-08-31', '2026-04-11'),
    ('CRM',  '2020-08-31', '2026-04-11'),
    # INTC removed 2024-11-08 (replaced by NVDA)
    ('INTC', '2013-01-01', '2024-11-07'),
    ('NVDA', '2024-11-08', '2026-04-11'),
    # SHW added 2024-11-08 (replaced DOW)
    ('SHW',  '2024-11-08', '2026-04-11'),
]


def _resolve_permnos(db, spells: list) -> pd.DataFrame:
    """
    Look up CRSP permnos for each (ticker, start, end) spell via stocknames.
    Returns a DataFrame with columns: permno, ticker, spell_start, spell_end.
    """
    all_tickers = list({t for t, _, _ in spells})
    tickers_sql = ','.join(f"'{t}'" for t in all_tickers)

    stocknames = db.raw_sql(f"""
        SELECT permno, ticker, namedt, nameenddt
        FROM   crsp_q_stock.stocknames
        WHERE  ticker IN ({tickers_sql})
        ORDER  BY ticker, namedt
    """, date_cols=['namedt', 'nameenddt'])

    stocknames['nameenddt'] = stocknames['nameenddt'].fillna(pd.Timestamp('2099-12-31'))

    records = []
    for ticker, spell_start, spell_end in spells:
        s = pd.Timestamp(spell_start)
        e = pd.Timestamp(spell_end)
        candidates = stocknames[
            (stocknames['ticker'] == ticker) &
            (stocknames['namedt'] <= e) &
            (stocknames['nameenddt'] >= s)
        ]
        if candidates.empty:
            print(f"  WARNING: no permno found for {ticker} ({spell_start} – {spell_end})")
            continue
        # Use the permno with the most recent namedt (active near end of spell)
        permno = int(candidates.sort_values('namedt').iloc[-1]['permno'])
        records.append({
            'permno'     : permno,
            'ticker'     : ticker,
            'spell_start': s,
            'spell_end'  : e,
        })

    return pd.DataFrame(records)


def load_data(force_reload: bool = False) -> dict:
    """
    Download and preprocess all data needed by the backtest.

    Results are cached in CACHE_DIR as a pickle file keyed to the date range.
    Pass force_reload=True to bypass the cache and re-download everything.

    Returns a dict with keys:
        tri_train, tri_prices         – Total Return Index (masked to DJIA membership)
        returns_train, returns        – daily CRSP returns (masked)
        prices_train, prices          – closing prices (masked, ffilled)
        tbill_train, tbill_daily      – daily T-bill rate from Fama-French
        dji_train, dji                – ^DJI index level (Yahoo Finance)
        vix_train, vix                – ^VIX level (Yahoo Finance)
    """
    cache_file = os.path.join(
        CACHE_DIR,
        f'data_{DOWNLOAD_START}_{ANALYSIS_END}_v2.pkl'.replace('-', ''),
    )

    if not force_reload and os.path.exists(cache_file):
        print(f"  Loading data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        if all(cached.get(k) is not None for k in ('returns', 'dji', 'tbill_daily', 'ceqq_panel')):
            return cached
        print("  Cache is incomplete — re-downloading …")
        os.remove(cache_file)

    print("━" * 60)
    print("  Connecting to WRDS …")
    print("━" * 60)

    db = wrds.Connection()

    # ── Resolve DJIA membership spells → CRSP permnos ────────────
    print("  Resolving DJIA membership spells to CRSP permnos …")
    spell_df   = _resolve_permnos(db, DJIA_SPELLS)
    all_permnos = spell_df['permno'].unique().tolist()
    permno_sql  = ','.join(str(p) for p in all_permnos)
    print(f"  Unique permnos resolved : {len(all_permnos)}")

    # ── Daily stock data from CRSP ────────────────────────────────
    print("  Downloading CRSP daily stock file …")
    crsp_daily = db.raw_sql(f"""
        SELECT date, permno, ret, ABS(prc) AS prc, shrout
        FROM   crsp_q_stock.dsf
        WHERE  date BETWEEN '{DOWNLOAD_START}' AND '{ANALYSIS_END}'
          AND  permno IN ({permno_sql})
        ORDER  BY permno, date
    """, date_cols=['date'])

    # ── Fama-French daily risk-free rate ──────────────────────────
    print("  Downloading Fama-French daily risk-free rate …")
    ff_rf = db.raw_sql(f"""
        SELECT date, rf
        FROM   ff.factors_daily
        WHERE  date BETWEEN '{ANALYSIS_START}' AND '{ANALYSIS_END}'
        ORDER  BY date
    """, date_cols=['date'])

    # # ── CRSP value-weighted market index (benchmark) ─────────────
    # print("  Downloading CRSP value-weighted market index …")
    # crsp_dsi = db.raw_sql(f"""
    #     SELECT date, vwretd
    #     FROM   crsp_q_stock.dsi
    #     WHERE  date BETWEEN '{ANALYSIS_START}' AND '{ANALYSIS_END}'
    #     ORDER  BY date
    # """, date_cols=['date'])
    # crsp_dsi = crsp_dsi.set_index('date')['vwretd']
    # # Convert daily returns to a cumulative price index (base = 1.0)
    # dji       = (1 + crsp_dsi.fillna(0)).cumprod()
    # dji_train = dji.loc[:TRAIN_END]

    # ── Actual Dow Jones Industrial Average benchmark via Yahoo Finance ─────────────
    print("  Downloading ^DJI from Yahoo Finance …")
    
    # Download daily closing prices for the Dow
    dji_prices = yf.download('^DJI', start=ANALYSIS_START, end=ANALYSIS_END, progress=False)['Close']
    
    # Handle yfinance occasionally returning a DataFrame instead of a Series
    if isinstance(dji_prices, pd.DataFrame):
        dji_prices = dji_prices.iloc[:, 0]
        
    # Calculate daily returns and convert to a cumulative wealth index (base = 1.0)
    # This ensures it matches the exact mathematical format your strategy engine expects
    dji_returns = dji_prices.pct_change().fillna(0)
    dji         = (1 + dji_returns).cumprod()
    dji_train   = dji.loc[:TRAIN_END]

    # ── Compustat quarterly fundamentals ─────────────────────────
    # Derive the daily date index from CRSP before closing the connection
    crsp_daily_dates = pd.DatetimeIndex(
        sorted(crsp_daily.loc[
            crsp_daily['date'] >= pd.Timestamp(ANALYSIS_START), 'date'
        ].unique())
    )
    print("  Downloading Compustat quarterly fundamentals …")
    fund_result = load_fundamentals(
        db, spell_df, DOWNLOAD_START, ANALYSIS_END,
        daily_index=crsp_daily_dates,
    )

    db.close()
    print("  WRDS connection closed.")

    # ── VIX via Yahoo Finance (regime filter only) ────────────────
    print("  Downloading ^VIX from Yahoo Finance …")
    vix = None
    delays = [15, 30, 60, 120, 180, 300]
    for attempt in range(6):
        try:
            raw = yf.download('^VIX', start=ANALYSIS_START, end=ANALYSIS_END,
                              auto_adjust=True, progress=False)['Close']
            if raw is not None and not raw.empty:
                if isinstance(raw, pd.DataFrame):
                    raw = raw.iloc[:, 0]
                vix = raw.loc[ANALYSIS_START:ANALYSIS_END]
                break
        except Exception:
            pass
        wait = delays[min(attempt, len(delays) - 1)]
        print(f"  ^VIX attempt {attempt+1} failed, retrying in {wait}s …")
        time.sleep(wait)
    vix_train = vix.loc[:TRAIN_END] if vix is not None else None
    if vix is None:
        print("  ^VIX unavailable — VIX regime filter will be disabled.")

    # ── Pivot CRSP to wide format (date × permno) ─────────────────
    returns_wide = (crsp_daily
                    .set_index(['date', 'permno'])['ret']
                    .unstack('permno')
                    .sort_index())

    prices_wide = (crsp_daily
                   .set_index(['date', 'permno'])['prc']
                   .unstack('permno')
                   .sort_index())

    shrout_wide = (crsp_daily
                   .set_index(['date', 'permno'])['shrout']
                   .unstack('permno')
                   .sort_index()
                   .ffill())

    prices_wide = prices_wide.ffill()

    # ── Point-in-time membership mask ────────────────────────────
    print("  Building point-in-time membership mask …")
    membership = pd.DataFrame(False,
                              index=returns_wide.index,
                              columns=returns_wide.columns)

    for _, row in spell_df.iterrows():
        p = row['permno']
        if p not in membership.columns:
            continue
        active = ((membership.index >= row['spell_start']) &
                  (membership.index <= row['spell_end']))
        membership.loc[active, p] = True

    returns_masked = returns_wide.where(membership)
    prices_masked  = prices_wide.where(membership)
    shrout_masked  = shrout_wide.where(membership)

    # ── Total Return Index ────────────────────────────────────────
    ret_fill   = returns_wide.fillna(0)
    tri        = (1 + ret_fill).cumprod()
    tri_masked = tri.where(membership)

    # ── Rename columns permno → ticker ───────────────────────────
    # Use the last (most recent) ticker recorded for each permno across spells.
    permno_to_ticker = (spell_df
                        .sort_values('spell_start')
                        .drop_duplicates('permno', keep='last')
                        .set_index('permno')['ticker']
                        .to_dict())

    def relabel(df):
        df.columns = [permno_to_ticker.get(c, str(c)) for c in df.columns]
        return df

    tri_masked     = relabel(tri_masked)
    returns_masked = relabel(returns_masked)
    prices_masked  = relabel(prices_masked)
    shrout_masked  = relabel(shrout_masked)

    # ── Trim to analysis / train windows ─────────────────────────
    tri_prices    = tri_masked.loc[ANALYSIS_START:ANALYSIS_END]
    returns       = returns_masked.loc[ANALYSIS_START:ANALYSIS_END]
    prices        = prices_masked.loc[ANALYSIS_START:ANALYSIS_END]

    tri_train     = tri_prices.loc[:TRAIN_END]
    returns_train = returns.loc[:TRAIN_END]
    prices_train  = prices.loc[:TRAIN_END]

    tbill_daily = (ff_rf
                   .set_index('date')['rf']
                   .reindex(returns.index)
                   .ffill()
                   .fillna(0))
    tbill_train = tbill_daily.loc[:TRAIN_END]

    # ── Summary ───────────────────────────────────────────────────
    active_in_window = returns.notna().any()
    print(f"  Unique DJIA members in window : {active_in_window.sum()} tickers")
    print(f"  Analysis period : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  Train set       : {returns_train.index[0].date()} → "
          f"{returns_train.index[-1].date()}  ({len(returns_train)} trading days)")
    missing_pct = returns_train.isnull().sum().sum() / returns_train.size * 100
    print(f"  Missing data    : {missing_pct:.2f}%  (non-membership periods)")
    print()

    shrout        = shrout_masked.loc[ANALYSIS_START:ANALYSIS_END]

    result = {
        'tri_prices'   : tri_prices,
        'tri_train'    : tri_train,
        'returns'      : returns,
        'returns_train': returns_train,
        'prices'       : prices,
        'prices_train' : prices_train,
        'shrout'       : shrout,
        'tbill_daily'  : tbill_daily,
        'tbill_train'  : tbill_train,
        'dji'          : dji,
        'dji_train'    : dji_train,
        'vix'          : vix,
        'vix_train'    : vix_train,
        'ceqq_panel'   : fund_result['ceqq_panel'],
        'niq_panel'    : fund_result['niq_panel'],
        'cshoq_panel'  : fund_result['cshoq_panel'],
    }

    print(f"  Saving data to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return result
