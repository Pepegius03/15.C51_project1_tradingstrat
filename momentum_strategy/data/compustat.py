"""
Compustat quarterly fundamental loader with point-in-time alignment.

Fetches book equity (ceqq) and net income (niq) for DJIA constituents via the
CRSP/Compustat Merged (CCM) link table, then aligns each observation to the
trading-day panel using only data that was publicly available as of each date.

Look-ahead bias guard
---------------------
The availability date for each quarterly observation is:

    avail_date = max(rdq + fiscal_lag, datadate + fiscal_lag)

where ``rdq`` is the actual earnings announcement date (the first date the
market could have seen the number) and ``fiscal_lag`` (default 60 days) is an
additional buffer for filings that lag the announcement.  When ``rdq`` is
missing we fall back to ``datadate + fiscal_lag``.

This ensures that, for example, a Q1-2020 filing (datadate 2020-03-31,
rdq ≈ 2020-05-01) does not appear in the factor until around 2020-07-01.
"""

import os
import pickle
import numpy as np
import pandas as pd

from momentum_strategy.config import CACHE_DIR, FISCAL_LAG_DAYS


def load_fundamentals(
    db,
    spell_df: pd.DataFrame,
    download_start: str,
    analysis_end: str,
    daily_index: pd.DatetimeIndex,
    fiscal_lag: int = FISCAL_LAG_DAYS,
) -> dict:
    """
    Download Compustat quarterly fundamentals and build point-in-time panels.

    Parameters
    ----------
    db             : open wrds.Connection
    spell_df       : DataFrame with columns [permno, ticker, spell_start, spell_end]
    download_start : earliest date to pull from Compustat (e.g. '2013-01-01')
    analysis_end   : latest date (e.g. '2025-04-11')
    daily_index    : pd.DatetimeIndex of all trading dates in the analysis window
    fiscal_lag     : days added to max(rdq, datadate) before a row becomes visible

    Returns
    -------
    dict with keys:
        'ceqq_panel' : pd.DataFrame  (date × ticker)  point-in-time book equity
        'niq_panel'  : pd.DataFrame  (date × ticker)  point-in-time net income (quarterly)
        'cshoq_panel': pd.DataFrame  (date × ticker)  point-in-time shares outstanding (000s)
    """
    cache_file = os.path.join(
        CACHE_DIR,
        f'fundamentals_{download_start}_{analysis_end}.pkl'.replace('-', ''),
    )

    if os.path.exists(cache_file):
        print(f"  Loading fundamentals from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("  Downloading Compustat fundamentals via CCM link …")

    # ── Step 1: CRSP → Compustat link ────────────────────────────────────────
    all_permnos = spell_df['permno'].unique().tolist()
    permno_sql  = ','.join(str(p) for p in all_permnos)

    link_df = db.raw_sql(f"""
        SELECT lpermno AS permno,
               gvkey,
               linkdt,
               linkenddt,
               linktype,
               linkprim
        FROM   crsp.ccmxpf_lnkhist
        WHERE  lpermno IN ({permno_sql})
          AND  linktype IN ('LU', 'LC', 'LS')
          AND  linkprim IN ('P', 'C', 'J')
        ORDER  BY lpermno, linkdt
    """, date_cols=['linkdt', 'linkenddt'])

    if link_df.empty:
        print("  WARNING: No CCM links found — fundamentals unavailable.")
        empty = pd.DataFrame(index=daily_index)
        return {'ceqq_panel': empty, 'niq_panel': empty, 'cshoq_panel': empty}

    link_df['linkenddt'] = link_df['linkenddt'].fillna(pd.Timestamp('2099-12-31'))

    # Warn about permnos without any link
    linked_permnos = set(link_df['permno'].unique())
    for _, row in spell_df.iterrows():
        if row['permno'] not in linked_permnos:
            print(f"  WARNING: No CCM link for {row['ticker']} (permno {row['permno']})")

    all_gvkeys = link_df['gvkey'].unique().tolist()
    gvkey_sql  = ','.join(f"'{g}'" for g in all_gvkeys)

    # ── Step 2: Compustat quarterly fundamentals ──────────────────────────────
    # Try common WRDS schema names for Compustat North America quarterly data.
    _fundq_schemas = ['comp.fundq', 'compa.fundq', 'comp_na_annual_all.fundq']
    fundq = None
    for schema in _fundq_schemas:
        try:
            fundq = db.raw_sql(f"""
                SELECT gvkey, datadate, rdq, ceqq, niq, cshoq
                FROM   {schema}
                WHERE  gvkey IN ({gvkey_sql})
                  AND  datadate BETWEEN '{download_start}' AND '{analysis_end}'
                  AND  ceqq IS NOT NULL
                ORDER  BY gvkey, datadate
            """, date_cols=['datadate', 'rdq'])
            print(f"  Compustat fundq found at schema: {schema}")
            break
        except Exception:
            continue
    if fundq is None:
        raise RuntimeError(
            "Could not find Compustat fundq table. "
            f"Tried: {_fundq_schemas}. "
            "Check your WRDS Compustat subscription."
        )

    if fundq.empty:
        print("  WARNING: Compustat returned no rows.")
        empty = pd.DataFrame(index=daily_index)
        return {'ceqq_panel': empty, 'niq_panel': empty, 'cshoq_panel': empty}

    # ── Step 3: Map gvkey → permno → ticker (point-in-time via link dates) ───
    # For each (gvkey, datadate) observation, find the permno whose link was
    # active at the time of the datadate, then map to ticker via spell_df.
    permno_to_ticker = (
        spell_df
        .sort_values('spell_start')
        .drop_duplicates('permno', keep='last')
        .set_index('permno')['ticker']
        .to_dict()
    )

    records = []
    for _, frow in fundq.iterrows():
        gvkey    = frow['gvkey']
        datadate = frow['datadate']
        rdq      = frow['rdq']

        # Find permno active at datadate for this gvkey
        active_links = link_df[
            (link_df['gvkey'] == gvkey) &
            (link_df['linkdt'] <= datadate) &
            (link_df['linkenddt'] >= datadate)
        ]
        if active_links.empty:
            # Relax: use any link that overlaps within ±1 year of datadate
            active_links = link_df[
                (link_df['gvkey'] == gvkey) &
                (link_df['linkdt'] <= datadate + pd.DateOffset(years=1)) &
                (link_df['linkenddt'] >= datadate - pd.DateOffset(years=1))
            ]
        if active_links.empty:
            continue

        permno = int(active_links.sort_values('linkdt').iloc[-1]['permno'])
        ticker = permno_to_ticker.get(permno)
        if ticker is None:
            continue

        # Point-in-time availability date
        lagged_datadate = datadate + pd.Timedelta(days=fiscal_lag)
        if pd.notna(rdq):
            avail_date = max(rdq + pd.Timedelta(days=fiscal_lag), lagged_datadate)
        else:
            avail_date = lagged_datadate

        records.append({
            'ticker'    : ticker,
            'datadate'  : datadate,
            'avail_date': avail_date,
            'ceqq'      : frow['ceqq'],
            'niq'       : frow['niq'],
            'cshoq'     : frow['cshoq'],
        })

    if not records:
        print("  WARNING: No fundamental records after CCM mapping.")
        empty = pd.DataFrame(index=daily_index)
        return {'ceqq_panel': empty, 'niq_panel': empty, 'cshoq_panel': empty}

    records_df = pd.DataFrame(records)

    # ── Step 4: Build point-in-time panels via merge_asof ────────────────────
    # For each ticker, align fundamentals to every trading date using the
    # most recent observation whose avail_date <= the trading date.
    daily_df = pd.DataFrame({'date': daily_index}).sort_values('date')

    ceqq_cols  = {}
    niq_cols   = {}
    cshoq_cols = {}

    for ticker, grp in records_df.groupby('ticker'):
        grp_sorted = grp.sort_values('avail_date').reset_index(drop=True)

        merged = pd.merge_asof(
            daily_df,
            grp_sorted[['avail_date', 'ceqq', 'niq', 'cshoq']],
            left_on='date',
            right_on='avail_date',
            direction='backward',   # use only data available on or before each date
        )
        merged = merged.set_index('date')

        ceqq_cols[ticker]  = merged['ceqq']
        niq_cols[ticker]   = merged['niq']
        cshoq_cols[ticker] = merged['cshoq']

    ceqq_panel  = pd.DataFrame(ceqq_cols,  index=daily_index)
    niq_panel   = pd.DataFrame(niq_cols,   index=daily_index)
    cshoq_panel = pd.DataFrame(cshoq_cols, index=daily_index)

    print(f"  Fundamentals panel: {len(ceqq_panel)} days × "
          f"{ceqq_panel.notna().any().sum()} tickers with data")

    result = {
        'ceqq_panel' : ceqq_panel,
        'niq_panel'  : niq_panel,
        'cshoq_panel': cshoq_panel,
    }

    print(f"  Saving fundamentals to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return result
