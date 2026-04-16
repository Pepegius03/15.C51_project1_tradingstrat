"""
Data preparation pipeline:
  1. Load FF factors; compute excess returns
  2. 252-day rolling beta via OLS
  3. CAPM residuals
  4. DJ30 membership mask
  5. Cross-sectional z-scoring of residuals
  6. Feature engineering (8 features)
  7. Next-day reversion target
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dj30_changes import build_membership_matrix
from data_download import EXCLUDED_TICKERS

DATA_DIR = Path(__file__).parent / "data"
FF_PATH  = Path(__file__).parent / "F-F_Research_Data_Factors_daily.csv"

TRAIN_END = "2019-12-31"
VAL_END   = "2022-12-31"
TEST_END  = "2025-12-31"


def load_ff_factors() -> pd.DataFrame:
    ff = pd.read_csv(FF_PATH, skiprows=0, index_col=0)
    ff.index = pd.to_datetime(ff.index.astype(str), format="%Y%m%d")
    ff.columns = [c.strip() for c in ff.columns]
    ff = ff / 100.0   # FF reports in percent
    return ff[["Mkt-RF", "RF"]]


def rolling_beta(excess_ret: pd.Series, mkt_rf: pd.Series, window: int = 252) -> pd.Series:
    """OLS beta of excess_ret ~ mkt_rf using rolling window."""
    cov = excess_ret.rolling(window).cov(mkt_rf)
    var = mkt_rf.rolling(window).var()
    return cov / var


def build_features(returns: pd.DataFrame, ff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    features  : DataFrame, shape (T*N, 8), MultiIndex (date, ticker)
    targets   : Series, next-day excess return sign (0/1), same index
    membership: boolean DataFrame (T, N)
    """
    DATA_DIR.mkdir(exist_ok=True)

    feat_cache   = DATA_DIR / "features.pkl"
    target_cache = DATA_DIR / "targets.pkl"
    zscore_cache = DATA_DIR / "zscores.pkl"

    if feat_cache.exists() and target_cache.exists() and zscore_cache.exists():
        print("Loading cached features/targets...")
        features  = pd.read_pickle(feat_cache)
        targets   = pd.read_pickle(target_cache)
        zscores   = pd.read_pickle(zscore_cache)
        return features, targets, zscores

    # Align dates
    common_dates = returns.index.intersection(ff.index)
    returns = returns.loc[common_dates]
    ff      = ff.loc[common_dates]

    rf     = ff["RF"]
    mkt_rf = ff["Mkt-RF"]

    # Excess returns: stock return − RF (broadcast RF across columns)
    excess = returns.subtract(rf, axis=0)

    # Rolling beta and CAPM residuals
    print("Computing rolling betas (this takes a while)...")
    betas   = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    resid   = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for ticker in returns.columns:
        b = rolling_beta(excess[ticker], mkt_rf)
        betas[ticker] = b
        resid[ticker] = excess[ticker] - b * mkt_rf

    # DJ30 membership mask
    print("Building membership matrix...")
    membership = build_membership_matrix(
        start=str(returns.index[0].date()),
        end=str(returns.index[-1].date()),
    )
    membership = membership.reindex(index=returns.index, columns=returns.columns, fill_value=False)
    membership = membership.drop(columns=[c for c in EXCLUDED_TICKERS if c in membership.columns])

    # Mask residuals to DJ30 members only
    resid_masked = resid.where(membership)

    # Cross-sectional z-score of CAPM residuals
    z_mean = resid_masked.mean(axis=1)
    z_std  = resid_masked.std(axis=1)
    zscores = resid_masked.subtract(z_mean, axis=0).divide(z_std.replace(0, np.nan), axis=0)
    zscores = zscores.where(membership)

    zscores.to_pickle(zscore_cache)

    # ── Feature engineering ──────────────────────────────────────
    print("Engineering features...")
    z1      = zscores.shift(1)
    z5      = zscores.shift(5)
    z_m5    = zscores.rolling(5).mean()
    z_s5    = zscores.rolling(5).std()
    cr5     = resid_masked.rolling(5).sum()
    beta_df = betas.where(membership)
    rvol20  = excess.rolling(20).std().where(membership)

    tickers = returns.columns.tolist()
    dates   = returns.index

    rows = []
    y_rows = []

    # Next-day sign of excess return (forward-fill safe: shift(-1))
    next_excess = excess.shift(-1)

    for date in dates:
        members = membership.loc[date]
        active  = members[members].index.tolist()
        if not active:
            continue
        for tk in active:
            feat = {
                "z_t":           zscores.at[date, tk],
                "z_t1":          z1.at[date, tk],
                "z_t5":          z5.at[date, tk],
                "z_roll5_mean":  z_m5.at[date, tk],
                "z_roll5_std":   z_s5.at[date, tk],
                "cum_resid_5d":  cr5.at[date, tk],
                "beta":          beta_df.at[date, tk],
                "realized_vol_20d": rvol20.at[date, tk],
            }
            rows.append((date, tk, feat))
            y_val = next_excess.at[date, tk]
            y_rows.append((date, tk, 1 if y_val > 0 else 0))

    idx = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["date", "ticker"])
    features = pd.DataFrame([r[2] for r in rows], index=idx)
    targets  = pd.Series([r[2] for r in y_rows], index=idx, name="target", dtype=int)

    # Drop rows with NaN in any feature or target
    valid = features.notna().all(axis=1) & targets.notna()
    features = features[valid]
    targets  = targets[valid]

    features.to_pickle(feat_cache)
    targets.to_pickle(target_cache)
    print(f"Features: {features.shape}, Targets: {targets.shape}")
    return features, targets, zscores


if __name__ == "__main__":
    from data_download import download_prices, compute_returns
    prices  = download_prices()
    returns = compute_returns(prices)
    ff      = load_ff_factors()
    features, targets, zscores = build_features(returns, ff)
    print(features.head())
    print(targets.value_counts())
