import os

BASE       = '/Users/giuseppeiannone/15.C51_project1_tradingstrat'
OUTPUT_DIR = os.path.join(BASE, 'momentum_strategy', 'output_momentum')
FIG_DIR    = os.path.join(OUTPUT_DIR, 'figures')

CACHE_DIR  = os.path.join(BASE, 'momentum_strategy', 'data', 'cache')

os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Data window
DOWNLOAD_START = '2013-01-01'
ANALYSIS_START = '2015-01-01'
TRAIN_END      = '2021-12-31'
ANALYSIS_END   = '2025-04-11'

# Core strategy parameters
FORMATION_DAYS  = 252   # ~12 months lookback
FORMATION_SHORT = 126   # ~6 months lookback (dual-horizon signal)
SKIP_DAYS       = 21    # ~1 month skip (avoids short-term reversal)
HOLDING_DAYS    = 21    # rebalance frequency (~1 month)
N_LONG          = 3      # long positions (top momentum)
N_SHORT         = 0      # short positions (0 = long-only)
TC              = 0.001 # 10 bps one-way transaction cost

# Enhancements
VOL_WINDOW          = 63    # trailing days for realized-vol estimate (position sizing)
CONSISTENCY_WEIGHT  = 0.20  # blend weight for momentum consistency vs raw return

# ── Multi-factor strategy ─────────────────────────────────────────────────────
STRATEGY_NAME = 'momentum'       # 'momentum' | 'multi_factor'

OUTPUT_MF_DIR = os.path.join(BASE, 'momentum_strategy', 'output_mf')
FIG_MF_DIR    = os.path.join(OUTPUT_MF_DIR, 'figures')
os.makedirs(FIG_MF_DIR, exist_ok=True)

N_LONG_MF  = 6   # top 20% of ~30-stock universe
N_SHORT_MF = 6   # bottom 20%

FACTOR_WEIGHTS = {
    'momentum': 0.30,
    'value'   : 0.25,
    'quality' : 0.25,
    'low_vol' : 0.20,
}

FISCAL_LAG_DAYS = 60    # days after fiscal quarter-end before data is used (anti look-ahead)
LOW_VOL_DAYS    = 63    # trailing window for realized-vol factor
WINSOR_CLIP     = 0.05  # winsorize each factor at 5th/95th percentile
MAX_SECTOR_TILT = 2     # max stocks from one sector in either book

SECTOR_MAP = {
    'AAPL': 'Technology',  'MSFT': 'Technology',  'IBM': 'Technology',
    'CSCO': 'Technology',  'CRM':  'Technology',  'INTC': 'Technology',
    'NVDA': 'Technology',
    'GS':   'Financials',  'JPM':  'Financials',  'AXP': 'Financials',
    'V':    'Financials',  'TRV':  'Financials',
    'JNJ':  'Healthcare',  'MRK':  'Healthcare',  'AMGN': 'Healthcare',
    'UNH':  'Healthcare',  'PFE':  'Healthcare',
    'BA':   'Industrials', 'CAT':  'Industrials', 'MMM': 'Industrials',
    'HON':  'Industrials', 'RTX':  'Industrials', 'UTX': 'Industrials',
    'NKE':  'ConsDisc',    'MCD':  'ConsDisc',    'DIS': 'ConsDisc',
    'WMT':  'ConsDisc',    'WBA':  'ConsDisc',    'AMZN': 'ConsDisc',
    'HD':   'ConsDisc',
    'CVX':  'Energy',      'XOM':  'Energy',
    'VZ':   'Telecom',
    'DOW':  'Materials',   'DD':   'Materials',   'DWDP': 'Materials',
    'SHW':  'Materials',
    'PG':   'ConsStaples', 'KO':   'ConsStaples',
    'AA':   'Materials',   'BAC':  'Financials',  'HPQ': 'Technology',
    'T':    'Telecom',     'GE':   'Industrials',
}
