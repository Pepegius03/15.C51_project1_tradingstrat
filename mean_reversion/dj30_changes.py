"""
DJ30 Historical Component Changes (2004–2024)
==============================================
All reconstitution events for the Dow Jones Industrial Average
from April 2004 onward, covering 10+ years of backtesting.

Sources: Wikipedia "Historical components of the DJIA", S&P Dow Jones Indices

Usage:
    from dj30_changes import changes_df, get_members_at, stable_members, all_ever_members
"""

import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 1. RAW CHANGE EVENTS
# ─────────────────────────────────────────────────────────────
changes = [
    # (effective_date, ticker_added, company_added, ticker_removed, company_removed, reason)

    # 2004-04-08: 3 changes
    ("2004-04-08", "PFE",  "Pfizer",                  "EK",   "Eastman Kodak",           "Modernize index"),
    ("2004-04-08", "VZ",   "Verizon Communications",   "IP",   "International Paper",     "Modernize index"),
    ("2004-04-08", "AIG",  "American Intl Group",      "T",    "AT&T Corporation",        "Modernize index"),

    # 2008-02-19: 2 changes
    ("2008-02-19", "BAC",  "Bank of America",          "MO",   "Altria Group",            "Diversify sectors"),
    ("2008-02-19", "CVX",  "Chevron",                  "HON",  "Honeywell International", "Diversify sectors"),

    # 2008-09-22: 1 change
    ("2008-09-22", "KFT",  "Kraft Foods",              "AIG",  "American Intl Group",     "AIG bailout / financial crisis"),

    # 2009-06-08: 2 changes
    ("2009-06-08", "CSCO", "Cisco Systems",            "C",    "Citigroup",               "Citigroup crisis / GM bankruptcy"),
    ("2009-06-08", "TRV",  "Travelers Companies",      "GM",   "General Motors",          "GM bankruptcy"),

    # 2012-09-24: 1 change
    ("2012-09-24", "UNH",  "UnitedHealth Group",       "KFT",  "Kraft Foods",             "Kraft split into Mondelez"),

    # 2013-09-23: 3 changes
    ("2013-09-23", "GS",   "Goldman Sachs",            "BAC",  "Bank of America",         "Low stock price / diversify"),
    ("2013-09-23", "NKE",  "Nike",                     "AA",   "Alcoa",                   "Low stock price / diversify"),
    ("2013-09-23", "V",    "Visa",                     "HPQ",  "Hewlett-Packard",         "Low stock price / diversify"),

    # 2015-03-19: 1 change
    ("2015-03-19", "AAPL", "Apple",                    "T",    "AT&T Inc",                "AT&T low price after split"),

    # 2017-09-01: name change only — DuPont → DowDuPont (no ticker swap needed for most data)

    # 2018-06-26: WBA excluded (not available on yfinance); GE removal still tracked
    ("2018-06-26", None,   None,                       "GE",   "General Electric",        "GE long decline"),

    # 2019-04-02: 1 change (DowDuPont spinoff → Dow Inc replaces DowDuPont)
    ("2019-04-02", "DOW",  "Dow Inc",                  "DWDP", "DowDuPont",               "DowDuPont spinoff"),

    # 2020-04-06: name change only — United Technologies → Raytheon Technologies (UTX → RTX)

    # 2020-08-31: 3 changes
    ("2020-08-31", "CRM",  "Salesforce",               "XOM",  "Exxon Mobil",             "Modernize / WMT split prompted"),
    ("2020-08-31", "AMGN", "Amgen",                    "PFE",  "Pfizer",                  "Modernize / WMT split prompted"),
    ("2020-08-31", "HON",  "Honeywell International",  "RTX",  "Raytheon Technologies",   "Modernize / WMT split prompted"),

    # 2024-02-26: 1 change — WBA excluded; AMZN addition still tracked
    ("2024-02-26", "AMZN", "Amazon",                   None,   None,                      "WMT split prompted rebalance"),

    # 2024-11-08: 2 changes
    ("2024-11-08", "NVDA", "Nvidia",                   "INTC", "Intel",                   "Upgrade tech representation"),
    ("2024-11-08", "SHW",  "Sherwin-Williams",         "DOW",  "Dow Inc",                 "Upgrade industrials representation"),
]

changes_df = pd.DataFrame(changes, columns=[
    "effective_date", "ticker_added", "company_added",
    "ticker_removed", "company_removed", "reason"
])
changes_df["effective_date"] = pd.to_datetime(changes_df["effective_date"])


# ─────────────────────────────────────────────────────────────
# 2. DJ30 MEMBERSHIP AS OF A REFERENCE DATE
# ─────────────────────────────────────────────────────────────

# DJ30 as of January 1, 2004 (before any changes in our table)
DJ30_INITIAL_2004 = [
    "MMM",   # 3M
    "AA",    # Alcoa
    "MO",    # Altria (Philip Morris)
    "AXP",   # American Express
    "T",     # AT&T Corporation
    "BA",    # Boeing
    "CAT",   # Caterpillar
    "C",     # Citigroup
    "KO",    # Coca-Cola
    "DD",    # DuPont
    "XOM",   # Exxon Mobil
    "GE",    # General Electric
    "HPQ",   # Hewlett-Packard
    "HD",    # Home Depot
    "HON",   # Honeywell International
    "INTC",  # Intel
    "IBM",   # IBM
    "JNJ",   # Johnson & Johnson
    "JPM",   # JPMorgan Chase
    "MCD",   # McDonald's
    "MRK",   # Merck
    "MSFT",  # Microsoft
    "PFE",   # Pfizer
    "PG",    # Procter & Gamble
    "SBC",   # SBC Communications (became AT&T Inc)
    # UTX excluded — not available on yfinance
    "VZ",    # Verizon
    "WMT",   # Walmart
    "DIS",   # Walt Disney
    "IP",    # International Paper
]

# DJ30 as of January 1, 2014 (common starting point for 10-year backtest)
DJ30_AS_OF_2014 = [
    "MMM",   # 3M
    "AXP",   # American Express
    "T",     # AT&T Inc (was SBC, renamed)
    "BA",    # Boeing
    "CAT",   # Caterpillar
    "CVX",   # Chevron
    "CSCO",  # Cisco
    "KO",    # Coca-Cola
    "DD",    # DuPont
    "XOM",   # Exxon Mobil
    "GE",    # General Electric
    "GS",    # Goldman Sachs
    "HD",    # Home Depot
    "IBM",   # IBM
    "INTC",  # Intel
    "JNJ",   # Johnson & Johnson
    "JPM",   # JPMorgan Chase
    "MCD",   # McDonald's
    "MRK",   # Merck
    "MSFT",  # Microsoft
    "NKE",   # Nike
    "PFE",   # Pfizer
    "PG",    # Procter & Gamble
    "TRV",   # Travelers
    "UNH",   # UnitedHealth
    # UTX excluded — not available on yfinance
    "V",     # Visa
    "VZ",    # Verizon
    "WMT",   # Walmart
    "DIS",   # Walt Disney
]

# DJ30 as of today (April 2026 — latest composition)
DJ30_CURRENT = [
    "MMM",   # 3M
    "AMZN",  # Amazon
    "AXP",   # American Express
    "AMGN",  # Amgen
    "AAPL",  # Apple
    "BA",    # Boeing
    "CAT",   # Caterpillar
    "CVX",   # Chevron
    "CSCO",  # Cisco
    "KO",    # Coca-Cola
    "DIS",   # Walt Disney
    "GS",    # Goldman Sachs
    "HD",    # Home Depot
    "HON",   # Honeywell
    "IBM",   # IBM
    "JNJ",   # Johnson & Johnson
    "JPM",   # JPMorgan Chase
    "MCD",   # McDonald's
    "MRK",   # Merck
    "MSFT",  # Microsoft
    "NKE",   # Nike
    "NVDA",  # Nvidia
    "PG",    # Procter & Gamble
    "CRM",   # Salesforce
    "SHW",   # Sherwin-Williams
    "TRV",   # Travelers
    "UNH",   # UnitedHealth
    "V",     # Visa
    "VZ",    # Verizon
    "WMT",   # Walmart
]


# ─────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_members_at(date, initial_members=None, changes_table=None):
    """
    Return the set of DJ30 tickers as of a given date.
    
    Parameters
    ----------
    date : str or datetime
        The date to query membership for.
    initial_members : list, optional
        Starting membership list. Defaults to DJ30_AS_OF_2014.
    changes_table : DataFrame, optional
        Table of changes. Defaults to changes_df.
    
    Returns
    -------
    set of str
        Tickers that were DJ30 members on that date.
    """
    if initial_members is None:
        initial_members = DJ30_AS_OF_2014
    if changes_table is None:
        changes_table = changes_df
    
    date = pd.Timestamp(date)
    members = set(initial_members)
    
    # Only apply changes from 2014 onward (matching initial_members)
    relevant = changes_table[
        (changes_table["effective_date"] > "2013-12-31") &
        (changes_table["effective_date"] <= date)
    ]
    
    for _, row in relevant.iterrows():
        if row["ticker_removed"]:
            members.discard(row["ticker_removed"])
        if row["ticker_added"]:
            members.add(row["ticker_added"])
    
    return members


def build_membership_matrix(start="2014-01-01", end="2024-12-31"):
    """
    Build a (T × N) boolean DataFrame: True if stock was a DJ30 member on that day.
    
    Returns
    -------
    pd.DataFrame
        Index = business dates, Columns = all tickers ever in DJ30 during period.
    """
    dates = pd.bdate_range(start, end)
    all_tickers = set(DJ30_AS_OF_2014)
    
    relevant_changes = changes_df[
        (changes_df["effective_date"] > "2013-12-31") &
        (changes_df["effective_date"] <= end)
    ]
    for _, row in relevant_changes.iterrows():
        if row["ticker_added"]:
            all_tickers.add(row["ticker_added"])
    
    all_tickers = sorted(all_tickers)
    membership = pd.DataFrame(False, index=dates, columns=all_tickers)
    
    current = set(DJ30_AS_OF_2014)
    change_idx = 0
    sorted_changes = relevant_changes.sort_values("effective_date").reset_index(drop=True)
    
    for date in dates:
        while (change_idx < len(sorted_changes) and
               sorted_changes.loc[change_idx, "effective_date"] <= date):
            if sorted_changes.loc[change_idx, "ticker_removed"]:
                current.discard(sorted_changes.loc[change_idx, "ticker_removed"])
            if sorted_changes.loc[change_idx, "ticker_added"]:
                current.add(sorted_changes.loc[change_idx, "ticker_added"])
            change_idx += 1
        
        for ticker in current:
            if ticker in membership.columns:
                membership.loc[date, ticker] = True
    
    return membership


# ─────────────────────────────────────────────────────────────
# 4. CONVENIENCE LISTS
# ─────────────────────────────────────────────────────────────

# Stocks in DJ30 continuously from 2014-01-01 through 2024-12-31
stable_members = sorted(
    get_members_at("2014-01-01") & get_members_at("2024-12-31")
)

# All tickers that were ever in DJ30 during 2014–2024
all_ever_members = sorted(
    get_members_at("2014-01-01") | {t for t in changes_df[changes_df["effective_date"] > "2013-12-31"]["ticker_added"] if t is not None}
)


# ─────────────────────────────────────────────────────────────
# 5. PRINT SUMMARY IF RUN DIRECTLY
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("DJ30 COMPONENT CHANGES (2004 – 2024)")
    print("=" * 70)
    print()
    print(changes_df.to_string(index=False))
    print()
    print(f"Total change events: {len(changes_df)}")
    print()
    
    print("=" * 70)
    print("DJ30 AS OF 2014-01-01")
    print("=" * 70)
    print(sorted(DJ30_AS_OF_2014))
    print()

    print("=" * 70)
    print("CHANGES DURING 2014–2024 (relevant for 10-year backtest)")
    print("=" * 70)
    recent = changes_df[changes_df["effective_date"] >= "2014-01-01"]
    print(recent.to_string(index=False))
    print()
    
    print("=" * 70)
    print(f"STABLE MEMBERS 2014–2024 ({len(stable_members)} stocks)")
    print("=" * 70)
    print(stable_members)
    print()
    
    print("=" * 70)
    print(f"ALL EVER MEMBERS 2014–2024 ({len(all_ever_members)} stocks)")
    print("=" * 70)
    print(all_ever_members)
