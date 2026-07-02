"""
data_ingestion.py
------------------
Handles all communication with Yahoo Finance via the `yfinance` library.

This module is the "front door" of the whole pipeline: every downstream
calculation (implied vol, SVI calibration, Monte Carlo, risk) depends on
getting a clean, well-shaped option chain out of this module. Yahoo's raw
option chain data is noisy — stale quotes, zero-volume contracts, and
crossed/zero bid-ask spreads are common — so a meaningful chunk of this
module is dedicated to cleaning rather than just fetching.

In the original coursework, the option chain was a static spreadsheet
(`MANG3117_common_dataset.xlsx`) with clean, pre-generated prices for a
fictional stock. Here we replace that spreadsheet with a live feed, which
means we now have to handle the data-quality problems a real feed brings.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


class DataIngestionError(Exception):
    """Raised when we cannot retrieve usable market data for a ticker."""


def get_spot_price(ticker: str) -> float:
    """
    Fetches the current (or most recent close) spot price for a ticker.

    Parameters
    ----------
    ticker : The stock ticker symbol, e.g. "AAPL".

    Returns
    -------
    The latest spot price as a float.

    Raises
    ------
    DataIngestionError if no price data can be found at all (e.g. the
    ticker does not exist or Yahoo Finance returned nothing).
    """
    try:
        tk = yf.Ticker(ticker)

        # fast_info is much quicker than the full `.info` dict and is the
        # recommended way to grab a live price in modern yfinance versions.
        price = tk.fast_info.get("lastPrice")
        if price and price > 0:
            return float(price)

        # Fallback: if fast_info doesn't have a price (can happen for some
        # tickers), fall back to the most recent daily close.
        hist = tk.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])

        raise DataIngestionError(f"No spot price could be found for '{ticker}'.")

    except DataIngestionError:
        raise
    except Exception as exc:  # yfinance can raise many different errors
        raise DataIngestionError(
            f"Failed to fetch spot price for '{ticker}': {exc}"
        ) from exc


def get_risk_free_rate() -> float:
    """
    Estimates the risk-free rate using the 3-month US Treasury yield.

    We use the "^IRX" ticker on Yahoo Finance, which quotes the 13-week
    T-bill discount rate in *percentage points* (e.g. a value of 5.10 means
    5.10%). Dividing by 100 converts it into a decimal annualised rate
    (e.g. 0.051), which is the convention used everywhere else in this
    codebase (consistent with r = 0.02 in the original coursework).

    Returns
    -------
    The risk-free rate as a decimal (e.g. 0.05 for 5%). Falls back to a
    conservative default of 2% if the data feed is unavailable, since a
    missing risk-free rate should not be allowed to crash the whole app.
    """
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            latest_yield_pct = float(hist["Close"].iloc[-1])
            return latest_yield_pct / 100.0
    except Exception:
        # We deliberately swallow the error here: the risk-free rate has a
        # sane fallback, so a Treasury-yield outage shouldn't break the app.
        pass

    # Fallback consistent with the original coursework's r = 0.02 assumption.
    return 0.02


def _time_to_expiry_years(expiry_str: str, valuation_date: Optional[dt.date] = None) -> float:
    """
    Converts an expiry date string (e.g. "2025-06-20") into time-to-expiry
    in years, measured from today (or a supplied valuation date).

    We use a 365.25-day year (calendar days, not trading days) because
    option expiry conventions in the market are calendar-based, not
    trading-day based — this matches how the time value in market option
    prices is actually built.
    """
    if valuation_date is None:
        valuation_date = dt.date.today()

    expiry_date = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
    days = (expiry_date - valuation_date).days

    # Treat options expiring "today" as having a tiny sliver of time left
    # rather than exactly zero, so downstream BSM maths doesn't divide by 0.
    days = max(days, 1)

    return days / 365.25


def fetch_option_chain(ticker: str, max_expiries: Optional[int] = None) -> pd.DataFrame:
    """
    Pulls and cleans the full live option chain for `ticker` across all
    (or the first `max_expiries`) available expiry dates.

    What "cleaning" means here, and why each step matters:
      1. Drop contracts with zero volume AND zero open interest — these are
         "dead" contracts nobody is trading, and their quotes are often
         stale or simply wrong (e.g. a bid/ask left over from weeks ago).
      2. Drop contracts where both bid and ask are zero — there is no
         tradeable market price here at all, so no implied vol can be
         meaningfully backed out.
      3. Compute the mid-price as (bid + ask) / 2 — the standard market
         convention for "the" price of an option, since the last traded
         price can be stale while bid/ask reflect where the market is now.

    Parameters
    ----------
    ticker : The stock ticker symbol, e.g. "AAPL".
    max_expiries : Optional cap on the number of expiry dates to pull, to
        keep the request fast for very liquid names with many expiries.

    Returns
    -------
    A DataFrame with columns:
        ['strike', 'mid_price', 'expiry', 'T', 'option_type',
         'bid', 'ask', 'volume', 'open_interest']

    Raises
    ------
    DataIngestionError if the ticker has no option chain, or if every
    contract gets filtered out by the cleaning step.
    """
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
    except Exception as exc:
        raise DataIngestionError(
            f"Failed to fetch option expiries for '{ticker}': {exc}"
        ) from exc

    if not expiries:
        raise DataIngestionError(
            f"'{ticker}' has no listed options on Yahoo Finance."
        )

    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    valuation_date = dt.date.today()
    all_rows = []

    for expiry in expiries:
        try:
            chain = tk.option_chain(expiry)
        except Exception:
            # Some individual expiries occasionally fail to load even when
            # the ticker overall is fine — skip that expiry rather than
            # aborting the whole fetch.
            continue

        T = _time_to_expiry_years(expiry, valuation_date)

        calls = chain.calls.copy()
        calls["option_type"] = "call"

        puts = chain.puts.copy()
        puts["option_type"] = "put"

        for leg in (calls, puts):
            leg["expiry"] = expiry
            leg["T"] = T

        all_rows.append(calls)
        all_rows.append(puts)

    if not all_rows:
        raise DataIngestionError(
            f"Could not retrieve any option chain data for '{ticker}'."
        )

    raw = pd.concat(all_rows, ignore_index=True)

    # yfinance column names: strike, bid, ask, volume, openInterest, lastPrice
    raw = raw.rename(columns={"openInterest": "open_interest"})

    # Some fields can come back as NaN for illiquid contracts; treat NaN as 0
    # so the filtering logic below behaves predictably.
    for col in ("bid", "ask", "volume", "open_interest"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)

    # --- Cleaning step 1: remove contracts nobody is actually trading ---
    # A contract with zero volume AND zero open interest has no current
    # market participation, so its quote is unreliable for calibration.
    raw = raw[(raw["volume"] > 0) | (raw["open_interest"] > 0)]

    # --- Cleaning step 2: remove contracts with no tradeable market price ---
    raw = raw[~((raw["bid"] == 0) & (raw["ask"] == 0))]

    if raw.empty:
        raise DataIngestionError(
            f"No usable option contracts remained for '{ticker}' after cleaning "
            "(all contracts had zero volume/open interest or zero bid/ask)."
        )

    # --- Cleaning step 3: compute the market mid-price ---
    raw["mid_price"] = (raw["bid"] + raw["ask"]) / 2.0

    # Drop options that, despite passing the above filters, still have a
    # non-positive mid-price — these cannot be used for BSM inversion since
    # a non-positive price has no valid implied volatility.
    raw = raw[raw["mid_price"] > 0]

    if raw.empty:
        raise DataIngestionError(
            f"No option contracts for '{ticker}' had a positive mid-price after cleaning."
        )

    result = raw[
        ["strike", "mid_price", "expiry", "T", "option_type", "bid", "ask", "volume", "open_interest"]
    ].reset_index(drop=True)

    return result
