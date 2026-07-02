"""
implied_vol.py
---------------
Black-Scholes-Merton (BSM) pricing and implied volatility extraction.

This is a direct, real-data application of the coursework's BSM inversion
step: given an observed market price, what volatility would the BSM model
need to be fed in order to reproduce that price? That volatility is the
"implied volatility" — the market's own view of future volatility, baked
into the price it's willing to pay/sell at.

We use the same numerical method as the coursework: scipy's `brentq`
bracketed root-finder, applied to the function
    f(sigma) = BSM_price(sigma) - market_price
which has a single root because BSM price is monotonically increasing in
sigma (more volatility always makes an option worth more, never less).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]


def bsm_price(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
    """
    Computes the Black-Scholes-Merton price of a European call or put.

    Parameters
    ----------
    S : Current spot price of the underlying.
    K : Strike price of the option.
    T : Time to expiry, in years.
    r : Continuously-compounded risk-free rate (decimal, e.g. 0.02 for 2%).
    sigma : Volatility of the underlying (decimal, e.g. 0.20 for 20%).
    option_type : 'call' or 'put'.

    Returns
    -------
    The theoretical BSM price. Returns NaN for invalid inputs (T < 0 or
    sigma <= 0), and returns the intrinsic value for T == 0 (an option at
    the instant of expiry is worth exactly its payoff, with no time value
    left to price).

    Assumptions
    -----------
    Standard BSM assumptions: continuous trading, no dividends, no
    transaction costs, log-normal terminal price distribution, constant
    volatility over [0, T]. (The "constant volatility" assumption is
    exactly what implied volatility "fixes" locally — see vol_surface.py
    for how we relax it across strikes/maturities.)
    """
    option_type = option_type.lower()

    if T < 0 or sigma <= 0:
        return float("nan")

    if T == 0:
        if option_type == "call":
            return max(S - K, 0.0)
        elif option_type == "put":
            return max(K - S, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    # d1 measures, in standard-deviation units, how far in-the-money the
    # option is expected to land — it blends the log-moneyness log(S/K)
    # with the expected drift of the stock under the risk-neutral measure
    # over the life of the option (r + 0.5*sigma^2 is the risk-neutral
    # drift of log(S), the 0.5*sigma^2 term being the usual Ito correction).
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    # d2 = d1 minus one "unit" of volatility-over-time (sigma*sqrt(T)).
    # N(d2) is literally the risk-neutral probability that the option
    # expires in-the-money.
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        # S*N(d1) is the present value of "receiving the stock if ITM",
        # discounted via the stock's own risk-neutral drift (self-discounting).
        # K*exp(-rT)*N(d2) is the present value of "paying the strike if ITM",
        # weighted by the actual risk-neutral probability of exercise.
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        # Same logic mirrored for a put: receive the strike, give up the
        # stock, weighted by N(-d2) / N(-d1) — the probabilities of
        # finishing in-the-money for a put (i.e. S_T < K).
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        return float("nan")

    return float(price)


def compute_implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    low_vol: float = 1e-6,
    high_vol: float = 5.0,
) -> float:
    """
    Solves for the volatility that makes the BSM price equal the observed
    market price, using Brent's method (scipy's `brentq`).

    Why Brent's method: BSM price is a smooth, strictly increasing function
    of sigma, so the equation BSM(sigma) - market_price = 0 has exactly one
    root in any sensible volatility range. Brent's method is a robust
    bracketed root-finder that is guaranteed to converge as long as the
    root is "bracketed" (i.e. the function has opposite signs at the two
    ends of the search range) — which is exactly what we check before
    calling it.

    Parameters
    ----------
    market_price : The observed mid-price of the option in the market.
    S, K, T, r : Same meaning as in bsm_price.
    option_type : 'call' or 'put'.
    low_vol, high_vol : Search bounds for the implied volatility (default
        0.0001% to 500%, wide enough to cover virtually any real market
        quote without numerical issues at sigma = 0).

    Returns
    -------
    The implied volatility as a decimal, or NaN if:
      - any input is invalid (S, K, T <= 0, or a negative price), or
      - the market price falls outside the no-arbitrage bounds achievable
        within [low_vol, high_vol] (i.e. the root is not bracketed), or
      - the solver otherwise fails to converge.
    """
    if S <= 0 or K <= 0 or T <= 0 or market_price < 0:
        return float("nan")

    def objective(sigma: float) -> float:
        return bsm_price(S, K, T, r, sigma, option_type) - market_price

    try:
        # Brent's method requires the objective to have opposite signs at
        # the two bracket endpoints — otherwise there's no guaranteed root
        # in [low_vol, high_vol] (the market price may violate arbitrage
        # bounds, e.g. be below intrinsic value or above the spot price).
        if objective(low_vol) * objective(high_vol) > 0:
            return float("nan")

        return float(brentq(objective, low_vol, high_vol))
    except Exception:
        return float("nan")


def add_implied_vols(df: pd.DataFrame, S: float, r: float) -> pd.DataFrame:
    """
    Applies compute_implied_vol to every row of an option-chain DataFrame
    and appends the result as a new 'implied_vol' column.

    Parameters
    ----------
    df : DataFrame with at least the columns ['strike', 'mid_price', 'T',
        'option_type'] (the schema produced by data_ingestion.fetch_option_chain).
    S : Current spot price of the underlying, used for every row.
    r : Risk-free rate, used for every row.

    Returns
    -------
    A copy of `df` with an added 'implied_vol' column, with rows where
    no implied volatility could be computed removed. Also prints a short
    summary of how many options were processed vs. how many succeeded,
    so failures (e.g. from bad market quotes) are visible rather than silent.
    """
    out = df.copy()

    out["implied_vol"] = out.apply(
        lambda row: compute_implied_vol(
            market_price=row["mid_price"],
            S=S,
            K=row["strike"],
            T=row["T"],
            r=r,
            option_type=row["option_type"],
        ),
        axis=1,
    )

    n_total = len(out)
    out = out[out["implied_vol"].notna()].reset_index(drop=True)
    n_success = len(out)

    print(
        f"Implied volatility computation: {n_success}/{n_total} options succeeded "
        f"({n_total - n_success} dropped due to non-convergent or arbitrage-violating prices)."
    )

    return out
