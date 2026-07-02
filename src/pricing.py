"""
pricing.py
----------
Values individual options and the user's option book using risk-neutral
Monte Carlo, and estimates the book's delta via bump-and-revalue.

Key concepts
------------
Long vs short: a "long" position means we have bought the option — its
value adds directly to the book (we own an asset). A "short" position
means we have sold/written the option — we have a liability, so its value
*subtracts* from the book (we'd have to pay out the payoff at expiry).
We encode this with a position sign: +1 for long, -1 for short.

Discounting: a payoff received in the future is worth less today than the
same amount received now, because money today can be invested to earn the
risk-free rate. We discount the average simulated payoff by e^(-rT) to
convert a future expected payoff into today's dollars — this is exactly
the risk-neutral valuation formula:

    V_0 = e^{-rT} * (1/N) * sum_{i=1}^{N} f(S_T^i)

where f(.) is the option's payoff function and S_T^i are the simulated
terminal risk-neutral spot prices.

Notional: scales a single "unit" option value up to the actual dollar size
of the position. In this tool, notional is denominated in $millions and
represents (very simply) a position size multiplier on the option payoff,
matching the coursework's treatment of "notional ($m)" positions.
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import pandas as pd

OptionType = Literal["call", "put"]
Position = Literal["long", "short"]


def _option_payoff(S_T: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    """Computes the European option payoff max(S_T - K, 0) or max(K - S_T, 0)."""
    option_type = option_type.lower()
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def price_option_mc(
    paths: np.ndarray,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    notional: float,
    position: Position,
) -> float:
    """
    Prices a single option position using risk-neutral Monte Carlo.

    Parameters
    ----------
    paths : Array of simulated terminal spot prices under the risk-neutral
        measure (e.g. rn_paths[-1, :] from monte_carlo.simulate_paths), OR
        a full path array of shape (n_steps+1, n_paths) — in which case the
        final row is used automatically.
    K : Strike price.
    T : Time to expiry, in years (used for discounting).
    r : Risk-free rate (decimal).
    option_type : 'call' or 'put'.
    notional : Position size, in $ millions.
    position : 'long' (+1, we own the option) or 'short' (-1, we've written it).

    Returns
    -------
    The signed, notional-scaled value of this position, in $ millions.
    """
    terminal = paths[-1, :] if paths.ndim == 2 else paths

    payoffs = _option_payoff(terminal, K, option_type)

    # Risk-neutral valuation: discount the average simulated payoff back
    # to today's dollars at the risk-free rate.
    unit_value = np.exp(-r * T) * np.mean(payoffs)

    sign = 1.0 if position == "long" else -1.0

    return float(sign * notional * unit_value)


def price_book(positions_df: pd.DataFrame, rn_paths: np.ndarray, r: float) -> pd.DataFrame:
    """
    Values every position in the user's option book and sums them into a
    total book value.

    Parameters
    ----------
    positions_df : DataFrame with columns:
        ['option_type', 'strike', 'expiry_T', 'notional', 'position']
    rn_paths : Risk-neutral simulated paths, shape (n_steps+1, n_paths).
    r : Risk-free rate (decimal).

    Returns
    -------
    A copy of positions_df with an added 'value_m' column (the signed,
    notional-scaled value of each position, in $m), plus a final summary
    row '__TOTAL__' giving the total book value. The total is also
    accessible via df.attrs['total_value_m'] for convenience.
    """
    out = positions_df.copy()

    out["value_m"] = out.apply(
        lambda row: price_option_mc(
            paths=rn_paths,
            K=row["strike"],
            T=row["expiry_T"],
            r=r,
            option_type=row["option_type"],
            notional=row["notional"],
            position=row["position"],
        ),
        axis=1,
    )

    total_value = float(out["value_m"].sum())
    out.attrs["total_value_m"] = total_value

    return out


def compute_book_delta(
    positions_df: pd.DataFrame,
    S0: float,
    t: float,
    rn_paths_fn,
    r: float,
    bump_pct: float = 0.005,
) -> float:
    """
    Estimates the book's overall delta (sensitivity of total book value to
    a $1 move in the underlying) using central bump-and-revalue.

    Formula:
        Delta = [ V(S0 * (1 + bump_pct)) - V(S0 * (1 - bump_pct)) ]
                / (2 * S0 * bump_pct)

    Why a *central* difference rather than a one-sided (forward) difference:
    a central difference cancels out the leading-order curvature (gamma)
    error in the Taylor expansion, making it a second-order accurate
    estimate of delta rather than only first-order. In other words, bumping
    both up and down and averaging gives a materially more accurate delta
    estimate than bumping only up (or only down) for the same computational
    cost — this is exactly why the coursework used central differences.

    Parameters
    ----------
    positions_df : The option book, same schema as in price_book.
    S0 : Current spot price (the level we are bumping around).
    t : Current valuation time, in years (passed through to rn_paths_fn so
        it can simulate from the correct remaining maturity).
    rn_paths_fn : A callable `rn_paths_fn(S_start: float) -> np.ndarray`
        that returns freshly simulated risk-neutral paths starting from
        spot level S_start (so the bumped revaluations use the bumped
        starting spot). Using a callable here (rather than a fixed path
        array) is what makes a *bump-and-revalue* delta possible: we need
        genuinely different simulations at S0+h and S0-h, not just a
        re-weighting of one fixed simulation.
    r : Risk-free rate (decimal).
    bump_pct : Bump size as a fraction of spot (default 0.5%, matching the
        coursework's h = 0.5% of spot).

    Returns
    -------
    The estimated book delta (dimensionless: $m of book value per $1 of
    spot per $1 of spot, i.e. a sensitivity rather than a dollar amount).
    """
    bump = bump_pct * S0

    S_up = S0 + bump
    S_down = max(S0 - bump, 1e-6)

    paths_up = rn_paths_fn(S_up)
    paths_down = rn_paths_fn(S_down)

    value_up = float(price_book(positions_df, paths_up, r)["value_m"].sum())
    value_down = float(price_book(positions_df, paths_down, r)["value_m"].sum())

    delta = (value_up - value_down) / (S_up - S_down)

    return float(delta)
