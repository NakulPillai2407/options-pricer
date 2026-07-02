"""
greeks.py
---------
Computes the option book's risk sensitivities ("Greeks") - Delta, Gamma,
Vega, Theta, and Rho - via bump-and-revalue under the calibrated local
volatility surface, and tracks how they evolve as time passes (spot held
fixed) up to a chosen horizon.

Every Greek here answers the same underlying question - "if I nudge one
input up and down slightly and reprice the book both times, how much does
the book's value move?" - just nudging a different input each time:

    Delta : nudge the spot price
    Gamma : how much Delta itself changes when spot is nudged
            (the curvature of value vs. spot)
    Vega  : nudge the volatility level
    Theta : nudge time forward by one day, spot held fixed
            (how much value is lost purely from time passing)
    Rho   : nudge the risk-free rate

All bumps are central (up AND down, then averaged), which cancels the
leading-order error and gives a materially more accurate estimate than a
one-sided bump for the same computational cost (see pricing.py's
compute_book_delta for the same rationale, applied there to Delta only).

To keep each pair of bumped valuations comparable (isolating the effect of
the bump itself, rather than fresh Monte Carlo noise), every bumped
revaluation of a given leg reuses the SAME random seed - a common random
numbers scheme - so the underlying shock sequence is identical across the
base/up/down runs for that leg.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .monte_carlo import simulate_paths
from .pricing import _option_payoff

Params = Dict[str, float]


def _leg_value(
    S_eval: float,
    valuation_time: float,
    row: pd.Series,
    r: float,
    params: Params,
    vol_shift: float = 0.0,
    seed: int | None = None,
    n_inner_paths: int = 300,
    n_inner_steps_per_year: int = 52,
) -> float:
    """
    Values a single option leg at one (spot, time) point, optionally under a
    shifted volatility level (for Vega). Pass a bumped `r` directly for Rho -
    it is used consistently as both the simulation drift and the discount
    rate, matching how the rest of the app prices options risk-neutrally.
    """
    tau = row["expiry_T"] - valuation_time
    sign = 1.0 if row["position"] == "long" else -1.0

    if tau <= 0:
        # An expired leg's value is just its fixed intrinsic payoff - no
        # volatility, rate, or further time exposure left to bump.
        payoff = _option_payoff(np.array([S_eval]), row["strike"], row["option_type"])[0]
        return sign * row["notional"] * payoff

    n_steps = max(int(np.ceil(tau * n_inner_steps_per_year)), 1)

    _, _, rn_paths = simulate_paths(
        S0=S_eval,
        mu=r,
        r=r,
        T=tau,
        n_steps=n_steps,
        n_paths=n_inner_paths,
        S0_ref=params["S0"],
        params=params,
        vol_shift=vol_shift,
        seed=seed,
    )
    terminal = rn_paths[-1, :]
    payoff = float(np.mean(_option_payoff(terminal, row["strike"], row["option_type"])))
    return sign * row["notional"] * np.exp(-r * tau) * payoff


def _book_value(
    S_eval: float,
    valuation_time: float,
    positions_df: pd.DataFrame,
    r: float,
    params: Params,
    vol_shift: float = 0.0,
    n_inner_paths: int = 300,
    n_inner_steps_per_year: int = 52,
) -> float:
    """
    Sums `_leg_value` across every position in the book. Each leg gets its
    own fixed seed (derived from its row position), reused across every
    bumped variant of that leg within a single `compute_book_greeks` call -
    this is what gives the common-random-numbers variance reduction.
    """
    total = 0.0
    for leg_idx, (_, row) in enumerate(positions_df.iterrows()):
        seed = 1_000 * (leg_idx + 1)
        total += _leg_value(
            S_eval,
            valuation_time,
            row,
            r,
            params,
            vol_shift=vol_shift,
            seed=seed,
            n_inner_paths=n_inner_paths,
            n_inner_steps_per_year=n_inner_steps_per_year,
        )
    return total


def compute_book_greeks(
    S: float,
    valuation_time: float,
    positions_df: pd.DataFrame,
    r: float,
    params: Params,
    bump_pct: float = 0.005,
    vol_bump: float = 0.01,
    r_bump: float = 0.0025,
    theta_dt: float = 1.0 / 365.0,
    n_inner_paths: int = 300,
    n_inner_steps_per_year: int = 52,
) -> Dict[str, float]:
    """
    Computes the full set of book-level Greeks at one (spot, time) point.

    Parameters
    ----------
    S : Spot price to evaluate at.
    valuation_time : Time from today, in years, at which to value the book
        (0.0 = today).
    positions_df : The option book (schema matching pricing.price_book).
    r : Risk-free rate (decimal).
    params : Calibrated local volatility parameters.
    bump_pct : Spot bump size as a fraction of S, for Delta/Gamma.
    vol_bump : Volatility bump size in decimal (0.01 = 1 percentage point),
        for Vega.
    r_bump : Rate bump size in decimal (0.0025 = 25bp), for Rho.
    theta_dt : Time step for Theta, in years (default: one calendar day).
    n_inner_paths, n_inner_steps_per_year : Inner Monte Carlo resolution.

    Returns
    -------
    Dictionary with:
        value : book value at (S, valuation_time), in $m
        delta : $m change in book value per $1 move in spot
        gamma : change in delta per $1 move in spot
        vega  : $m change in book value per 1 percentage point
                (e.g. 20% -> 21%) move in volatility
        theta : $m change in book value from one calendar day passing,
                spot and volatility held fixed (typically negative -
                "time decay")
        rho   : $m change in book value per 1 percentage point
                (e.g. 2% -> 3%) move in the risk-free rate
    """
    common_kwargs = dict(n_inner_paths=n_inner_paths, n_inner_steps_per_year=n_inner_steps_per_year)

    bump_S = bump_pct * S
    S_up, S_down = S + bump_S, max(S - bump_S, 1e-6)

    value = _book_value(S, valuation_time, positions_df, r, params, **common_kwargs)
    value_S_up = _book_value(S_up, valuation_time, positions_df, r, params, **common_kwargs)
    value_S_down = _book_value(S_down, valuation_time, positions_df, r, params, **common_kwargs)

    delta = (value_S_up - value_S_down) / (S_up - S_down)
    # Second-order central difference: the standard finite-difference
    # estimate of curvature (d^2 V / dS^2), i.e. how fast Delta itself moves.
    gamma = (value_S_up - 2.0 * value + value_S_down) / (bump_S ** 2)

    value_vol_up = _book_value(S, valuation_time, positions_df, r, params, vol_shift=+vol_bump, **common_kwargs)
    value_vol_down = _book_value(S, valuation_time, positions_df, r, params, vol_shift=-vol_bump, **common_kwargs)
    # Raw central difference is $ per unit (100 percentage points) of vol;
    # rescale to the market convention of $ per ONE percentage point.
    vega = (value_vol_up - value_vol_down) / (2.0 * vol_bump) * 0.01

    value_r_up = _book_value(S, valuation_time, positions_df, r + r_bump, params, **common_kwargs)
    value_r_down = _book_value(S, valuation_time, positions_df, r - r_bump, params, **common_kwargs)
    # Same rescaling idea as Vega: $ per ONE percentage point move in rates.
    rho = (value_r_up - value_r_down) / (2.0 * r_bump) * 0.01

    value_next_day = _book_value(S, valuation_time + theta_dt, positions_df, r, params, **common_kwargs)
    # theta_dt is exactly one calendar day in years, so this difference
    # already IS the $ change over one day - no further rescaling needed.
    theta = value_next_day - value

    return {
        "value": value,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def compute_greeks_over_time(
    positions_df: pd.DataFrame,
    S0: float,
    r: float,
    params: Params,
    max_horizon: float,
    n_time_points: int = 12,
    **greek_kwargs,
) -> pd.DataFrame:
    """
    Computes the full set of book Greeks at a series of points in time from
    today (t=0) up to `max_horizon`, holding spot fixed at S0 throughout -
    i.e. "if the stock doesn't move, how do the book's Greeks evolve purely
    from time passing?" This isolates pure time-decay behaviour from
    spot-sensitivity behaviour (which the Delta/Gamma snapshot already
    captures at t=0).

    Parameters
    ----------
    positions_df : The option book.
    S0 : Spot price held fixed at every time point.
    r : Risk-free rate.
    params : Calibrated local volatility parameters.
    max_horizon : Latest valuation time to evaluate, in years. Typically set
        to the book's shortest time-to-expiry, since after that point the
        book's composition changes qualitatively (a leg expires).
    n_time_points : Number of points to evaluate between t=0 and max_horizon.
    **greek_kwargs : Passed through to compute_book_greeks (bump sizes,
        inner Monte Carlo resolution).

    Returns
    -------
    DataFrame with columns ['t', 'value', 'delta', 'gamma', 'vega', 'theta', 'rho'].
    """
    t_grid = np.linspace(0.0, max_horizon, n_time_points)
    rows: List[Dict[str, float]] = []

    for t in t_grid:
        greeks = compute_book_greeks(S0, float(t), positions_df, r, params, **greek_kwargs)
        rows.append({"t": float(t), **greeks})

    return pd.DataFrame(rows)
