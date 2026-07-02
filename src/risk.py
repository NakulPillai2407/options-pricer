"""
risk.py
-------
Risk metrics for the option book: naked (unhedged) P&L distribution,
Value-at-Risk (VaR), Expected Shortfall (ES), and a daily delta-hedging
simulation that mirrors the coursework's hedging analysis.

This module is intentionally the most "infrastructure-heavy" part of the
codebase, because faithfully revaluing a book of options at a future
horizon (or at every day along a hedging path) requires *nested* Monte
Carlo: an outer simulation of possible future worlds, and an inner
simulation (run at each outer scenario / time / spot point) to price the
options that are still alive in that scenario. To keep this fast enough
for an interactive app, we follow the coursework's approach of building a
(time, spot) grid of pre-computed book values/deltas via local-vol Monte
Carlo, then interpolating cheaply for every actual scenario/day — instead
of re-running a fresh Monte Carlo at every single one of the (5,000
scenarios) x (252 days) points, which would be computationally infeasible.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .monte_carlo import simulate_paths
from .pricing import _option_payoff

Params = Dict[str, float]


# ---------------------------------------------------------------------------
# Nested-MC book valuation helpers
# ---------------------------------------------------------------------------

def _book_value_and_delta_at_point(
    S: float,
    valuation_time: float,
    positions_df: pd.DataFrame,
    r: float,
    params: Params,
    n_inner_paths: int = 1000,
    n_inner_steps_per_year: int = 52,
    bump_pct: float = 0.005,
) -> tuple[float, float]:
    """
    Values the full book (and estimates its delta) at one specific
    (spot, time) point, using inner local-vol Monte Carlo for any options
    that have not yet expired by `valuation_time`.

    For options whose expiry has already passed by valuation_time, the
    value is just the (already-known) intrinsic payoff — no simulation
    needed. For options still alive, we simulate forward over the
    remaining time-to-expiry under the risk-neutral measure and discount
    the average payoff.

    Delta is estimated via central bump-and-revalue (see pricing.py for the
    rationale on why central, not one-sided, differences are used), reusing
    the SAME random shocks for the base/up/down revaluations so that the
    delta estimate isn't itself contaminated by Monte Carlo noise between
    the three runs (a standard variance-reduction trick: common random
    numbers).
    """
    bump = bump_pct * S
    S_up = S + bump
    S_down = max(S - bump, 1e-6)

    def value_at(S_eval: float) -> float:
        total = 0.0
        for _, row in positions_df.iterrows():
            tau = row["expiry_T"] - valuation_time
            sign = 1.0 if row["position"] == "long" else -1.0

            if tau <= 0:
                payoff = _option_payoff(np.array([S_eval]), row["strike"], row["option_type"])[0]
                total += sign * row["notional"] * payoff
            else:
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
                )
                terminal = rn_paths[-1, :]
                payoff = float(np.mean(_option_payoff(terminal, row["strike"], row["option_type"])))
                discounted = np.exp(-r * tau) * payoff
                total += sign * row["notional"] * discounted
        return total

    value = value_at(S)
    value_up = value_at(S_up)
    value_down = value_at(S_down)

    delta = (value_up - value_down) / (S_up - S_down)

    return value, delta


def build_horizon_revaluation_grids(
    positions_df: pd.DataFrame,
    horizon: float,
    spot_range: tuple[float, float],
    r: float,
    params: Params,
    n_grid: int = 40,
    n_inner_paths: int = 1000,
    n_inner_steps_per_year: int = 52,
) -> Dict[int, dict]:
    """
    Builds, for each option still alive at the risk horizon, a 1D grid of
    {spot -> price} so that the book can be cheaply revalued by
    interpolation across thousands of terminal-spot scenarios, instead of
    re-running Monte Carlo per scenario.

    Parameters
    ----------
    positions_df : The option book.
    horizon : The risk horizon, in years (e.g. 1.0 for a one-year horizon).
    spot_range : (min, max) spot levels the grid needs to cover — should
        span both the simulated terminal spot distribution and the book's
        strikes, with some padding.
    r : Risk-free rate.
    params : Calibrated local volatility parameters.
    n_grid : Number of spot grid points per option.
    n_inner_paths, n_inner_steps_per_year : Inner Monte Carlo resolution.

    Returns
    -------
    Dict keyed by position row index, each value a dict with
    {'spot_grid': ..., 'price_grid': ...} for options with positive
    remaining maturity at the horizon. Options already expired by the
    horizon are omitted (their value is just intrinsic payoff, computed
    directly rather than via a grid).
    """
    spot_grid = np.linspace(max(spot_range[0], 1e-6), spot_range[1], n_grid)
    grids: Dict[int, dict] = {}

    for idx, row in positions_df.iterrows():
        remaining = row["expiry_T"] - horizon
        if remaining <= 0:
            continue

        n_steps = max(int(np.ceil(remaining * n_inner_steps_per_year)), 1)
        shocks_seed_paths = n_inner_paths

        prices = []
        for S in spot_grid:
            _, _, rn_paths = simulate_paths(
                S0=float(S),
                mu=r,
                r=r,
                T=remaining,
                n_steps=n_steps,
                n_paths=shocks_seed_paths,
                S0_ref=params["S0"],
                params=params,
            )
            terminal = rn_paths[-1, :]
            payoff = float(np.mean(_option_payoff(terminal, row["strike"], row["option_type"])))
            prices.append(np.exp(-r * remaining) * payoff)

        grids[idx] = {
            "spot_grid": spot_grid,
            "price_grid": np.array(prices),
            "remaining_term": remaining,
        }

    return grids


def book_value_at_horizon(
    S_T: np.ndarray,
    positions_df: pd.DataFrame,
    horizon: float,
    revaluation_grids: Dict[int, dict],
) -> np.ndarray:
    """
    Computes total book value across many simulated terminal spot
    scenarios `S_T`, using direct intrinsic payoff for expired options and
    grid interpolation (from build_horizon_revaluation_grids) for options
    still alive at the horizon.

    Parameters
    ----------
    S_T : Array of simulated terminal spot prices (one per scenario).
    positions_df : The option book.
    horizon : The risk horizon, in years.
    revaluation_grids : Output of build_horizon_revaluation_grids.

    Returns
    -------
    Array of total book values (one per scenario), same length as S_T.
    """
    total = np.zeros_like(S_T, dtype=float)

    for idx, row in positions_df.iterrows():
        remaining = row["expiry_T"] - horizon
        sign = 1.0 if row["position"] == "long" else -1.0

        if remaining <= 0:
            unit_value = _option_payoff(S_T, row["strike"], row["option_type"])
        else:
            grid = revaluation_grids[idx]
            unit_value = np.interp(S_T, grid["spot_grid"], grid["price_grid"])

        total += sign * row["notional"] * unit_value

    return total


# ---------------------------------------------------------------------------
# Naked P&L
# ---------------------------------------------------------------------------

def compute_naked_pnl(
    positions_df: pd.DataFrame,
    real_paths: np.ndarray,
    horizon: float,
    initial_book_value: float,
    r: float,
    params: Params,
    n_grid: int = 40,
    n_inner_paths: int = 1000,
    n_inner_steps_per_year: int = 52,
) -> np.ndarray:
    """
    Computes the unhedged ("naked") one-period P&L of the option book
    across all simulated real-world scenarios.

    P&L per scenario = (book value at the horizon, given that scenario's
    terminal spot) minus (today's book value, carried forward to the
    horizon at the risk-free rate).

    We carry the initial book value forward at the risk-free rate (rather
    than comparing today's value directly to the future value) because
    holding cash also earns interest — comparing like-for-like requires
    accounting for the time value of money on the initial investment too.

    Parameters
    ----------
    positions_df : The option book.
    real_paths : Real-world simulated paths, shape (n_steps+1, n_paths).
    horizon : Risk horizon, in years (typically the final time in real_paths).
    initial_book_value : Today's book value, in $m (from pricing.price_book).
    r : Risk-free rate.
    params : Calibrated local volatility parameters.
    n_grid, n_inner_paths, n_inner_steps_per_year : Inner-MC resolution for
        revaluing options still alive at the horizon.

    Returns
    -------
    Array of P&L values (in $m), one per simulated scenario.
    """
    terminal_spots = real_paths[-1, :]

    spot_min = min(float(terminal_spots.min()), float(positions_df["strike"].min())) * 0.9
    spot_max = max(float(terminal_spots.max()), float(positions_df["strike"].max())) * 1.1

    grids = build_horizon_revaluation_grids(
        positions_df=positions_df,
        horizon=horizon,
        spot_range=(spot_min, spot_max),
        r=r,
        params=params,
        n_grid=n_grid,
        n_inner_paths=n_inner_paths,
        n_inner_steps_per_year=n_inner_steps_per_year,
    )

    horizon_values = book_value_at_horizon(terminal_spots, positions_df, horizon, grids)

    accumulated_initial_value = initial_book_value * np.exp(r * horizon)

    return horizon_values - accumulated_initial_value


# ---------------------------------------------------------------------------
# VaR and Expected Shortfall
# ---------------------------------------------------------------------------

def compute_var(pnl_array: np.ndarray, confidence: float = 0.99) -> float:
    """
    Value-at-Risk (VaR): the loss level that is exceeded in only
    (1 - confidence) of scenarios.

    Plain-English meaning: a 99% VaR of $0.124m means "there is only a 1%
    chance of losing MORE than $0.124m over the risk horizon" — it is a
    *threshold*, not the worst possible loss, and it tells you nothing
    about how bad losses beyond that threshold could get (that's what ES,
    below, is for).

    Parameters
    ----------
    pnl_array : Array of simulated P&L outcomes (can be naked or hedged).
    confidence : Confidence level, e.g. 0.99 for 99% VaR.

    Returns
    -------
    VaR as a positive number representing a loss (in the same units as
    pnl_array).
    """
    losses = -np.asarray(pnl_array)
    return float(np.quantile(losses, confidence))


def compute_es(pnl_array: np.ndarray, confidence: float = 0.99) -> float:
    """
    Expected Shortfall (ES), also known as Conditional VaR (CVaR): the
    AVERAGE loss across only the worst (1 - confidence) tail of scenarios
    (i.e. averaged over all scenarios at least as bad as the VaR threshold).

    Why regulators (Basel/FRTB) increasingly prefer ES over VaR: VaR only
    tells you the loss threshold at a given confidence level, but says
    nothing about how severe losses get beyond that threshold — two
    portfolios can have identical VaR but very different tail severity
    (one might have a long, fat tail of catastrophic losses just beyond
    the VaR cutoff). ES directly captures that severity by averaging over
    the whole tail, which is why FRTB (Fundamental Review of the Trading
    Book) moved the standard capital metric from 99% VaR to 97.5% ES.

    Parameters
    ----------
    pnl_array : Array of simulated P&L outcomes.
    confidence : Confidence level, e.g. 0.99 for the 99% ES.

    Returns
    -------
    ES as a positive number representing a loss.
    """
    losses = -np.asarray(pnl_array)
    var = np.quantile(losses, confidence)
    tail_losses = losses[losses >= var]
    return float(np.mean(tail_losses))


# ---------------------------------------------------------------------------
# Delta hedging
# ---------------------------------------------------------------------------

def build_hedge_grids(
    positions_df: pd.DataFrame,
    time_grid: np.ndarray,
    spot_range: tuple[float, float],
    r: float,
    params: Params,
    n_time_grid: int = 13,
    n_spot_grid: int = 35,
    n_inner_paths: int = 500,
    n_inner_steps_per_year: int = 52,
    bump_pct: float = 0.005,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    """
    Pre-computes book value and book delta on a coarse (time, spot) grid
    via nested local-vol Monte Carlo, then wraps the grids in fast
    interpolators. This is what makes daily delta-hedging over thousands
    of scenarios computationally tractable: instead of running fresh
    Monte Carlo at every (scenario, day) pair, we run it only on a coarse
    grid once, and interpolate cheaply afterwards.

    Parameters
    ----------
    positions_df : The option book.
    time_grid : 1D array spanning [0, horizon] — only its min/max are used
        to build a *coarser* internal grid of size n_time_grid.
    spot_range : (min, max) spot levels to cover, generously padded to
        contain the full range of simulated paths plus book strikes.
    r : Risk-free rate.
    params : Calibrated local volatility parameters.
    n_time_grid, n_spot_grid : Resolution of the coarse grid.
    n_inner_paths, n_inner_steps_per_year : Inner Monte Carlo resolution.
    bump_pct : Bump size (as a fraction of spot) for the delta estimate.

    Returns
    -------
    (value_interpolator, delta_interpolator) : two scipy
    RegularGridInterpolator objects, each callable as
    `interpolator(np.column_stack([t_array, S_array]))`.
    """
    hedge_time_grid = np.linspace(time_grid.min(), time_grid.max(), n_time_grid)
    hedge_spot_grid = np.linspace(max(spot_range[0], 1e-6), spot_range[1], n_spot_grid)

    value_grid = np.zeros((n_time_grid, n_spot_grid))
    delta_grid = np.zeros((n_time_grid, n_spot_grid))

    for ti, t in enumerate(hedge_time_grid):
        for si, S in enumerate(hedge_spot_grid):
            value, delta = _book_value_and_delta_at_point(
                S=float(S),
                valuation_time=float(t),
                positions_df=positions_df,
                r=r,
                params=params,
                n_inner_paths=n_inner_paths,
                n_inner_steps_per_year=n_inner_steps_per_year,
                bump_pct=bump_pct,
            )
            value_grid[ti, si] = value
            delta_grid[ti, si] = delta

    value_interp = RegularGridInterpolator(
        (hedge_time_grid, hedge_spot_grid), value_grid, bounds_error=False, fill_value=None
    )
    delta_interp = RegularGridInterpolator(
        (hedge_time_grid, hedge_spot_grid), delta_grid, bounds_error=False, fill_value=None
    )

    return value_interp, delta_interp


def run_delta_hedge_simulation(
    positions_df: pd.DataFrame,
    real_paths: np.ndarray,
    time_grid: np.ndarray,
    horizon_book_values: np.ndarray,
    initial_book_value: float,
    delta_interpolator: RegularGridInterpolator,
    r: float,
) -> np.ndarray:
    """
    Simulates the P&L of a daily-rebalanced delta-hedged book, replicating
    the coursework's hedging mechanics exactly:

      1. At t=0, compute the book's delta and take an offsetting stock
         position: hedge_units = -delta (a short stock position if delta
         is positive, since a positive book delta means the book gains
         value when the stock rises — we sell stock to cancel that).
      2. A cash account finances the hedge: buying/selling stock for the
         hedge moves cash in the opposite direction, and the cash balance
         itself earns interest at the risk-free rate between rebalances.
      3. Every day, re-compute the book's delta at the new (time, spot)
         point and adjust (rebalance) the hedge position to match, paying
         for the trade out of the cash account.
      4. At the horizon, total hedged portfolio value = option book value
         + stock hedge value + cash account balance. Hedged P&L compares
         this to the initial book value carried forward at the risk-free
         rate (consistent with how naked P&L is computed in
         compute_naked_pnl).

    Parameters
    ----------
    positions_df : The option book.
    real_paths : Real-world simulated paths, shape (n_steps+1, n_paths).
    time_grid : The simulation time grid, shape (n_steps+1,), matching real_paths.
    horizon_book_values : Pre-computed option book values at the horizon for
        each scenario (from book_value_at_horizon) — reused here so we
        don't recompute them.
    initial_book_value : Today's book value, in $m.
    delta_interpolator : Output of build_hedge_grids (the delta grid).
    r : Risk-free rate.

    Returns
    -------
    Array of hedged P&L values (in $m), one per scenario.
    """
    n_steps = real_paths.shape[0] - 1
    n_paths = real_paths.shape[1]

    S0_array = real_paths[0, :]

    initial_points = np.column_stack([np.zeros(n_paths), S0_array])
    initial_delta = delta_interpolator(initial_points)

    # We go short stock in proportion to the book's delta, so that a $1
    # move in the stock changes the hedge value by -delta dollars,
    # exactly offsetting the book's own delta exposure.
    hedge_units = -initial_delta

    # The cash account is credited/debited by the cost of putting on the
    # hedge: selling stock (hedge_units negative) brings cash IN.
    cash_account = -hedge_units * S0_array

    for step in range(1, n_steps + 1):
        dt_step = time_grid[step] - time_grid[step - 1]
        S_t = real_paths[step, :]

        # Cash sitting in the account between rebalances earns interest at
        # the risk-free rate, just like a real margin/cash account would.
        cash_account = cash_account * np.exp(r * dt_step)

        if step < n_steps:
            t = time_grid[step]
            points = np.column_stack([np.full(n_paths, t), S_t])
            current_delta = delta_interpolator(points)

            new_hedge_units = -current_delta
            hedge_trade = new_hedge_units - hedge_units

            # Buying stock to increase the hedge costs cash (cash falls);
            # selling stock to reduce the hedge raises cash.
            cash_account = cash_account - hedge_trade * S_t

            hedge_units = new_hedge_units

    final_spots = real_paths[-1, :]
    final_hedge_value = hedge_units * final_spots

    final_hedged_value = horizon_book_values + final_hedge_value + cash_account

    accumulated_initial_value = initial_book_value * np.exp(r * time_grid[-1])

    hedged_pnl = final_hedged_value - accumulated_initial_value

    return hedged_pnl


def compare_naked_vs_hedged(
    naked_pnl: np.ndarray, hedged_pnl: np.ndarray, confidence: float = 0.99
) -> Dict[str, float]:
    """
    Builds a side-by-side comparison of naked vs delta-hedged risk metrics,
    matching the coursework's Table 2 (hedging effectiveness summary).

    Parameters
    ----------
    naked_pnl, hedged_pnl : Arrays of simulated P&L outcomes.
    confidence : Confidence level for VaR/ES, e.g. 0.99.

    Returns
    -------
    Dictionary with naked/hedged VaR, ES, standard deviation, and the
    percentage reduction achieved by hedging for each metric.
    """
    naked_var = compute_var(naked_pnl, confidence)
    hedged_var = compute_var(hedged_pnl, confidence)

    naked_es = compute_es(naked_pnl, confidence)
    hedged_es = compute_es(hedged_pnl, confidence)

    naked_std = float(np.std(naked_pnl))
    hedged_std = float(np.std(hedged_pnl))

    def pct_reduction(naked: float, hedged: float) -> float:
        if naked == 0:
            return 0.0
        return float((naked - hedged) / naked * 100.0)

    return {
        "naked_var": naked_var,
        "hedged_var": hedged_var,
        "var_reduction_pct": pct_reduction(naked_var, hedged_var),
        "naked_es": naked_es,
        "hedged_es": hedged_es,
        "es_reduction_pct": pct_reduction(naked_es, hedged_es),
        "naked_std": naked_std,
        "hedged_std": hedged_std,
        "std_reduction_pct": pct_reduction(naked_std, hedged_std),
    }
