"""
monte_carlo.py
---------------
The simulation engine: generates simulated future paths for the underlying
spot price under the calibrated local volatility surface from
vol_surface.py.

The single most important conceptual point in this module — and the one
most often glossed over in practice — is the distinction between two
different "measures" (i.e. two different assumed drift rates) used for two
different jobs:

1. REAL-WORLD measure (drift = mu): used to answer "what might actually
   happen to the stock and my P&L over the next year?" This is what risk
   management cares about. mu is the real-world expected return investors
   require for holding the stock (e.g. 5%), and it is NOT used for pricing.

2. RISK-NEUTRAL measure (drift = r): used to answer "what is this option
   worth today?" Option pricing theory (Black-Scholes, and more generally
   no-arbitrage pricing) shows that derivative prices can be computed as a
   discounted expectation under a *hypothetical* risk-neutral world where
   every asset grows at the risk-free rate r — regardless of investors'
   actual real-world risk appetite. This isn't because anyone thinks the
   stock will really grow at r; it's a mathematical device that removes
   the (unobservable) market price of risk from the pricing formula.

Mixing these up is a classic and serious error: pricing options using mu
instead of r would systematically misprice every option in the book (and
violate no-arbitrage), while computing real-world P&L scenarios using r
instead of mu would understate/overstate how the stock is actually expected
to drift, distorting VaR and ES.

Both simulations use the *same* calibrated local volatility surface
sigma(t, S) — only the drift differs.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .vol_surface import active_local_vol

Params = Dict[str, float]


def simulate_paths(
    S0: float,
    mu: float,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    S0_ref: float,
    params: Params,
    seed: int | None = None,
    vol_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates stock price paths under BOTH the real-world measure (drift mu)
    and the risk-neutral measure (drift r), over a common time grid, using
    the calibrated local volatility surface.

    Discretisation scheme (Euler-Maruyama applied to the log-price, which
    guarantees S stays positive — unlike a naive Euler scheme on S itself):

        S_{t+dt} = S_t * exp( (drift - 0.5*sigma_local^2) * dt
                               + sigma_local * sqrt(dt) * Z )

    where Z ~ N(0, 1) is a fresh independent standard normal random shock
    at every step, and sigma_local = sigma(t, S_t) is looked up fresh at
    every step from the calibrated local vol surface (this is what makes it
    a *local* volatility simulation rather than a constant-volatility
    Black-Scholes simulation: volatility responds to where the spot price
    currently is, and to how much time is left).

    Term-by-term intuition:
      - (drift - 0.5*sigma_local^2) * dt: the expected change in log-price
        over the step. The "-0.5*sigma^2" piece is the Ito/Jensen
        correction — without it, exp(drift*dt) would overstate the
        expected *level* of S, because volatility itself drags down the
        expected log-return (variance reduces the median relative to the
        mean of a log-normal distribution).
      - sigma_local * sqrt(dt) * Z: the random shock for this step. The
        sqrt(dt) scaling reflects that the *standard deviation* of a
        Brownian increment grows with the square root of time (variance
        grows linearly with time, since each step's noise is independent).

    Parameters
    ----------
    S0 : Starting spot price for both sets of simulated paths.
    mu : Real-world drift rate (decimal), used for the real-world paths.
    r : Risk-free rate (decimal), used for the risk-neutral paths.
    T : Simulation horizon, in years.
    n_steps : Number of discrete time steps over [0, T] (e.g. 252 for daily
        steps over one year).
    n_paths : Number of independent Monte Carlo paths to simulate.
    S0_ref : The reference spot price the local vol surface was calibrated
        around (passed through to active_local_vol — usually equal to S0,
        but kept separate in case the simulation starts from a different
        spot than the calibration anchor).
    params : Calibrated local volatility parameters from vol_surface.py.
    seed : Optional random seed for reproducibility.
    vol_shift : Optional additive shift applied to the local volatility at
        every step (e.g. 0.01 to shift volatility up by 1 percentage point).
        Used by greeks.py to estimate Vega by repricing under a shifted
        volatility level; zero by default, which reproduces the plain
        calibrated surface exactly.

    Returns
    -------
    (time_grid, real_paths, rn_paths)
        time_grid : 1D array of shape (n_steps + 1,), the simulation times.
        real_paths : array of shape (n_steps + 1, n_paths), simulated under
            the real-world measure (drift mu) — used for P&L/risk scenarios.
        rn_paths : array of shape (n_steps + 1, n_paths), simulated under
            the risk-neutral measure (drift r) — used for pricing.

    Note
    ----
    real_paths and rn_paths use independent random shocks (not the same Z
    draws), since they represent two genuinely different simulated worlds
    used for two different purposes, exactly as in the coursework.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    time_grid = np.linspace(0.0, T, n_steps + 1)

    real_paths = np.zeros((n_steps + 1, n_paths))
    rn_paths = np.zeros((n_steps + 1, n_paths))
    real_paths[0, :] = S0
    rn_paths[0, :] = S0

    for step in range(1, n_steps + 1):
        t_prev = time_grid[step - 1]

        # --- Real-world path update (drift = mu) ---
        S_real_prev = real_paths[step - 1, :]
        # Floored so a large negative vol_shift can never push volatility to
        # zero or below, which would break the sqrt(dt) scaling below.
        sigma_real = np.maximum(active_local_vol(t_prev, S_real_prev, params) + vol_shift, 1e-4)
        Z_real = np.random.standard_normal(n_paths)
        log_return_real = (mu - 0.5 * sigma_real ** 2) * dt + sigma_real * np.sqrt(dt) * Z_real
        real_paths[step, :] = S_real_prev * np.exp(log_return_real)

        # --- Risk-neutral path update (drift = r) ---
        S_rn_prev = rn_paths[step - 1, :]
        sigma_rn = np.maximum(active_local_vol(t_prev, S_rn_prev, params) + vol_shift, 1e-4)
        Z_rn = np.random.standard_normal(n_paths)
        log_return_rn = (r - 0.5 * sigma_rn ** 2) * dt + sigma_rn * np.sqrt(dt) * Z_rn
        rn_paths[step, :] = S_rn_prev * np.exp(log_return_rn)

    return time_grid, real_paths, rn_paths


def compute_terminal_distribution(paths: np.ndarray) -> Dict[str, float]:
    """
    Extracts the terminal (final time step) prices from a set of simulated
    paths and computes summary statistics.

    Parameters
    ----------
    paths : Array of shape (n_steps + 1, n_paths), as returned by
        simulate_paths.

    Returns
    -------
    A dictionary with the terminal mean, standard deviation, and the 1%,
    5%, 50%, 95%, 99% quantiles, plus min/max — giving a quick numerical
    picture of how dispersed the simulated outcomes are a year out.
    """
    terminal = paths[-1, :]

    return {
        "mean": float(np.mean(terminal)),
        "std": float(np.std(terminal)),
        "min": float(np.min(terminal)),
        "p01": float(np.quantile(terminal, 0.01)),
        "p05": float(np.quantile(terminal, 0.05)),
        "median": float(np.quantile(terminal, 0.50)),
        "p95": float(np.quantile(terminal, 0.95)),
        "p99": float(np.quantile(terminal, 0.99)),
        "max": float(np.max(terminal)),
    }
