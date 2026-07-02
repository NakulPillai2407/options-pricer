"""
vol_surface.py
---------------
Calibrates a parametric local volatility surface sigma(t, S) to the market
implied volatilities computed in implied_vol.py.

Why we need this at all: implied_vol.py gives us one volatility number per
*traded contract* (one (strike, maturity) pair). But the Monte Carlo engine
in monte_carlo.py needs to know "what volatility should I use right now",
for *any* spot price the simulation might wander to, at *any* time between
0 and T — including points that were never directly quoted in the market.
This module fits a smooth functional form through the scattered market
points so we have a volatility value everywhere on the (t, S) plane.

Two parametric forms are implemented, exactly mirroring the coursework:

1. A quadratic benchmark model (simpler, used as a baseline).
2. An SVI-inspired model (Stochastic Volatility Inspired), which is the
   industry-standard parameterisation real quant desks use to fit
   volatility smiles, because it captures skew (asymmetry between OTM
   puts and calls) and smile curvature with relatively few parameters.

In both models, every parameter is itself a function of time t that decays
exponentially towards a long-run value — this captures the well-known
market feature that volatility smiles are steep for short-dated options
and flatten out for longer-dated ones.

This is a practical, simplified take on Dupire's (1994) local volatility
framework: rather than deriving sigma_local(t, S) analytically from the
full implied vol surface via Dupire's PDE, we directly parameterise and
calibrate sigma_local(t, S) to fit the *observed* implied vols. This is a
common simplification in coursework/teaching contexts and is computationally
much lighter than solving Dupire's equation, at the cost of not being a
strict no-arbitrage local volatility surface.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

Params = Dict[str, float]


# ---------------------------------------------------------------------------
# Quadratic benchmark model
# ---------------------------------------------------------------------------

def quadratic_vol(t: np.ndarray, S: np.ndarray, params: Params) -> np.ndarray:
    """
    Evaluates the quadratic local volatility benchmark model.

    sigma(t, S) = a(t) + b(t) * (S - S0) + c(t) * (S - S0)^2

    where a(t), b(t), c(t) all decay exponentially from a short-term value
    towards a long-run value (for a(t)) or towards zero (for b(t), c(t)).

    Parameters
    ----------
    t : Time(s) in years (scalar or array).
    S : Spot or strike level(s) (scalar or array, broadcastable with t).
    params : Dictionary with keys:
        S0, atm0, a_min, atm_decay, skew0, skew_decay, c0, curv_decay

    Returns
    -------
    The model volatility, floored at 1% to avoid non-physical negative or
    near-zero volatility values destabilising downstream Monte Carlo paths.
    """
    # a(t): ATM volatility level, decaying from its short-term value 'atm0'
    # towards the long-run level 'a_min' as maturity increases.
    a_t = params["a_min"] + (params["atm0"] - params["a_min"]) * np.exp(-params["atm_decay"] * t)

    # b(t): linear skew slope, decaying towards zero — skew is strongest for
    # short maturities and fades out for longer-dated options.
    b_t = params["skew0"] * np.exp(-params["skew_decay"] * t)

    # c(t): curvature of the smile, also decaying towards zero over time.
    c_t = params["c0"] * np.exp(-params["curv_decay"] * t)

    vol = a_t + b_t * (S - params["S0"]) + c_t * (S - params["S0"]) ** 2

    return np.maximum(vol, 0.01)


def _quadratic_array_to_params(p: np.ndarray, S0: float) -> Params:
    return {
        "model": "quadratic",
        "S0": S0,
        "atm0": p[0],
        "a_min": p[1],
        "atm_decay": p[2],
        "skew0": p[3],
        "skew_decay": p[4],
        "c0": p[5],
        "curv_decay": p[6],
    }


def calibrate_quadratic(df: pd.DataFrame, S0: float) -> Tuple[Params, float]:
    """
    Calibrates the quadratic local volatility model to market implied vols
    by minimising mean squared error via L-BFGS-B, exactly as in the
    coursework.

    Parameters
    ----------
    df : DataFrame with columns ['T', 'strike', 'implied_vol'].
    S0 : Current spot price of the underlying.

    Returns
    -------
    (params, rmse) : the calibrated parameter dictionary and the
        root-mean-squared calibration error (in volatility units, e.g.
        0.05 means the average fitting error is 5 vol points).
    """
    maturities = df["T"].to_numpy()
    strikes = df["strike"].to_numpy()
    market_ivs = df["implied_vol"].to_numpy()

    def objective(p: np.ndarray) -> float:
        params = _quadratic_array_to_params(p, S0)
        model_ivs = quadratic_vol(maturities, strikes, params)
        return float(np.mean((model_ivs - market_ivs) ** 2))

    initial_guess = np.array([0.25, 0.20, 0.50, -0.01, 1.5, 0.001, 2.00])
    bounds = [
        (0.01, 1.0),
        (0.01, 1.0),
        (0.001, 10.0),
        (-0.05, 0.05),
        (0.001, 10.0),
        (-0.001, 0.001),
        (0.001, 10.0),
    ]

    result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

    params = _quadratic_array_to_params(result.x, S0)
    rmse = float(np.sqrt(result.fun))

    print(f"Quadratic calibration {'succeeded' if result.success else 'did not fully converge'}; RMSE = {rmse:.4f}")

    return params, rmse


# ---------------------------------------------------------------------------
# SVI-inspired model
# ---------------------------------------------------------------------------

def svi_vol(S: np.ndarray, S0: float, t: np.ndarray, params: Params) -> np.ndarray:
    """
    Evaluates the SVI-inspired local volatility model:

        sigma(t, S) = a(t) + b(t)*(S - S0)
                      + c(t) * [ rho*(S - S0) + sqrt( (S - S0)^2 + eta(t)^2 ) ]

    Each term has a direct market interpretation:
      - a(t): the at-the-money (ATM) volatility level at maturity t.
      - b(t): a linear skew/slope term.
      - c(t): the overall strength of the SVI curvature term — how
        "bent" the smile is away from the linear part.
      - rho (constant, doesn't decay with time): controls the *asymmetry*
        of the smile. A negative rho (typical for equities) makes
        downside (low strike / OTM put) volatility rise faster than
        upside volatility — this is the well-documented equity skew,
        reflecting that markets price in a higher probability/severity
        of crashes than rallies.
      - eta(t): a smoothness parameter — larger eta makes the curvature
        term behave more like a smooth, rounded bowl; smaller eta makes
        it sharper around S0.

    The sqrt(... + eta^2) term is the classic "SVI" building block: as
    |S - S0| grows large, this term grows approximately linearly (like
    rho*(S-S0) plus a roughly linear sqrt term), which is what allows SVI
    to fit both the curvature near the money and the near-linear wings
    far from the money that real implied volatility smiles exhibit.

    Parameters
    ----------
    S : Spot or strike level(s).
    S0 : Reference spot price (the smile is parameterised around this point).
    t : Time(s) in years.
    params : Dictionary with keys:
        atm0, a_min, atm_decay, skew0, skew_decay,
        c_svi0, c_svi_decay, eta0, eta_decay, rho

    Returns
    -------
    The model volatility, floored at 1%.
    """
    a_t = params["a_min"] + (params["atm0"] - params["a_min"]) * np.exp(-params["atm_decay"] * t)
    b_t = params["skew0"] * np.exp(-params["skew_decay"] * t)
    c_svi_t = params["c_svi0"] * np.exp(-params["c_svi_decay"] * t)
    eta_t = params["eta0"] * np.exp(-params["eta_decay"] * t)
    rho = params["rho"]

    moneyness = S - S0

    # The core SVI "wing" shape: linear asymmetry (rho term) plus a smooth
    # sqrt bowl that flattens the curvature far from the money.
    svi_component = rho * moneyness + np.sqrt(moneyness ** 2 + eta_t ** 2)

    vol = a_t + b_t * moneyness + c_svi_t * svi_component

    return np.maximum(vol, 0.01)


def _svi_array_to_params(p: np.ndarray, S0: float) -> Params:
    return {
        "model": "svi",
        "S0": S0,
        "atm0": p[0],
        "a_min": p[1],
        "atm_decay": p[2],
        "skew0": p[3],
        "skew_decay": p[4],
        "c_svi0": p[5],
        "c_svi_decay": p[6],
        "eta0": p[7],
        "eta_decay": p[8],
        "rho": p[9],
    }


def calibrate_svi(df: pd.DataFrame, S0: float) -> Tuple[Params, float]:
    """
    Calibrates the SVI-inspired local volatility model to market implied
    vols by minimising mean squared error via L-BFGS-B, exactly as in the
    coursework.

    Parameters
    ----------
    df : DataFrame with columns ['T', 'strike', 'implied_vol'].
    S0 : Current spot price of the underlying.

    Returns
    -------
    (params, rmse) : the calibrated parameter dictionary and the
        root-mean-squared calibration error (in volatility units).
    """
    maturities = df["T"].to_numpy()
    strikes = df["strike"].to_numpy()
    market_ivs = df["implied_vol"].to_numpy()

    def objective(p: np.ndarray) -> float:
        params = _svi_array_to_params(p, S0)
        model_ivs = svi_vol(strikes, S0, maturities, params)
        return float(np.mean((model_ivs - market_ivs) ** 2))

    initial_guess = np.array([0.25, 0.20, 0.50, -0.002, 1.50, 0.0005, 2.00, 50.00, 1.00, -0.50])
    bounds = [
        (0.01, 1.00),
        (0.01, 1.00),
        (0.001, 10.00),
        (-0.02, 0.02),
        (0.001, 10.00),
        (0.00, 0.01),
        (0.001, 10.00),
        (1.00, 200.00),
        (0.001, 10.00),
        (-0.99, 0.99),
    ]

    result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

    params = _svi_array_to_params(result.x, S0)
    rmse = float(np.sqrt(result.fun))

    print(f"SVI calibration {'succeeded' if result.success else 'did not fully converge'}; RMSE = {rmse:.4f}")

    return params, rmse


# ---------------------------------------------------------------------------
# Shared utilities: surface evaluation and lookup during simulation
# ---------------------------------------------------------------------------

def active_local_vol(t: np.ndarray, S: np.ndarray, params: Params) -> np.ndarray:
    """
    Dispatches to the correct local volatility function based on the
    'model' key stored inside `params` ('quadratic' or 'svi'). This lets
    the rest of the codebase (Monte Carlo, pricing) stay model-agnostic —
    it just calls this function and doesn't need to know which surface
    was actually calibrated.
    """
    model = params.get("model", "svi")

    if model == "quadratic":
        return quadratic_vol(t, S, params)
    elif model == "svi":
        return svi_vol(S, params["S0"], t, params)
    else:
        raise ValueError("Unknown local volatility model. Use 'quadratic' or 'svi'.")


def get_local_vol(t: float, S: np.ndarray, S0: float, params: Params) -> np.ndarray:
    """
    Looks up the local volatility at a given time t and spot level(s) S
    from an already-calibrated parameter set. This is the function the
    Monte Carlo engine calls at every time step.

    Since both the quadratic and SVI forms are smooth closed-form
    functions of (t, S) rather than a discretised grid, "interpolation"
    here just means directly evaluating the formula — there is no lookup
    error, unlike a grid-based local vol surface.

    Parameters
    ----------
    t : Current simulation time, in years.
    S : Current spot price(s) — can be a scalar or an array of paths.
    S0 : Reference spot price used when the surface was calibrated.
    params : Calibrated parameter dictionary (quadratic or SVI).

    Returns
    -------
    The local volatility value(s), same shape as S.
    """
    return active_local_vol(t, S, params)


def build_vol_surface(
    S0: float, params: Params, S_grid: np.ndarray, T_grid: np.ndarray
) -> np.ndarray:
    """
    Evaluates the calibrated local volatility model over a full grid of
    spot/strike values and maturities, for plotting a 3D surface.

    Parameters
    ----------
    S0 : Reference spot price.
    params : Calibrated parameter dictionary (quadratic or SVI).
    S_grid : 1D array of spot/strike values.
    T_grid : 1D array of maturities, in years.

    Returns
    -------
    A 2D numpy array of shape (len(T_grid), len(S_grid)) giving
    sigma(T, S) at every grid point.
    """
    S_mesh, T_mesh = np.meshgrid(S_grid, T_grid)
    surface = active_local_vol(T_mesh, S_mesh, params)
    return surface
