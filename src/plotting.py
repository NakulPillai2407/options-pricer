"""
plotting.py
-----------
All chart/visualisation functions for the app, built with Plotly so they
render as interactive charts inside Streamlit (zoom, hover tooltips, pan)
rather than static matplotlib images.

No maths or Streamlit calls live in this module — it only takes already-
computed data (DataFrames, numpy arrays, parameter dicts) and turns them
into Plotly figure objects. app.py is responsible for calling `st.plotly_chart`
on whatever these functions return.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .vol_surface import build_vol_surface


def plot_iv_smile(df: pd.DataFrame) -> go.Figure:
    """
    Plots implied volatility against strike, with one line per maturity —
    the classic "volatility smile/skew" chart (coursework Figure 1).

    Parameters
    ----------
    df : DataFrame with columns ['strike', 'T', 'implied_vol'] (and
        optionally 'option_type').
    """
    fig = px.line(
        df.sort_values("strike"),
        x="strike",
        y="implied_vol",
        color="T",
        markers=True,
        labels={"strike": "Strike", "implied_vol": "Implied Volatility", "T": "Maturity (yrs)"},
        title="Implied Volatility Smile by Maturity",
    )
    fig.update_yaxes(tickformat=".1%")
    return fig


def plot_iv_surface_3d(df: pd.DataFrame) -> go.Figure:
    """
    Plots a 3D scatter/mesh of implied volatility against strike and
    maturity, showing the full surface shape (coursework Figure 2).

    Parameters
    ----------
    df : DataFrame with columns ['strike', 'T', 'implied_vol'].
    """
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=df["strike"],
                y=df["T"],
                z=df["implied_vol"],
                intensity=df["implied_vol"],
                colorscale="Viridis",
                opacity=0.85,
            )
        ]
    )
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Maturity (yrs)",
            zaxis_title="Implied Volatility",
        ),
    )
    return fig


def plot_svi_calibration_fit(df: pd.DataFrame, S0: float, params: Dict[str, float]) -> go.Figure:
    """
    Plots market implied vol points against the calibrated model's fitted
    volatility, one colour per maturity (coursework Figure 3) — the key
    diagnostic chart for judging how well the SVI/quadratic fit matches
    the real market smile.

    Parameters
    ----------
    df : DataFrame with columns ['strike', 'T', 'implied_vol'].
    S0 : Reference spot price used in calibration.
    params : Calibrated model parameters (quadratic or SVI), including a
        'model' key.
    """
    from .vol_surface import active_local_vol

    fig = go.Figure()
    maturities = sorted(df["T"].unique())
    colors = px.colors.qualitative.Plotly

    for i, T in enumerate(maturities):
        color = colors[i % len(colors)]
        slice_df = df[df["T"] == T].sort_values("strike")

        fig.add_trace(
            go.Scatter(
                x=slice_df["strike"],
                y=slice_df["implied_vol"],
                mode="markers",
                name=f"Market IV (T={T:.2f})",
                marker=dict(color=color, size=7),
                legendgroup=f"T{i}",
            )
        )

        strike_grid = np.linspace(slice_df["strike"].min(), slice_df["strike"].max(), 100)
        fitted = active_local_vol(np.full_like(strike_grid, T), strike_grid, params)

        fig.add_trace(
            go.Scatter(
                x=strike_grid,
                y=fitted,
                mode="lines",
                name=f"Fitted (T={T:.2f})",
                line=dict(color=color),
                legendgroup=f"T{i}",
            )
        )

    fig.update_layout(
        title=f"{params.get('model', 'Model').upper()} Calibration Fit by Maturity",
        xaxis_title="Strike",
        yaxis_title="Volatility",
        yaxis_tickformat=".1%",
    )
    return fig


def plot_local_vol_surface(S0: float, params: Dict[str, float], n_grid: int = 40) -> go.Figure:
    """
    Plots the full calibrated local volatility surface sigma(t, S) as a 3D
    surface (coursework Figure 4).

    Parameters
    ----------
    S0 : Reference spot price.
    params : Calibrated model parameters.
    n_grid : Resolution of the spot/maturity grid used for plotting.
    """
    S_grid = np.linspace(S0 * 0.6, S0 * 1.4, n_grid)
    T_grid = np.linspace(0.05, 2.0, n_grid)

    surface = build_vol_surface(S0, params, S_grid, T_grid)

    fig = go.Figure(
        data=[go.Surface(x=S_grid, y=T_grid, z=surface, colorscale="Viridis")]
    )
    fig.update_layout(
        title=f"Calibrated {params.get('model', 'Local Vol').upper()} Local Volatility Surface",
        scene=dict(
            xaxis_title="Spot / Strike",
            yaxis_title="Maturity (yrs)",
            zaxis_title="Local Volatility",
        ),
    )
    return fig


def plot_spot_paths(paths: np.ndarray, time_grid: np.ndarray, n_display: int = 50) -> go.Figure:
    """
    Plots a sample of simulated spot price paths over time (coursework Figure 6).

    Parameters
    ----------
    paths : Array of shape (n_steps+1, n_paths).
    time_grid : Array of shape (n_steps+1,).
    n_display : How many individual paths to draw (drawing all 5,000 would
        be unreadable and slow to render).
    """
    n_display = min(n_display, paths.shape[1])
    fig = go.Figure()

    for i in range(n_display):
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=paths[:, i],
                mode="lines",
                line=dict(width=1),
                opacity=0.5,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"Simulated Spot Price Paths (showing {n_display} of {paths.shape[1]})",
        xaxis_title="Time (years)",
        yaxis_title="Spot Price",
    )
    return fig


def plot_fan_chart(paths: np.ndarray, time_grid: np.ndarray) -> go.Figure:
    """
    Plots a percentile "fan chart" showing how the distribution of
    simulated spot prices widens over time (coursework Figure 7).

    Parameters
    ----------
    paths : Array of shape (n_steps+1, n_paths).
    time_grid : Array of shape (n_steps+1,).
    """
    mean_path = np.mean(paths, axis=1)
    p01 = np.quantile(paths, 0.01, axis=1)
    p05 = np.quantile(paths, 0.05, axis=1)
    p25 = np.quantile(paths, 0.25, axis=1)
    p75 = np.quantile(paths, 0.75, axis=1)
    p95 = np.quantile(paths, 0.95, axis=1)
    p99 = np.quantile(paths, 0.99, axis=1)

    fig = go.Figure()

    band_specs = [
        (p01, p99, "1%-99% range", 0.15),
        (p05, p95, "5%-95% range", 0.25),
        (p25, p75, "25%-75% range", 0.35),
    ]

    for lower, upper, label, opacity in band_specs:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_grid, time_grid[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=f"rgba(31,119,180,{opacity})",
                line=dict(color="rgba(255,255,255,0)"),
                name=label,
            )
        )

    fig.add_trace(
        go.Scatter(x=time_grid, y=mean_path, mode="lines", line=dict(width=2.5, color="darkblue"), name="Mean spot path")
    )

    fig.add_hline(y=paths[0, 0], line_dash="dash", annotation_text="Initial spot")

    fig.update_layout(
        title="Monte Carlo Spot Price Fan Chart",
        xaxis_title="Time (years)",
        yaxis_title="Spot Price",
    )
    return fig


def plot_pnl_distribution(
    pnl_array: np.ndarray, label: str, var: Optional[float] = None, es: Optional[float] = None
) -> go.Figure:
    """
    Plots a histogram of simulated P&L outcomes, with optional vertical
    lines marking the VaR and ES cutoffs (coursework Figure 8-style chart).

    Parameters
    ----------
    pnl_array : Array of simulated P&L values.
    label : Chart title / series label, e.g. "Naked P&L" or "Delta-Hedged P&L".
    var, es : Optional VaR/ES values (as positive loss numbers) to mark as
        vertical lines at -var and -es on the P&L axis.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl_array, nbinsx=50, name=label, marker_line_color="black", marker_line_width=1))

    fig.add_vline(x=float(np.mean(pnl_array)), line_dash="dash", annotation_text="Mean P&L")

    if var is not None:
        fig.add_vline(x=-var, line_dash="dot", line_color="red", annotation_text="99% VaR cutoff")
    if es is not None:
        fig.add_vline(x=-es, line_dash="dot", line_color="darkred", annotation_text="99% ES cutoff")

    fig.update_layout(title=f"{label} Distribution", xaxis_title="P&L ($m)", yaxis_title="Frequency")
    return fig


def plot_naked_vs_hedged(
    naked_pnl: np.ndarray, hedged_pnl: np.ndarray, naked_var: float, hedged_var: float
) -> go.Figure:
    """
    Overlays naked vs delta-hedged P&L histograms with their respective
    99% VaR cutoffs marked, directly mirroring coursework Figures 9-10 —
    the headline chart showing how dramatically hedging compresses the
    P&L distribution.

    Parameters
    ----------
    naked_pnl, hedged_pnl : Arrays of simulated P&L outcomes.
    naked_var, hedged_var : VaR values (as positive loss numbers) for each.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=naked_pnl, nbinsx=50, name="Naked P&L", opacity=0.55, marker_line_color="black", marker_line_width=1))
    fig.add_trace(go.Histogram(x=hedged_pnl, nbinsx=50, name="Delta-Hedged P&L", opacity=0.55, marker_line_color="black", marker_line_width=1))
    fig.update_layout(barmode="overlay")

    fig.add_vline(x=-naked_var, line_dash="dash", line_color="blue", annotation_text="Naked 99% VaR")
    fig.add_vline(x=-hedged_var, line_dash="dot", line_color="orange", annotation_text="Hedged 99% VaR")

    fig.update_layout(
        title="Naked vs Delta-Hedged P&L Distribution",
        xaxis_title="P&L ($m)",
        yaxis_title="Frequency",
    )
    return fig


def plot_greeks_evolution(greeks_df: pd.DataFrame) -> go.Figure:
    """
    Plots how each of the book's Greeks evolves as time passes with spot
    held fixed, as one small line-chart subplot per Greek sharing a common
    time axis - the "comprehensive overview" chart for the Greeks tab.

    Parameters
    ----------
    greeks_df : DataFrame from greeks.compute_greeks_over_time, with columns
        ['t', 'delta', 'gamma', 'vega', 'theta', 'rho'].
    """
    specs = [
        ("delta", "Delta ($m per $1 spot move)"),
        ("gamma", "Gamma (change in Delta per $1 spot move)"),
        ("vega", "Vega ($m per 1pt vol move)"),
        ("theta", "Theta ($m per day)"),
        ("rho", "Rho ($m per 1% rate move)"),
    ]
    grid_positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]

    fig = make_subplots(rows=3, cols=2, subplot_titles=[label for _, label in specs])

    for (column, _), (row_i, col_i) in zip(specs, grid_positions):
        fig.add_trace(
            go.Scatter(x=greeks_df["t"], y=greeks_df[column], mode="lines+markers", showlegend=False),
            row=row_i,
            col=col_i,
        )

    fig.update_xaxes(title_text="Time from today (years)")
    fig.update_layout(title="Book Greeks Evolution Over Time (spot held fixed)", height=800)
    return fig


def plot_risk_summary_table(comparison_dict: Dict[str, float]) -> go.Figure:
    """
    Renders the naked-vs-hedged risk comparison dictionary (from
    risk.compare_naked_vs_hedged) as a clean Plotly table, matching the
    coursework's Table 2 summary.

    Parameters
    ----------
    comparison_dict : Output of risk.compare_naked_vs_hedged.
    """
    rows = [
        ("99% VaR ($m)", comparison_dict["naked_var"], comparison_dict["hedged_var"], comparison_dict["var_reduction_pct"]),
        ("99% Expected Shortfall ($m)", comparison_dict["naked_es"], comparison_dict["hedged_es"], comparison_dict["es_reduction_pct"]),
        ("Std. Dev. of P&L ($m)", comparison_dict["naked_std"], comparison_dict["hedged_std"], comparison_dict["std_reduction_pct"]),
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Risk Measure", "Naked", "Delta-Hedged", "Reduction (%)"], fill_color="#1f77b4", font=dict(color="white")),
                cells=dict(
                    values=[
                        [r[0] for r in rows],
                        [f"{r[1]:.4f}" for r in rows],
                        [f"{r[2]:.4f}" for r in rows],
                        [f"{r[3]:.1f}%" for r in rows],
                    ]
                ),
            )
        ]
    )
    fig.update_layout(title="Hedging Effectiveness Summary")
    return fig
