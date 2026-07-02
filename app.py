"""
app.py
------
Streamlit entry point for the Options Pricing & Risk Management Tool.

This file is intentionally "thin" — it only handles UI layout, session
state, and wiring together calls into the `src/` modules. No pricing maths
or data cleaning logic lives here; that separation keeps the quantitative
core testable and reusable outside of Streamlit.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data_ingestion import DataIngestionError, fetch_option_chain, get_risk_free_rate, get_spot_price
from src.greeks import compute_book_greeks, compute_greeks_over_time
from src.implied_vol import add_implied_vols
from src.monte_carlo import compute_terminal_distribution, simulate_paths
from src.pricing import compute_book_delta, price_book
from src.plotting import (
    plot_fan_chart,
    plot_greeks_evolution,
    plot_iv_smile,
    plot_iv_surface_3d,
    plot_local_vol_surface,
    plot_naked_vs_hedged,
    plot_pnl_distribution,
    plot_risk_summary_table,
    plot_spot_paths,
    plot_svi_calibration_fit,
)
from src.risk import (
    build_hedge_grids,
    build_horizon_revaluation_grids,
    book_value_at_horizon,
    compare_naked_vs_hedged,
    compute_es,
    compute_naked_pnl,
    compute_var,
    run_delta_hedge_simulation,
)
from src.vol_surface import calibrate_quadratic, calibrate_svi

st.set_page_config(page_title="Options Pricing & Risk Tool", layout="wide")


# ---------------------------------------------------------------------------
# Cached data/calibration helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_load_market_data(ticker: str, max_expiries: int):
    """Fetches spot, risk-free rate, and a cleaned option chain for `ticker`."""
    spot = get_spot_price(ticker)
    rate = get_risk_free_rate()
    chain = fetch_option_chain(ticker, max_expiries=max_expiries)
    return spot, rate, chain


@st.cache_data(show_spinner=False)
def cached_implied_vols(chain: pd.DataFrame, spot: float, rate: float):
    return add_implied_vols(chain, S=spot, r=rate)


@st.cache_data(show_spinner=False)
def cached_calibrate_svi(iv_df: pd.DataFrame, spot: float):
    return calibrate_svi(iv_df, S0=spot)


@st.cache_data(show_spinner=False)
def cached_calibrate_quadratic(iv_df: pd.DataFrame, spot: float):
    return calibrate_quadratic(iv_df, S0=spot)


@st.cache_data(show_spinner=False)
def cached_simulate_paths(spot, mu, rate, horizon, n_steps, n_paths, params, seed):
    return simulate_paths(
        S0=spot, mu=mu, r=rate, T=horizon, n_steps=n_steps, n_paths=n_paths, S0_ref=spot, params=params, seed=seed
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Options Pricing & Risk Tool")

ticker = st.sidebar.text_input("Stock ticker", value="AAPL").strip().upper()

manual_rate_override = st.sidebar.checkbox("Override risk-free rate manually")
if manual_rate_override:
    risk_free_rate_input = st.sidebar.number_input("Risk-free rate (decimal)", value=0.02, step=0.001, format="%.4f")
else:
    risk_free_rate_input = None

mu = st.sidebar.slider("Real-world drift μ (annual)", min_value=-0.20, max_value=0.30, value=0.05, step=0.01)
n_paths = st.sidebar.slider("Number of Monte Carlo paths", min_value=1000, max_value=10000, value=5000, step=500)
max_expiries = st.sidebar.slider("Number of expiries to pull", min_value=2, max_value=12, value=6)

load_clicked = st.sidebar.button("Load Data", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Developed from a university derivatives pricing coursework (BSM implied "
    "volatility inversion → SVI local volatility calibration → Monte Carlo → "
    "VaR/ES → delta hedging), extended to live market data."
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for key in ["spot", "rate", "chain", "iv_df", "svi_params", "svi_rmse", "quad_params", "quad_rmse",
            "positions", "real_paths", "rn_paths", "time_grid", "book_df", "initial_book_value",
            "book_greeks_snapshot", "greeks_evolution_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state["positions"] is None:
    st.session_state["positions"] = pd.DataFrame(
        columns=["option_type", "strike", "expiry_label", "expiry_T", "notional", "position"]
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

if load_clicked:
    try:
        with st.spinner(f"Fetching option chain for {ticker}..."):
            spot, rate, chain = cached_load_market_data(ticker, max_expiries)

        if manual_rate_override:
            rate = risk_free_rate_input

        with st.spinner("Computing implied volatilities..."):
            iv_df = cached_implied_vols(chain, spot, rate)

        if iv_df.empty:
            st.error("No valid implied volatilities could be computed from this option chain. Try a more liquid ticker.")
        else:
            with st.spinner("Calibrating SVI local volatility surface..."):
                svi_params, svi_rmse = cached_calibrate_svi(iv_df, spot)
            with st.spinner("Calibrating quadratic benchmark surface..."):
                quad_params, quad_rmse = cached_calibrate_quadratic(iv_df, spot)

            st.session_state.update(
                spot=spot,
                rate=rate,
                chain=chain,
                iv_df=iv_df,
                svi_params=svi_params,
                svi_rmse=svi_rmse,
                quad_params=quad_params,
                quad_rmse=quad_rmse,
                # Reset downstream state when new data is loaded.
                real_paths=None,
                rn_paths=None,
                time_grid=None,
                book_df=None,
                initial_book_value=None,
            )
            st.success(f"Loaded {len(chain)} contracts for {ticker} (spot = ${spot:,.2f}, r = {rate:.2%}).")

    except DataIngestionError as exc:
        st.error(f"Data error: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the user
        st.error(f"Unexpected error while loading data: {exc}")


data_loaded = st.session_state["iv_df"] is not None


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Market Data & Implied Volatility", "Build Your Options Book", "Monte Carlo Simulation", "Risk Analysis", "Greeks"]
)

# --- Tab 1: Market Data & Implied Volatility -------------------------------
with tab1:
    if not data_loaded:
        st.info("Use the sidebar to load a ticker's live option chain.")
    else:
        spot = st.session_state["spot"]
        rate = st.session_state["rate"]
        chain = st.session_state["chain"]
        iv_df = st.session_state["iv_df"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Spot price", f"${spot:,.2f}")
        c2.metric("Risk-free rate (3m T-bill)", f"{rate:.2%}")
        c3.metric("Contracts (post-cleaning)", f"{len(chain)}")

        st.subheader("Raw Option Chain")
        st.dataframe(chain, use_container_width=True, height=250)

        st.subheader("Implied Volatility Smile")
        st.plotly_chart(plot_iv_smile(iv_df), use_container_width=True)

        st.subheader("Implied Volatility Surface (3D)")
        st.plotly_chart(plot_iv_surface_3d(iv_df), use_container_width=True)

        st.subheader("Local Volatility Calibration")
        col1, col2 = st.columns(2)
        col1.metric("SVI calibration RMSE", f"{st.session_state['svi_rmse']:.4f}")
        col2.metric("Quadratic calibration RMSE", f"{st.session_state['quad_rmse']:.4f}")
        if st.session_state["svi_rmse"] <= st.session_state["quad_rmse"]:
            st.caption("SVI provides a tighter fit to the market smile than the quadratic benchmark.")
        else:
            st.caption("The quadratic benchmark fit at least as well as SVI for this surface.")

        st.plotly_chart(
            plot_svi_calibration_fit(iv_df, spot, st.session_state["svi_params"]), use_container_width=True
        )
        st.plotly_chart(plot_local_vol_surface(spot, st.session_state["svi_params"]), use_container_width=True)

# --- Tab 2: Build Your Options Book ----------------------------------------
with tab2:
    if not data_loaded:
        st.info("Load market data first to build a book against real strikes and expiries.")
    else:
        chain = st.session_state["chain"]
        available_expiries = sorted(chain["expiry"].unique())

        st.subheader("Add a Position")
        with st.form("add_position_form", clear_on_submit=True):
            cols = st.columns(5)
            option_type = cols[0].selectbox("Type", ["call", "put"])
            expiry_label = cols[1].selectbox("Expiry", available_expiries)
            expiry_T = float(chain.loc[chain["expiry"] == expiry_label, "T"].iloc[0])

            strikes_for_expiry = sorted(
                chain.loc[(chain["expiry"] == expiry_label) & (chain["option_type"] == option_type), "strike"].unique()
            )
            strike = cols[2].selectbox("Strike", strikes_for_expiry) if strikes_for_expiry else None
            notional = cols[3].number_input("Notional ($m)", min_value=0.01, value=1.0, step=0.5)
            position = cols[4].selectbox("Position", ["long", "short"])

            submitted = st.form_submit_button("Add to Book")
            if submitted and strike is not None:
                new_row = pd.DataFrame(
                    [{
                        "option_type": option_type,
                        "strike": float(strike),
                        "expiry_label": expiry_label,
                        "expiry_T": expiry_T,
                        "notional": float(notional),
                        "position": position,
                    }]
                )
                st.session_state["positions"] = pd.concat(
                    [st.session_state["positions"], new_row], ignore_index=True
                )
                # Adding a position invalidates any previously computed book valuation.
                st.session_state["book_df"] = None
                st.session_state["initial_book_value"] = None
                st.session_state["book_greeks_snapshot"] = None
                st.session_state["greeks_evolution_df"] = None

        st.subheader("Current Book")
        positions = st.session_state["positions"]
        if positions.empty:
            st.info("No positions added yet.")
        else:
            st.dataframe(positions, use_container_width=True)

            remove_idx = st.selectbox(
                "Remove a position by row index", options=["None"] + list(positions.index), index=0
            )
            if remove_idx != "None" and st.button("Remove Selected Position"):
                st.session_state["positions"] = positions.drop(index=int(remove_idx)).reset_index(drop=True)
                st.session_state["book_df"] = None
                st.session_state["initial_book_value"] = None
                st.session_state["book_greeks_snapshot"] = None
                st.session_state["greeks_evolution_df"] = None
                st.rerun()

            if st.button("Value Book (Risk-Neutral Monte Carlo)"):
                spot = st.session_state["spot"]
                rate = st.session_state["rate"]
                params = st.session_state["svi_params"]

                with st.spinner("Pricing book under the calibrated local volatility surface..."):
                    longest_T = float(positions["expiry_T"].max())
                    n_steps = max(int(longest_T * 252), 1)
                    _, _, rn_paths = simulate_paths(
                        S0=spot, mu=rate, r=rate, T=longest_T, n_steps=n_steps,
                        n_paths=n_paths, S0_ref=spot, params=params, seed=42,
                    )

                    book_df = price_book(positions, rn_paths, rate)
                    initial_value = float(book_df["value_m"].sum())

                    def rn_paths_fn(S_start: float):
                        _, _, paths = simulate_paths(
                            S0=S_start, mu=rate, r=rate, T=longest_T, n_steps=n_steps,
                            n_paths=n_paths, S0_ref=spot, params=params, seed=43,
                        )
                        return paths

                    book_delta = compute_book_delta(positions, spot, 0.0, rn_paths_fn, rate)

                st.session_state["book_df"] = book_df
                st.session_state["initial_book_value"] = initial_value
                st.session_state["book_delta"] = book_delta

            if st.session_state["book_df"] is not None:
                st.dataframe(st.session_state["book_df"], use_container_width=True)
                c1, c2 = st.columns(2)
                c1.metric("Initial book value ($m)", f"{st.session_state['initial_book_value']:.4f}")
                c2.metric("Book delta", f"{st.session_state['book_delta']:.4f}")

# --- Tab 3: Monte Carlo Simulation ------------------------------------------
with tab3:
    if not data_loaded:
        st.info("Load market data first.")
    elif st.session_state["positions"].empty:
        st.info("Add at least one position to your book in the previous tab before running the simulation.")
    else:
        st.subheader("Run One-Year Real-World Simulation")
        horizon = st.number_input("Risk horizon (years)", min_value=0.1, max_value=3.0, value=1.0, step=0.25)

        if st.button("Run Monte Carlo Simulation"):
            spot = st.session_state["spot"]
            rate = st.session_state["rate"]
            params = st.session_state["svi_params"]
            n_steps = max(int(horizon * 252), 1)

            with st.spinner(f"Simulating {n_paths:,} paths over {horizon:.2f} years..."):
                time_grid, real_paths, rn_paths = cached_simulate_paths(
                    spot, mu, rate, horizon, n_steps, n_paths, params, 35066393
                )

            st.session_state["real_paths"] = real_paths
            st.session_state["rn_paths"] = rn_paths
            st.session_state["time_grid"] = time_grid
            st.session_state["horizon"] = horizon
            st.success("Simulation complete.")

        if st.session_state["real_paths"] is not None:
            real_paths = st.session_state["real_paths"]
            time_grid = st.session_state["time_grid"]

            st.subheader("Sample Simulated Paths (Real-World Measure)")
            st.plotly_chart(plot_spot_paths(real_paths, time_grid), use_container_width=True)

            st.subheader("Fan Chart")
            st.plotly_chart(plot_fan_chart(real_paths, time_grid), use_container_width=True)

            st.subheader("Terminal Spot Distribution")
            stats = compute_terminal_distribution(real_paths)
            st.dataframe(pd.DataFrame(stats.items(), columns=["Statistic", "Value"]), use_container_width=True)

# --- Tab 4: Risk Analysis ---------------------------------------------------
with tab4:
    if not data_loaded:
        st.info("Load market data first.")
    elif st.session_state["real_paths"] is None or st.session_state["initial_book_value"] is None:
        st.info("Build and value your book, then run the Monte Carlo simulation, before viewing risk analysis.")
    else:
        spot = st.session_state["spot"]
        rate = st.session_state["rate"]
        params = st.session_state["svi_params"]
        positions = st.session_state["positions"]
        real_paths = st.session_state["real_paths"]
        time_grid = st.session_state["time_grid"]
        horizon = st.session_state["horizon"]
        initial_book_value = st.session_state["initial_book_value"]

        if st.button("Run Risk Analysis (Naked + Delta-Hedged)"):
            with st.spinner("Computing naked P&L distribution..."):
                naked_pnl = compute_naked_pnl(
                    positions_df=positions,
                    real_paths=real_paths,
                    horizon=horizon,
                    initial_book_value=initial_book_value,
                    r=rate,
                    params=params,
                )

            with st.spinner("Building delta-hedging grids (this can take a little while)..."):
                terminal_spots = real_paths[-1, :]
                spot_min = min(float(terminal_spots.min()), float(positions["strike"].min())) * 0.9
                spot_max = max(float(terminal_spots.max()), float(positions["strike"].max())) * 1.1

                revaluation_grids = build_horizon_revaluation_grids(
                    positions_df=positions, horizon=horizon, spot_range=(spot_min, spot_max), r=rate, params=params,
                )
                horizon_values = book_value_at_horizon(terminal_spots, positions, horizon, revaluation_grids)

                _, delta_interp = build_hedge_grids(
                    positions_df=positions, time_grid=time_grid, spot_range=(spot_min, spot_max), r=rate, params=params,
                )

            with st.spinner("Simulating daily delta-hedged P&L..."):
                hedged_pnl = run_delta_hedge_simulation(
                    positions_df=positions,
                    real_paths=real_paths,
                    time_grid=time_grid,
                    horizon_book_values=horizon_values,
                    initial_book_value=initial_book_value,
                    delta_interpolator=delta_interp,
                    r=rate,
                )

            st.session_state["naked_pnl"] = naked_pnl
            st.session_state["hedged_pnl"] = hedged_pnl

        if "naked_pnl" in st.session_state and st.session_state.get("naked_pnl") is not None:
            naked_pnl = st.session_state["naked_pnl"]
            hedged_pnl = st.session_state["hedged_pnl"]

            naked_var = compute_var(naked_pnl)
            naked_es = compute_es(naked_pnl)
            hedged_var = compute_var(hedged_pnl)
            hedged_es = compute_es(hedged_pnl)

            st.subheader("Naked P&L Distribution")
            st.plotly_chart(plot_pnl_distribution(naked_pnl, "Naked P&L", var=naked_var, es=naked_es), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("99% VaR ($m)", f"{naked_var:.4f}")
            c2.metric("99% Expected Shortfall ($m)", f"{naked_es:.4f}")

            st.subheader("Delta-Hedged P&L Distribution")
            st.plotly_chart(plot_pnl_distribution(hedged_pnl, "Delta-Hedged P&L", var=hedged_var, es=hedged_es), use_container_width=True)
            c3, c4 = st.columns(2)
            c3.metric("99% VaR ($m)", f"{hedged_var:.4f}")
            c4.metric("99% Expected Shortfall ($m)", f"{hedged_es:.4f}")

            st.subheader("Naked vs Delta-Hedged Comparison")
            st.plotly_chart(plot_naked_vs_hedged(naked_pnl, hedged_pnl, naked_var, hedged_var), use_container_width=True)

            comparison = compare_naked_vs_hedged(naked_pnl, hedged_pnl)
            st.plotly_chart(plot_risk_summary_table(comparison), use_container_width=True)

            st.markdown(
                r"""
                **What these metrics mean:**
                - **VaR (Value-at-Risk)** is a loss threshold: a 99% VaR of \$X means there is only a 1% chance
                  of losing more than \$X over the risk horizon. It does not say how bad losses beyond that
                  threshold could be.
                - **ES (Expected Shortfall)** is the average loss across that worst 1% of scenarios — it
                  captures the *severity* of tail losses, which is why regulators (Basel/FRTB) have moved
                  capital requirements from VaR towards ES: two books can share the same VaR but have very
                  different tail risk, and ES tells the two apart.
                - **Delta hedging** trades the underlying stock daily to offset the book's first-order
                  sensitivity to spot moves. It cannot remove all risk (it leaves gamma, vega, and other
                  higher-order exposures unhedged) but it typically removes the majority of P&L variance for
                  a book of vanilla options, as shown by the reduction percentages above.
                - **Regulatory capital (Basel/FRTB):** market risk capital requirements are calibrated to
                  tail-risk measures like VaR/ES. A well-hedged book that demonstrably reduces VaR/ES can, in
                  principle, support a lower regulatory capital charge than an unhedged book — which is the
                  core economic incentive for desks to hedge in the first place.
                """
            )

# --- Tab 5: Greeks -----------------------------------------------------------
with tab5:
    if not data_loaded:
        st.info("Load market data first.")
    elif st.session_state["positions"].empty:
        st.info("Add at least one position to your book in the 'Build Your Options Book' tab first.")
    else:
        spot = st.session_state["spot"]
        rate = st.session_state["rate"]
        params = st.session_state["svi_params"]
        positions = st.session_state["positions"]

        min_expiry = float(positions["expiry_T"].min())
        max_expiry = float(positions["expiry_T"].max())
        # Floor must never exceed the shortest expiry - e.g. an expiry that's
        # only a day away (T ~ 0.003) would otherwise violate a fixed 0.05
        # floor and crash the widget.
        horizon_floor = min(0.01, min_expiry)

        horizon = st.number_input(
            "Time horizon for Greeks evolution (years)",
            min_value=horizon_floor,
            max_value=max_expiry,
            value=min_expiry,
            step=0.05,
            help=(
                "Greeks are tracked from today up to this point in time, with spot held fixed at "
                "today's level. Defaults to the book's shortest expiry, since a Greek's behaviour "
                "changes discontinuously once a leg actually expires."
            ),
        )

        if st.button("Compute Greeks"):
            with st.spinner("Computing current Greeks snapshot..."):
                snapshot = compute_book_greeks(
                    S=spot, valuation_time=0.0, positions_df=positions, r=rate, params=params
                )

            with st.spinner("Computing Greeks evolution over time (this can take a little while)..."):
                evolution_df = compute_greeks_over_time(
                    positions_df=positions, S0=spot, r=rate, params=params, max_horizon=horizon
                )

            st.session_state["book_greeks_snapshot"] = snapshot
            st.session_state["greeks_evolution_df"] = evolution_df

        if st.session_state["book_greeks_snapshot"] is not None:
            snapshot = st.session_state["book_greeks_snapshot"]

            st.subheader("Current Greeks Snapshot")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Book value ($m)", f"{snapshot['value']:.4f}")
            c2.metric("Delta", f"{snapshot['delta']:.4f}")
            c3.metric("Gamma", f"{snapshot['gamma']:.6f}")
            c4.metric("Vega (per 1pt vol)", f"{snapshot['vega']:.4f}")
            c5.metric("Theta (per day)", f"{snapshot['theta']:.5f}")
            c6.metric("Rho (per 1% rate)", f"{snapshot['rho']:.4f}")

            st.subheader("How the Greeks Evolve Over Time")
            st.plotly_chart(plot_greeks_evolution(st.session_state["greeks_evolution_df"]), use_container_width=True)

            st.markdown(
                r"""
                **What each Greek means:**
                - **Delta** — how much the book's value changes for a \$1 move in the spot price.
                  A delta of +0.5 means the book gains about \$0.50 (per \$1m notional) if the stock rises \$1.
                - **Gamma** — how much Delta itself changes for a \$1 move in spot. High gamma means
                  the hedge ratio needs to be rebalanced more aggressively as the stock moves — it's
                  the main source of the residual hedging error discussed in the Risk Analysis tab.
                - **Vega** — how much the book's value changes if implied volatility rises by 1
                  percentage point (e.g. 20% to 21%). Long option positions are usually long vega
                  (they gain value when volatility rises); short positions are the mirror image.
                - **Theta** — how much value the book loses (or gains) purely from one day passing,
                  with spot and volatility held fixed. This is "time decay" — usually negative for
                  a book that is net long options, since options lose time value as expiry approaches.
                - **Rho** — how much the book's value changes if the risk-free rate rises by 1
                  percentage point. Usually the smallest-impact Greek for short-dated books, but can
                  matter more for longer-dated positions.

                All Greeks are estimated via central bump-and-revalue under the calibrated local
                volatility surface, using common random numbers across each bumped pair so the
                estimate isn't itself contaminated by fresh Monte Carlo noise.
                """
            )
