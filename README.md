# Options Pricing & Risk Management Tool

An interactive Streamlit application for pricing and risk-managing a book of
European vanilla options against **live market data**, built around a
volatility-surface-calibrated Monte Carlo pricing engine. The user picks any
real, listed stock ticker, the app pulls its live option chain, backs out
implied volatilities, calibrates a local volatility surface (SVI), and then
lets the user construct a custom options book and stress-test it with
Monte Carlo-based Value-at-Risk, Expected Shortfall, and delta-hedging
analysis.

## Live Demo
*https://nakul-options-pricer.streamlit.app/*

## What the Tool Does

- Pulls a live option chain for any ticker (e.g. `AAPL`, `TSLA`, `SPY`) via `yfinance`
- Cleans the chain (removes illiquid/zero-quote contracts) and computes mid-prices
- Backs out implied volatility for every contract using Black-Scholes-Merton inversion (Brent's method)
- Calibrates both a quadratic benchmark and an SVI-inspired local volatility surface to the market smile
- Lets the user build a custom book of long/short call/put positions across real strikes and expiries
- Prices the book via risk-neutral Monte Carlo simulation under the calibrated local volatility surface
- Simulates real-world spot price scenarios and computes the book's unhedged P&L distribution
- Computes 99% Value-at-Risk (VaR) and Expected Shortfall (ES)
- Simulates a daily-rebalanced delta-hedging strategy and quantifies its risk reduction
- Computes the book's Greeks (Delta, Gamma, Vega, Theta, Rho) via bump-and-revalue and tracks how they evolve over time as expiry approaches
- Presents every step through interactive Plotly charts in a multi-tab Streamlit dashboard

## The Mathematical Pipeline, in Plain English

1. **Implied volatility.** Every option's market price embeds the market's
   own view of future volatility. We invert the Black-Scholes formula
   (using a numerical root-finder, Brent's method) to recover that implied
   volatility for every contract in the chain.

2. **Volatility surface calibration.** Implied volatility isn't flat — it
   varies by strike (skew) and by maturity (term structure). We fit a
   smooth parametric surface, **SVI** (Stochastic Volatility Inspired) — the
   parameterisation real trading desks use — through the scattered market
   points, so we have a usable volatility number everywhere on the
   (time, spot) plane, not just at the strikes that happened to be quoted.
   A simpler quadratic model is fit alongside it as a benchmark.

3. **Monte Carlo simulation.** We simulate thousands of possible future
   paths for the stock price. Crucially, two different simulations are run
   for two different jobs: a **real-world** simulation (using the stock's
   actual expected return) for risk scenarios, and a **risk-neutral**
   simulation (using the risk-free rate) for pricing — mixing these up is a
   classic and serious pricing error.

4. **Book valuation.** Each option in the user's book is priced as a
   discounted average of its simulated payoffs under the risk-neutral
   measure. Long positions add value; short positions subtract it.

5. **Risk metrics.** The book is revalued across thousands of real-world
   future scenarios to build a P&L distribution. From that distribution we
   compute **VaR** (the loss exceeded only 1% of the time) and **Expected
   Shortfall** (the average loss in that worst 1% — the metric regulators
   under Basel/FRTB increasingly prefer, since it captures tail severity,
   not just a threshold).

6. **Delta hedging.** The book's sensitivity to the underlying ("delta") is
   estimated daily via bump-and-revalue, and an offsetting stock position
   is taken and rebalanced every day. The tool quantifies exactly how much
   this reduces P&L volatility, VaR, and ES.

7. **Greeks.** Beyond delta, the book's full risk sensitivities — Gamma,
   Vega, Theta, and Rho — are estimated the same way, via central
   bump-and-revalue (nudging spot, volatility, time, and the risk-free
   rate up and down and repricing) under the calibrated local volatility
   surface, using common random numbers so each bumped pair isn't
   contaminated by fresh Monte Carlo noise. The tool also tracks how these
   Greeks evolve purely from time passing, with spot held fixed, up to the
   book's shortest expiry.

## How to Install and Run Locally

```bash
git clone <your-repo-url>
cd options-pricing-tool
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Project Structure

```
options-pricing-tool/
├── app.py                  # Streamlit application entry point
├── src/
│   ├── data_ingestion.py   # Live option chain retrieval & cleaning (yfinance)
│   ├── implied_vol.py      # BSM pricing + implied volatility inversion
│   ├── vol_surface.py      # SVI / quadratic local volatility calibration
│   ├── monte_carlo.py      # Real-world & risk-neutral path simulation
│   ├── pricing.py          # Option/book Monte Carlo valuation & delta
│   ├── risk.py             # VaR, Expected Shortfall, delta-hedge simulation
│   ├── greeks.py           # Delta/Gamma/Vega/Theta/Rho via bump-and-revalue, over time
│   └── plotting.py         # All Plotly chart functions
├── requirements.txt
└── README.md
```

## Screenshots

*(Add screenshots here once you've run the app — e.g. the implied vol
surface, the SVI calibration fit, the fan chart, and the naked-vs-hedged
P&L comparison.)*

## Technologies Used

- **Python** — numpy, scipy (Brent root-finding, L-BFGS-B optimisation, interpolation), pandas
- **Streamlit** — interactive web app framework
- **Plotly** — interactive charting
- **yfinance** — live market data (option chains, spot prices, Treasury yields)

## Background

This project was developed from a university derivatives pricing coursework
(implied volatility surface construction, SVI local volatility calibration,
Monte Carlo simulation, VaR/Expected Shortfall, and delta-hedging analysis
on a fictional stock with a static dataset) and extended into a real-world,
interactive tool that runs the same pipeline against **live market data**
for any listed ticker.
