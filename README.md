# Options Pricing & Risk Management Tool

An interactive Streamlit application for pricing and risk-managing a book of European vanilla options against live market data, built around a volatility-surface-calibrated Monte Carlo pricing engine. The user picks any real, listed stock ticker, the app pulls its live option chain, backs out implied volatilities, calibrates a local volatility surface (SVI), and lets the user construct a custom options book and stress-test it with Monte Carlo-based Value-at-Risk, Expected Shortfall, and delta-hedging analysis.

Built as a portfolio project for Quantitative Analyst / Financial Technology roles, extending a university derivatives-pricing coursework (implied volatility surface construction, SVI calibration, Monte Carlo VaR/Expected Shortfall, and delta-hedging on a static dataset) into a live-market, interactive tool.

## Live Demo

[https://nakul-options-pricer.streamlit.app/](https://nakul-options-pricer.streamlit.app/)

## Key Features

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

## Methodology

1. Implied volatility: every option's market price embeds the market's own view of future volatility. The Black-Scholes formula is inverted (using a numerical root-finder, Brent's method) to recover implied volatility for every contract in the chain.
2. Volatility surface calibration: implied volatility varies by strike (skew) and by maturity (term structure) rather than staying flat. A smooth parametric surface, SVI (Stochastic Volatility Inspired), the parameterisation used by real trading desks, is fit through the scattered market points, giving a usable volatility number everywhere on the (time, spot) plane rather than only at the strikes that happened to be quoted. A simpler quadratic model is fit alongside it as a benchmark.
3. Monte Carlo simulation: thousands of possible future paths for the stock price are simulated. Two different simulations are run for two different jobs: a real-world simulation (using the stock's actual expected return) for risk scenarios, and a risk-neutral simulation (using the risk-free rate) for pricing. Mixing these two up is a common and serious pricing error.
4. Book valuation: each option in the user's book is priced as a discounted average of its simulated payoffs under the risk-neutral measure. Long positions add value; short positions subtract it.
5. Risk metrics: the book is revalued across thousands of real-world future scenarios to build a P&L distribution. From that distribution, VaR (the loss exceeded only 1% of the time) and Expected Shortfall (the average loss in that worst 1%, the metric regulators under Basel/FRTB increasingly prefer since it captures tail severity rather than just a threshold) are computed.
6. Delta hedging: the book's sensitivity to the underlying ("delta") is estimated daily via bump-and-revalue, and an offsetting stock position is taken and rebalanced every day. The tool quantifies exactly how much this reduces P&L volatility, VaR, and ES.
7. Greeks: beyond delta, the book's full risk sensitivities (Gamma, Vega, Theta, and Rho) are estimated the same way, via central bump-and-revalue (nudging spot, volatility, time, and the risk-free rate up and down and repricing) under the calibrated local volatility surface, using common random numbers so each bumped pair isn't affected by fresh Monte Carlo noise. The tool also tracks how these Greeks evolve purely from time passing, with spot held fixed, up to the book's shortest expiry.

## Repo Structure

```
options-pricer/
├── app.py                  # Streamlit application entry point
├── requirements.txt
└── src/
    ├── data_ingestion.py   # Live option chain retrieval and cleaning (yfinance)
    ├── implied_vol.py      # BSM pricing and implied volatility inversion
    ├── vol_surface.py      # SVI / quadratic local volatility calibration
    ├── monte_carlo.py      # Real-world and risk-neutral path simulation
    ├── pricing.py          # Option/book Monte Carlo valuation and delta
    ├── risk.py             # VaR, Expected Shortfall, delta-hedge simulation
    ├── greeks.py           # Delta/Gamma/Vega/Theta/Rho via bump-and-revalue, over time
    └── plotting.py         # All Plotly chart functions
```

## Installation & Running Locally

```bash
git clone https://github.com/NakulPillai2407/options-pricer.git
cd options-pricer
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Tech Stack

- Python: numpy, scipy (Brent root-finding, L-BFGS-B optimisation, interpolation), pandas
- Streamlit: interactive web app framework
- Plotly: interactive charting
- [yfinance](https://github.com/ranaroussi/yfinance): live market data (option chains, spot prices, Treasury yields)

## Limitations

Pricing and risk outputs depend on the quality of the volatility surface calibration, which can be unstable for illiquid tickers with sparse option chains. Monte Carlo estimates carry simulation error that only shrinks with more simulated paths.

This is an educational tool built for portfolio purposes. It is not financial advice and should not be used for live trading or risk decisions.

## Author

**Nakul Pillai**
BSc Economics & Data Science, University of Southampton · Incoming MSc Financial Technology, Imperial College London

- LinkedIn: [linkedin.com/in/nakul-pillai](https://www.linkedin.com/in/nakul-pillai)
- GitHub: [@NakulPillai2407](https://github.com/NakulPillai2407)
