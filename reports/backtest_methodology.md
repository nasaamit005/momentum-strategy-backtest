# Momentum Strategy Backtesting — Methodology

## 1. Project objective

This project evaluates systematic equity strategies using historical adjusted closing prices. It compares a moving-average crossover strategy and a momentum strategy against a buy-and-hold SPY benchmark.

The project is an educational quantitative-finance research exercise. Historical performance does not guarantee future results.

## 2. Strategy configuration

- Assets: AAPL, MSFT, AMZN, GOOGL, NVDA, META
- Benchmark: SPY
- Historical start date: 2015-01-01
- Annualization convention: 252 trading days
- Moving-average windows: 50 and 200 trading days
- Momentum lookback: 126 trading days
- Momentum portfolio: Top 3 eligible assets
- Configured transaction cost: 0.001 per unit of modeled turnover
- Out-of-sample evaluation begins in 2023, subject to the dates recorded in the generated reports.

## 3. Daily asset returns

For an adjusted closing price P_t:

r_t = (P_t / P_(t-1)) - 1

Daily returns measure the percentage price change between consecutive observations. Adjusted prices help account for distributions and stock splits when supplied correctly by the data provider.

## 4. Moving-average crossover strategy

The short moving average is:

SMA_50(t) = mean(P_(t-49), ..., P_t)

The long moving average is:

SMA_200(t) = mean(P_(t-199), ..., P_t)

The strategy uses the relationship between the short and long moving averages to determine its exposure. The precise signal and position rules are implemented in src/run_backtest.py.

## 5. Momentum strategy

For a 126-trading-day lookback, a simple price momentum measure is:

Momentum_126(t) = P_t / P_(t-126) - 1

Eligible assets are ranked using the implemented momentum measure, and the configured top three are selected. Portfolio weights and position timing follow the implementation in src/run_backtest.py.

## 6. Portfolio returns and transaction costs

The portfolio return is calculated from the asset returns and portfolio weights used by the backtest:

r_portfolio(t) = sum_i [w_i(t-1) * r_i(t)]

The one-period weight lag is intended to prevent using a position before it is established. Confirm the precise signal timing in the source code before interpreting this as a fully verified no-look-ahead implementation.

The configured transaction-cost model is based on changes in portfolio weights:

cost(t) = transaction_cost * sum_i |w_i(t) - w_i(t-1)|

Net portfolio return is the gross portfolio return less the modeled transaction cost, according to the implementation.

This turnover convention uses the sum of absolute weight changes. It should not automatically be interpreted as one-way traded notional or multiplied by an additional factor without checking the execution model.

## 7. Equity curve and total return

Equity represents the compounded portfolio value:

E_t = E_(t-1) * (1 + r_net(t))

Total return over the evaluated period is:

Total Return = E_T / E_0 - 1

where E_0 is the first recorded equity value and E_T is the last recorded equity value.

## 8. Compound annual growth rate (CAGR)

Using N daily return observations and 252 trading days per year:

CAGR = (E_T / E_0)^(252/N) - 1

This formula uses both the starting and ending equity values. It avoids assuming that every stored equity series begins at exactly 1.0.

## 9. Annualized volatility

Sample standard deviation of daily returns is annualized as:

Annualized Volatility = SD(r_daily) * sqrt(252)

Volatility measures variability of returns; it does not measure the probability of loss by itself.

## 10. Sharpe ratio

Using a zero daily risk-free/target return convention:

Sharpe = mean(r_daily) / SD(r_daily) * sqrt(252)

This is a zero-risk-free-rate approximation. It is not an excess-return Sharpe ratio using a time-varying risk-free series.

## 11. Downside deviation and Sortino ratio

Negative daily returns relative to a zero target are used to estimate downside deviation:

Downside Deviation = sqrt(mean(min(r_daily, 0)^2)) * sqrt(252)

Sortino ratio is calculated as:

Sortino = mean(r_daily) * 252 / Downside Deviation

This implementation uses the zero-return target and the downside-deviation convention documented above. Other definitions can produce different values.

## 12. Maximum drawdown

Running peak equity is:

Peak_t = max(E_1, ..., E_t)

Drawdown is:

Drawdown_t = E_t / Peak_t - 1

Maximum Drawdown = min(Drawdown_t)

Maximum drawdown measures the largest peak-to-trough decline within the evaluated series.

## 13. Calmar ratio

Calmar ratio is calculated as:

Calmar = CAGR / abs(Maximum Drawdown)

It is undefined when maximum drawdown is zero or when CAGR is unavailable.

## 14. Out-of-sample evaluation

The out-of-sample period is evaluated separately from the full historical period. The exact dates and observation counts in the generated CSV reports should be treated as authoritative.

An out-of-sample period is useful for checking performance on data not used for initial strategy development. It does not eliminate selection bias, repeated experimentation, survivorship bias, or sensitivity to market conditions.

Avoid changing strategy parameters after inspecting out-of-sample results and then continuing to describe the same period as untouched test data.

## 15. Data validation

The risk analytics script checks:

- Date validity and duplicate dates.
- Required strategy columns.
- Numeric, finite return and equity values.
- Date alignment between returns and equity curves.
- Whether normalized equity growth matches compounded daily returns.

A successful equity-path validation checks mathematical consistency between the saved returns and equity curves. It does not independently prove that the strategy is profitable, correctly timed, or free from look-ahead bias.

## 16. Limitations and risks

- Backtests are simulations, not live trading results.
- Historical returns are not a forecast of future returns.
- Transaction costs are simplified and may omit bid-ask spreads, market impact, taxes, slippage, and liquidity constraints.
- Portfolio rebalancing and signal timing must be checked against the source code.
- The 252-day annualization convention is an approximation.
- Sharpe and Sortino ratios depend on the selected target-return convention.
- Results may be sensitive to asset selection, lookback windows, market regimes, and the selected sample period.
- Corporate-action adjustments and missing observations depend on the source data.
- Strategy selection after repeated testing can introduce overfitting.

## 17. Reproducibility

Key implementation files:

- src/run_backtest.py
- src/validate_out_of_sample.py
- src/create_oos_report.py
- src/risk_analytics_dashboard.py

Key generated outputs:

- reports/performance_summary.csv
- reports/out_of_sample_summary.csv
- reports/risk_analytics_summary.csv
- reports/risk_data_validation.csv
- reports/risk_analytics_dashboard.html

Re-run the relevant scripts after changing code. Confirm that report dates, observation counts, formulas, and equity paths agree before publishing results.

## 18. Interpretation

Compare strategies using total return, CAGR, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and Calmar ratio together. No single metric captures every aspect of investment risk.

The strongest historical result should not automatically be considered the best deployable strategy. Further testing should include realistic trading costs, parameter sensitivity, walk-forward evaluation, and paper trading.
