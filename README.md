# Momentum Strategy Backtesting

A Python-based quantitative finance project for evaluating momentum trading strategies through historical backtesting, performance analysis, risk analytics, out-of-sample validation, transaction-cost sensitivity, and interactive Plotly dashboards.

**GitHub Repository:** [momentum-strategy-backtest](https://github.com/nasaamit005/momentum-strategy-backtest)

## Project Objectives

- Evaluate momentum strategy performance using historical market data.
- Analyze portfolio growth, returns, volatility, and drawdowns.
- Compare in-sample and out-of-sample performance.
- Investigate the impact of transaction costs on strategy results.
- Build interactive charts and analytical dashboards.
- Develop a reproducible Python workflow with automated tests.

## Features

- Historical strategy backtesting
- Equity curve visualization
- Drawdown and rolling-volatility analysis
- Performance metrics comparison
- Out-of-sample evaluation
- Transaction-cost sensitivity analysis
- Interactive Plotly HTML charts
- CSV-based performance and risk reports
- Automated testing using pytest

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backtesting and analysis |
| pandas | Data processing |
| NumPy | Numerical calculations |
| Plotly | Interactive visualizations |
| pytest | Automated testing |
| Git and GitHub | Version control |

## Project Structure

```text
momentum-strategy-backtest/
├── charts/
│   ├── 01_equity_curves.html
│   ├── 02_drawdowns.html
│   ├── 03_rolling_volatility.html
│   ├── 04_metrics_comparison.html
│   ├── 05_out_of_sample_comparison.html
│   ├── 06_cost_sensitivity.html
│   └── moving_average_*.html
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── backtest_methodology.md
│   ├── backtest_report.md
│   ├── performance_summary.csv
│   ├── risk_analytics_summary.csv
│   ├── cost_sensitivity_summary.csv
│   ├── out_of_sample_summary.csv
│   ├── strategy_performance_dashboard.html
│   ├── risk_analytics_dashboard.html
│   └── out_of_sample_dashboard.html
├── src/
│   ├── run_backtest.py
│   ├── cost_sensitivity.py
│   ├── validate_out_of_sample.py
│   ├── create_oos_report.py
│   ├── interactive_cost_table.py
│   ├── risk_analytics_dashboard.py
│   └── strategy_performance_dashboard.py
├── tests/
│   └── test_backtest.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ## Key Financial Formulas

### 1. Simple Return

The simple return measures the percentage change in an asset's price between two consecutive periods.

\[
R_t = \frac{P_t-P_{t-1}}{P_{t-1}}
\]

### 2. Cumulative Return

Cumulative return measures the total portfolio return over the entire investment period.

\[
R_{\mathrm{cum}} = \frac{V_T}{V_0}-1
\]

### 3. Annualized Volatility

Annualized volatility estimates the yearly variability of returns using daily return volatility.

\[
\sigma_{\mathrm{annual}} = \sigma_{\mathrm{daily}}\sqrt{252}
\]

### 4. Sharpe Ratio

The Sharpe ratio measures risk-adjusted performance by comparing excess annual return with annualized volatility.

\[
S = \frac{R_{\mathrm{annual}}-R_f}{\sigma_{\mathrm{annual}}}
\]

### 5. Maximum Drawdown

Drawdown measures the decline in portfolio value relative to its previous running peak.

\[
DD_t = \frac{V_t}{\displaystyle\max_{s\leq t}V_s}-1
\]

Maximum drawdown is the largest peak-to-trough decline, expressed here as a negative return.

\[
MDD = \min_t DD_t
\]

### 6. Compound Annual Growth Rate (CAGR)

CAGR estimates the constant annual growth rate that would produce the observed change in portfolio value over the investment period.

\[
CAGR = \left(\frac{V_T}{V_0}\right)^{\frac{1}{Y}}-1
\]

### Definitions of Variables

- \(R_t\): Asset return at time \(t\).
- \(P_t\): Asset price at time \(t\).
- \(P_{t-1}\): Asset price in the previous period.
- \(R_{\mathrm{cum}}\): Cumulative return over the investment period.
- \(V_0\): Initial portfolio value.
- \(V_T\): Final portfolio value.
- \(\sigma_{\mathrm{daily}}\): Standard deviation of daily returns.
- \(\sigma_{\mathrm{annual}}\): Annualized standard deviation of returns.
- \(R_{\mathrm{annual}}\): Annualized portfolio return, using the convention implemented in the backtest.
- \(R_f\): Annual risk-free rate.
- \(DD_t\): Drawdown at time \(t\).
- \(MDD\): Maximum drawdown.
- \(Y\): Investment period in years.
- \(252\): Assumed number of trading days per year.

 These are standard financial formulas. The exact return calculation, annualization convention, risk-free rate assumption, and treatment of transaction costs should be verified against the project's source code and methodology report.

## Interactive Dashboards

The project contains interactive HTML visualizations for:

| Chart or Dashboard | Purpose |
|---|---|
| Equity curves | Evaluate portfolio growth |
| Drawdowns | Understand downside risk |
| Rolling volatility | Examine changing market risk |
| Metrics comparison | Compare reported performance metrics |
| Out-of-sample comparison | Evaluate performance on held-out data |
| Cost sensitivity | Analyze the impact of trading costs |
| Strategy performance dashboard | Explore performance results |
| Risk analytics dashboard | Explore risk measures |

To open a chart, navigate to the `charts/` or `reports/` directory and open the relevant `.html` file in your web browser.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nasaamit005/momentum-strategy-backtest.git
cd momentum-strategy-backtest
```

### 2. Create a virtual environment

For Windows with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the Project

Run the backtesting workflow:

```powershell
python .\src\run_backtest.py
```

Generate risk analytics:

```powershell
python .\src\risk_analytics_dashboard.py
```

Generate the performance dashboard:

```powershell
python .\src\strategy_performance_dashboard.py
```

Run automated tests:

```powershell
python -m pytest -v
```

Run scripts in the appropriate order if they depend on outputs generated by earlier steps.

## Testing

The project includes automated tests in `tests/test_backtest.py`.

A previous test run completed successfully with **6 passed**. Run the test suite again after making code changes to verify the current version.

## Out-of-Sample Validation

Out-of-sample testing evaluates a strategy on data not used for the relevant strategy selection or parameter tuning.

The objective is to investigate whether the observed historical behavior persists beyond the data used during development. Correct separation of training and evaluation periods is important to reduce look-ahead bias and overfitting.

See `src/validate_out_of_sample.py` and the reports directory for the implementation and results.

## Transaction-Cost Analysis

Transaction-cost sensitivity analysis evaluates how different assumed trading costs affect backtest performance.

It helps assess whether apparent strategy returns remain attractive after estimated execution costs. Results depend on assumptions about turnover, commissions, slippage, and market impact.

See `src/cost_sensitivity.py` and `reports/cost_sensitivity_summary.csv`.

## Limitations

- Historical results do not guarantee future performance.
- Results depend on data quality, strategy parameters, and execution assumptions.
- Transaction costs and slippage estimates may not capture all real-world trading expenses.
- Out-of-sample validation does not eliminate all overfitting risk.
- This project is for educational and analytical purposes, not investment advice.

## Future Improvements

- Add walk-forward validation.
- Expand transaction-cost and slippage scenarios.
- Add GitHub Actions for automated testing.
- Improve data validation and reproducibility.
- Publish dashboard previews and a detailed findings report.

## Author

**Amit Singh**

GitHub: [@nasaamit005](https://github.com/nasaamit005)

## License

Distributed under the MIT License. See the `LICENSE` file for details.

---

*Portfolio project demonstrating Python-based quantitative analysis, backtesting, risk measurement, and interactive financial data visualization.*