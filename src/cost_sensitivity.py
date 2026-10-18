
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_backtest import (
    TICKERS,
    BENCHMARK,
    make_signals,
    calculate_metrics,
)

RAW_FILE = ROOT / "data" / "raw" / "adjusted_close_prices.csv"
REPORTS = ROOT / "reports"
CHARTS = ROOT / "charts"
PROCESSED = ROOT / "data" / "processed"

TEST_START = pd.Timestamp("2023-01-01")
COSTS = [0.0, 0.001, 0.002, 0.005]


def apply_costs(prices, weights, cost_rate):
    returns = prices[TICKERS].pct_change().fillna(0.0)

    # Weights are shifted by make_signals, so they are known
    # before the returns they earn.
    gross_return = (weights * returns).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    net_return = gross_return - turnover * cost_rate
    return net_return, turnover


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "Price data not found. Run src/run_backtest.py first."
        )

    prices = pd.read_csv(RAW_FILE, parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices[TICKERS + [BENCHMARK]].apply(
        pd.to_numeric, errors="coerce"
    )
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()

    crossover_w, momentum_w, *_ = make_signals(prices)

    # Fixed chronological holdout; do not tune parameters on this period.
    mask = prices.index >= TEST_START
    test_prices = prices.loc[mask]
    if test_prices.empty:
        raise ValueError("No data found in the holdout period.")

    benchmark_returns = test_prices[BENCHMARK].pct_change().fillna(0.0)
    records = []

    strategies = {
        "MA Crossover": crossover_w.loc[test_prices.index],
        "Momentum": momentum_w.loc[test_prices.index],
    }

    for strategy, weights in strategies.items():
        # Keep the original shifted weights. The first holdout day may
        # carry a position established using pre-holdout information.
        for cost in COSTS:
            net_returns, turnover = apply_costs(
                test_prices, weights, cost
            )
            equity = (1 + net_returns).cumprod()
            metrics = calculate_metrics(net_returns, equity)

            records.append({
                "Strategy": strategy,
                "Transaction Cost (%)": cost * 100,
                **metrics,
                "Average Daily Turnover": turnover.mean(),
            })

    # Benchmark has no portfolio rebalancing costs in this model.
    benchmark_equity = (1 + benchmark_returns).cumprod()
    records.append({
        "Strategy": "SPY Buy & Hold",
        "Transaction Cost (%)": 0.0,
        **calculate_metrics(benchmark_returns, benchmark_equity),
        "Average Daily Turnover": 0.0,
    })

    results = pd.DataFrame(records)
    REPORTS.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(exist_ok=True)

    results.to_csv(
        REPORTS / "cost_sensitivity_summary.csv", index=False
    )

    chart_data = results[results["Strategy"] != "SPY Buy & Hold"]
    fig = px.line(
        chart_data,
        x="Transaction Cost (%)",
        y="CAGR",
        color="Strategy",
        markers=True,
        title="Out-of-Sample CAGR vs. Transaction Costs",
        hover_data={
            "Transaction Cost (%)": ":.2f",
            "CAGR": ":.2%",
            "Sharpe Ratio": ":.3f",
            "Maximum Drawdown": ":.2%",
        },
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_title="Transaction cost (% of turnover)",
        yaxis_title="CAGR",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    fig.write_html(
        str(CHARTS / "06_cost_sensitivity.html"),
        include_plotlyjs=True,
    )

    print("\n=== TRANSACTION-COST SENSITIVITY ===")
    print(
        results[
            [
                "Strategy",
                "Transaction Cost (%)",
                "CAGR",
                "Sharpe Ratio",
                "Maximum Drawdown",
                "Calmar Ratio",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print("\nSaved report:", REPORTS / "cost_sensitivity_summary.csv")
    print("Saved chart:", CHARTS / "06_cost_sensitivity.html")


if __name__ == "__main__":
    main()
