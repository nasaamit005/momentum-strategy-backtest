
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_backtest import (
    TICKERS,
    BENCHMARK,
    make_signals,
    run_strategy,
    calculate_metrics,
)

RAW_FILE = ROOT / "data" / "raw" / "adjusted_close_prices.csv"
REPORTS = ROOT / "reports"
CHARTS = ROOT / "charts"

# Keep this cutoff fixed for the initial holdout evaluation.
TEST_START = pd.Timestamp("2023-01-01")


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Price file not found: {RAW_FILE}\n"
            "Run src/run_backtest.py first."
        )

    prices = pd.read_csv(RAW_FILE, parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    required = TICKERS + [BENCHMARK]
    missing = [ticker for ticker in required if ticker not in prices.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    prices = prices[required].apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)

    # Do not fill missing stock prices indefinitely.
    # Keep dates where all required prices are available.
    prices = prices.dropna(subset=required)

    if prices.index.max() <= TEST_START:
        raise ValueError(
            "There are no observations after the test start date."
        )

    # Generate signals using historical data, preserving warm-up history.
    # make_signals shifts weights by one day to avoid same-day look-ahead.
    crossover_w, momentum_w, *_ = make_signals(prices)

    crossover = run_strategy(prices, crossover_w)
    momentum = run_strategy(prices, momentum_w)

    # Calculate daily returns for the benchmark.
    benchmark_returns = prices[BENCHMARK].pct_change().fillna(0.0)

    all_returns = pd.DataFrame({
        "MA Crossover": crossover["net_return"],
        "Momentum": momentum["net_return"],
        "SPY Buy & Hold": benchmark_returns,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    # Only evaluate returns dated on or after the fixed holdout cutoff.
    test_returns = all_returns.loc[all_returns.index >= TEST_START].copy()

    if len(test_returns) < 252:
        print(
            f"Warning: holdout contains only {len(test_returns)} "
            "trading observations; interpret annualized metrics cautiously."
        )

    # Normalize each strategy to $1 at the start of the holdout period.
    test_equity = (1.0 + test_returns).cumprod()

    metric_rows = []
    for name in test_returns.columns:
        metric_rows.append({
            "Strategy": name,
            "Test Start": str(test_returns.index.min().date()),
            "Test End": str(test_returns.index.max().date()),
            "Observations": len(test_returns),
            **calculate_metrics(test_returns[name], test_equity[name]),
        })

    metrics = pd.DataFrame(metric_rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        REPORTS / "out_of_sample_summary.csv", index=False
    )
    test_returns.to_csv(
        ROOT / "data" / "processed" / "out_of_sample_returns.csv",
        index_label="Date",
    )
    test_equity.to_csv(
        ROOT / "data" / "processed" / "out_of_sample_equity.csv",
        index_label="Date",
    )

    fig = go.Figure()
    for name in test_equity.columns:
        fig.add_trace(go.Scatter(
            x=test_equity.index,
            y=test_equity[name],
            mode="lines",
            name=name,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Value of $1: $%{y:.3f}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.update_layout(
        title="Out-of-Sample Strategy Comparison (2023 Onward)",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(
        str(CHARTS / "05_out_of_sample_comparison.html"),
        include_plotlyjs=True,
    )

    print("\n=== OUT-OF-SAMPLE RESULTS ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    print(REPORTS / "out_of_sample_summary.csv")
    print(CHARTS / "05_out_of_sample_comparison.html")

    print(
        "\nInterpretation note: this is a chronological holdout evaluation, "
        "not proof of future profitability. The strategies were specified "
        "before this evaluation, and the holdout should not be used for "
        "repeated parameter tuning."
    )


if __name__ == "__main__":
    main()
