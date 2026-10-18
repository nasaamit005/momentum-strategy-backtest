from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CHARTS = ROOT / "charts"
REPORTS = ROOT / "reports"

for folder in [RAW, PROCESSED, CHARTS, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META"]
BENCHMARK = "SPY"
START_DATE = "2015-01-01"
SHORT_WINDOW = 50
LONG_WINDOW = 200
MOMENTUM_LOOKBACK = 126
TOP_N = 3
TRANSACTION_COST = 0.001
TRADING_DAYS = 252


def download_prices():
    tickers = TICKERS + [BENCHMARK]
    print("Downloading historical market data...")

    raw = yf.download(
        tickers,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True
    )

    if raw.empty:
        raise RuntimeError(
            "No market data downloaded. Check your internet connection "
            "and try again."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers[:1]

    close = close.reindex(columns=tickers)
    close = close.sort_index().ffill()
    close.index = pd.to_datetime(close.index)
    close = close.dropna(subset=[BENCHMARK])

    close.to_csv(RAW / "adjusted_close_prices.csv", index_label="Date")
    print(f"Saved prices: {close.shape[0]} dates, {close.shape[1]} tickers")
    return close


def make_signals(prices):
    assets = prices[TICKERS].copy()

    # Strategy 1: 50-day / 200-day moving-average crossover.
    short_ma = assets.rolling(SHORT_WINDOW).mean()
    long_ma = assets.rolling(LONG_WINDOW).mean()
    crossover = (short_ma > long_ma).astype(float)
    crossover = crossover.where(long_ma.notna(), 0.0)

    # Strategy 2: Rank stocks by trailing 126-trading-day momentum.
    momentum = assets.pct_change(MOMENTUM_LOOKBACK)
    ranks = momentum.rank(axis=1, ascending=False, method="first")
    momentum_signal = (
        (ranks <= TOP_N) & (momentum > 0)
    ).astype(float)

    # Shift weights by one trading day to avoid using today's close
    # to earn today's return.
    crossover_weights = crossover.shift(1).fillna(0.0)
    momentum_weights = momentum_signal.shift(1).fillna(0.0)

    # Equal weight across the selected stocks.
    crossover_weights = crossover_weights.div(
        crossover_weights.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0.0)

    momentum_weights = momentum_weights.div(
        momentum_weights.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0.0)

    return crossover_weights, momentum_weights, short_ma, long_ma, momentum


def run_strategy(prices, weights):
    returns = prices[TICKERS].pct_change().fillna(0.0)

    # Portfolio return based on previous-close-known target weights.
    gross_return = (weights * returns).sum(axis=1)

    # Trading cost based on changes in target weights.
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    costs = turnover * TRANSACTION_COST
    net_return = gross_return - costs
    equity = (1.0 + net_return).cumprod()

    return pd.DataFrame({
        "gross_return": gross_return,
        "turnover": turnover,
        "transaction_cost": costs,
        "net_return": net_return,
        "equity": equity
    }, index=prices.index)


def calculate_metrics(returns, equity):
    returns = returns.dropna()
    equity = equity.dropna()

    if len(returns) < 2 or equity.empty:
        return {
            "Total Return": np.nan,
            "CAGR": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Maximum Drawdown": np.nan,
            "Calmar Ratio": np.nan
        }

    years = len(returns) / TRADING_DAYS
    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    cagr = (
    (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    if years > 0
    and equity.iloc[0] > 0
    and equity.iloc[-1] > 0
    else np.nan
)

    volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if returns.std(ddof=1) > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()
    calmar = (
        cagr / abs(max_drawdown)
        if pd.notna(cagr) and max_drawdown < 0
        else np.nan
    )

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown,
        "Calmar Ratio": calmar
    }


def save_chart(fig, filename):
    path = CHARTS / filename
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Chart saved: {path}")


def main():
    prices = download_prices()

    crossover_w, momentum_w, short_ma, long_ma, momentum = make_signals(
        prices
    )

    crossover = run_strategy(prices, crossover_w)
    momentum_result = run_strategy(prices, momentum_w)

    # Buy-and-hold benchmark.
    benchmark_returns = prices[BENCHMARK].pct_change().fillna(0.0)
    benchmark_equity = (1 + benchmark_returns).cumprod()

    results = pd.DataFrame({
        "MA Crossover": crossover["net_return"],
        "Momentum": momentum_result["net_return"],
        "SPY Buy & Hold": benchmark_returns
    }).fillna(0.0)

    equity = (1 + results).cumprod()
    equity.to_csv(PROCESSED / "equity_curves.csv", index_label="Date")
    results.to_csv(PROCESSED / "daily_returns.csv", index_label="Date")

    metric_rows = []
    for name in results.columns:
        metric_rows.append({
            "Strategy": name,
            **calculate_metrics(results[name], equity[name])
        })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORTS / "performance_summary.csv", index=False)

    # Daily strategy holdings and trading costs.
    holdings = pd.concat({
        "MA_Crossover": crossover_w,
        "Momentum": momentum_w
    }, axis=1)
    holdings.to_csv(PROCESSED / "daily_positions.csv", index_label="Date")

    details = pd.DataFrame({
        "MA_Crossover_Return": crossover["net_return"],
        "MA_Crossover_Turnover": crossover["turnover"],
        "MA_Crossover_Cost": crossover["transaction_cost"],
        "Momentum_Return": momentum_result["net_return"],
        "Momentum_Turnover": momentum_result["turnover"],
        "Momentum_Cost": momentum_result["transaction_cost"],
        "SPY_Return": benchmark_returns
    })
    details.to_csv(PROCESSED / "backtest_details.csv", index_label="Date")

    # Interactive chart 1: Growth of $1.
    fig = go.Figure()
    for col in equity.columns:
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity[col],
            mode="lines",
            name=col,
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio value: $%{y:.2f}<extra>%{fullData.name}</extra>"
        ))
    fig.update_layout(
        title="Momentum Strategy Backtest — Growth of $1",
        xaxis_title="Date",
        yaxis_title="Portfolio value ($)",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Strategy"
    )
    save_chart(fig, "01_equity_curves.html")

    # Interactive chart 2: Drawdowns.
    drawdown_fig = go.Figure()
    for col in equity.columns:
        dd = equity[col] / equity[col].cummax() - 1
        drawdown_fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd * 100,
            mode="lines",
            name=col,
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra>%{fullData.name}</extra>"
        ))
    drawdown_fig.update_layout(
        title="Portfolio Drawdowns",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        hovermode="x unified"
    )
    save_chart(drawdown_fig, "02_drawdowns.html")

    # Interactive chart 3: Rolling annualized volatility.
    vol_fig = go.Figure()
    for col in results.columns:
        rolling_vol = results[col].rolling(63).std() * np.sqrt(252) * 100
        vol_fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode="lines",
            name=col,
            hovertemplate="%{x|%Y-%m-%d}<br>Volatility: %{y:.2f}%<extra>%{fullData.name}</extra>"
        ))
    vol_fig.update_layout(
        title="Rolling 63-Day Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        template="plotly_white",
        hovermode="x unified"
    )
    save_chart(vol_fig, "03_rolling_volatility.html")

    # Interactive chart 4: Performance metric comparison.
    metric_cols = [
        "CAGR", "Annualized Volatility", "Sharpe Ratio",
        "Maximum Drawdown"
    ]
    comparison = metrics.set_index("Strategy")[metric_cols] * 100
    comparison.columns = [
        "CAGR (%)", "Volatility (%)", "Sharpe Ratio (x100)",
        "Maximum Drawdown (%)"
    ]
    metric_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(comparison.columns)
    )
    for i, col in enumerate(comparison.columns):
        row, column = i // 2 + 1, i % 2 + 1
        metric_fig.add_trace(go.Bar(
            x=comparison.index,
            y=comparison[col],
            name=col,
            showlegend=False,
            text=[f"{v:.2f}" for v in comparison[col]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"
        ), row=row, col=column)
    metric_fig.update_layout(
        title="Strategy Performance Comparison",
        template="plotly_white",
        height=700
    )
    save_chart(metric_fig, "04_metrics_comparison.html")

    # Interactive chart 5: Moving averages for each stock.
    for ticker in TICKERS:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices.index, y=prices[ticker],
            name=f"{ticker} Price", mode="lines"
        ))
        fig.add_trace(go.Scatter(
            x=short_ma.index, y=short_ma[ticker],
            name=f"{SHORT_WINDOW}-day MA", mode="lines"
        ))
        fig.add_trace(go.Scatter(
            x=long_ma.index, y=long_ma[ticker],
            name=f"{LONG_WINDOW}-day MA", mode="lines"
        ))
        fig.update_layout(
            title=f"{ticker} Price and Moving Averages",
            xaxis_title="Date",
            yaxis_title="Adjusted price ($)",
            template="plotly_white",
            hovermode="x unified"
        )
        save_chart(fig, f"moving_average_{ticker}.html")

    # Simple Markdown report.
    with open(REPORTS / "backtest_report.md", "w", encoding="utf-8") as f:
        f.write("# Momentum Strategy Backtest Report\n\n")
        f.write(f"Data period: {prices.index.min().date()} to {prices.index.max().date()}\n\n")
        f.write(f"Transaction cost assumption: {TRANSACTION_COST:.2%} per unit turnover\n\n")
        f.write("## Performance summary\n\n")
        f.write(metrics.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Important limitations\n\n")
        f.write(
            "- Historical performance does not guarantee future results.\n"
            "- Results are sensitive to lookback windows and transaction costs.\n"
            "- Taxes, slippage beyond the modeled cost, and market impact are not modeled.\n"
            "- This is an educational backtest, not investment advice.\n"
        )

    print("\n=== PERFORMANCE SUMMARY ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBacktest completed.")
    print(f"Interactive charts: {CHARTS}")
    print(f"Reports: {REPORTS}")


if __name__ == "__main__":
    main()
