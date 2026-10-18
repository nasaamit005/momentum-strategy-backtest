
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the src folder importable when pytest runs from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_backtest import (
    calculate_metrics,
    make_signals,
    run_strategy,
    TICKERS,
    TRANSACTION_COST,
)


def sample_prices(rows=260):
    """Create reproducible synthetic prices for six stocks."""
    dates = pd.bdate_range("2024-01-01", periods=rows)
    data = {}

    for i, ticker in enumerate(TICKERS):
        data[ticker] = np.linspace(
            100 + i * 10,
            150 + i * 10,
            rows,
        )

    data["SPY"] = np.linspace(100, 130, rows)
    return pd.DataFrame(data, index=dates)


def test_signals_have_valid_shapes():
    prices = sample_prices()
    crossover, momentum, *_ = make_signals(prices)

    assert crossover.shape == (len(prices), len(TICKERS))
    assert momentum.shape == (len(prices), len(TICKERS))


def test_portfolio_weights_are_long_only():
    prices = sample_prices()
    crossover, momentum, *_ = make_signals(prices)

    assert (crossover.to_numpy() >= 0).all()
    assert (momentum.to_numpy() >= 0).all()
    assert (crossover.sum(axis=1) <= 1.000001).all()
    assert (momentum.sum(axis=1) <= 1.000001).all()


def test_strategy_returns_are_finite():
    prices = sample_prices()
    crossover, *_ = make_signals(prices)
    result = run_strategy(prices, crossover)

    assert np.isfinite(result["net_return"]).all()
    assert np.isfinite(result["equity"]).all()
    assert (result["equity"] > 0).all()


def test_transaction_costs_are_nonnegative():
    prices = sample_prices()
    crossover, *_ = make_signals(prices)
    result = run_strategy(prices, crossover)

    assert (result["transaction_cost"] >= 0).all()
    assert np.allclose(
        result["transaction_cost"],
        result["turnover"] * TRANSACTION_COST,
    )


def test_metrics_contain_expected_values():
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
    equity = (1 + returns).cumprod()

    metrics = calculate_metrics(returns, equity)

    assert metrics["Total Return"] > 0
    assert metrics["CAGR"] > -1
    assert metrics["Annualized Volatility"] >= 0
    assert metrics["Maximum Drawdown"] <= 0


def test_zero_return_series_has_zero_drawdown():
    returns = pd.Series([0.0] * 10)
    equity = pd.Series([1.0] * 10)

    metrics = calculate_metrics(returns, equity)

    assert metrics["Total Return"] == 0
    assert metrics["Maximum Drawdown"] == 0
    assert np.isnan(metrics["Sharpe Ratio"])
    assert np.isnan(metrics["Calmar Ratio"])
