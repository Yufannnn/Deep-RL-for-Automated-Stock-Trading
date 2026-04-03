"""
Classical baselines used for comparison against the RL agents.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from src.evaluate import metrics

BASELINE_METRICS_TEMPLATE = "baseline_metrics_{ticker}.csv"
BASELINE_PORTFOLIOS_TEMPLATE = "baseline_portfolios_{ticker}.csv"


def buy_and_hold_portfolio(df: pd.DataFrame) -> np.ndarray:
    prices = df["close"].to_numpy(dtype=np.float64)
    return prices / (prices[0] + 1e-12)


def _fallback_ar1_forecast(history: np.ndarray) -> float:
    """Lightweight AR(1) fallback when statsmodels is unavailable."""
    history = np.asarray(history, dtype=np.float64)
    history = history[np.isfinite(history)]

    if history.size == 0:
        return 0.0
    if history.size == 1:
        return float(history[-1])

    x = history[:-1]
    y = history[1:]
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-12:
        return float(y.mean())

    phi = float(np.dot(x_centered, y - y.mean()) / denom)
    intercept = float(y.mean() - phi * x.mean())
    return intercept + phi * float(history[-1])


def _arima_forecast(history: np.ndarray, order=(1, 0, 0)) -> float:
    history = np.asarray(history, dtype=np.float64)
    history = history[np.isfinite(history)]

    if history.size < max(order[0] + order[2] + 2, 5):
        return float(history.mean()) if history.size else 0.0

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return _fallback_ar1_forecast(history)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(
                history,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
        return float(np.asarray(forecast)[0])
    except Exception:
        return _fallback_ar1_forecast(history)


def _signal_to_position(forecast: float, min_position: int, max_position: int) -> int:
    if forecast > 0:
        return max(max_position, 0)
    if forecast < 0:
        return min(min_position, 0)
    return 0


def arima_portfolio(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_position: int = 0,
    max_position: int = 1,
    lookback: int = 252,
    order=(1, 0, 0),
) -> np.ndarray:
    """Walk-forward ARIMA baseline over test-set returns."""
    history = np.concatenate(
        [
            train_df["raw_return"].to_numpy(dtype=np.float64),
            val_df["raw_return"].to_numpy(dtype=np.float64),
        ]
    )
    history = np.nan_to_num(history, nan=0.0)
    test_returns = np.nan_to_num(test_df["raw_return"].to_numpy(dtype=np.float64), nan=0.0)

    portfolio = [1.0]
    for t in range(1, len(test_returns)):
        observed = np.concatenate([history, test_returns[:t]])
        forecast = _arima_forecast(observed[-lookback:], order=order)
        position = _signal_to_position(forecast, min_position=min_position, max_position=max_position)
        portfolio.append(portfolio[-1] * (1.0 + position * test_returns[t]))

    return np.asarray(portfolio, dtype=np.float64)


def _cache_paths(results_dir: str, ticker: str):
    return (
        os.path.join(results_dir, BASELINE_METRICS_TEMPLATE.format(ticker=ticker)),
        os.path.join(results_dir, BASELINE_PORTFOLIOS_TEMPLATE.format(ticker=ticker)),
    )


def _load_cache(results_dir: str, ticker: str):
    metrics_path, portfolios_path = _cache_paths(results_dir, ticker)
    if not (os.path.exists(metrics_path) and os.path.exists(portfolios_path)):
        return None

    metrics_df = pd.read_csv(metrics_path)
    portfolios_df = pd.read_csv(portfolios_path)

    results = {
        row["model"]: {
            "cumulative_return": float(row["cumulative_return"]),
            "sharpe": float(row["sharpe"]),
            "max_drawdown": float(row["max_drawdown"]),
        }
        for _, row in metrics_df.iterrows()
    }
    portfolios = {
        col: portfolios_df[col].dropna().to_numpy(dtype=np.float64)
        for col in portfolios_df.columns
    }
    return portfolios, results


def _save_cache(results_dir: str, ticker: str, portfolios, results) -> None:
    os.makedirs(results_dir, exist_ok=True)
    metrics_path, portfolios_path = _cache_paths(results_dir, ticker)

    metrics_df = pd.DataFrame(
        [{"model": name, **metric_values} for name, metric_values in results.items()]
    )
    metrics_df.to_csv(metrics_path, index=False)

    max_len = max(len(portfolio) for portfolio in portfolios.values())
    portfolio_df = pd.DataFrame(
        {
            name: np.pad(
                portfolio,
                (0, max_len - len(portfolio)),
                mode="constant",
                constant_values=np.nan,
            )
            for name, portfolio in portfolios.items()
        }
    )
    portfolio_df.to_csv(portfolios_path, index=False)


def run_all_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_position: int = 0,
    max_position: int = 1,
    results_dir: str | None = None,
    ticker: str | None = None,
    refresh: bool = False,
):
    if results_dir and ticker and not refresh:
        cached = _load_cache(results_dir, ticker)
        if cached is not None:
            return cached

    portfolios = {
        "buy_and_hold": buy_and_hold_portfolio(test_df),
        "arima": arima_portfolio(
            train_df,
            val_df,
            test_df,
            min_position=min_position,
            max_position=max_position,
        ),
    }
    results = {name: metrics(portfolio) for name, portfolio in portfolios.items()}

    if results_dir and ticker:
        _save_cache(results_dir, ticker, portfolios, results)

    return portfolios, results
