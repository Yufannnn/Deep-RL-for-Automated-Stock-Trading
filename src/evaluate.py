"""
Evaluation: cumulative return, Sharpe ratio, max drawdown.
Also runs buy-and-hold baseline.
"""
import numpy as np
import pandas as pd


def run_episode(agent, env, train=False, greedy=False, reset_kwargs=None):
    """Run one episode. If train=True, collect experience and update."""
    if reset_kwargs is None:
        reset_kwargs = {}

    state = env.reset(**reset_kwargs)
    total_reward = 0.0

    while True:
        action = agent.select_action(state, greedy=greedy)
        next_state, reward, done, info = env.step(action)
        if train:
            agent.push(state, action, reward, next_state, float(done))
            agent.update()
        state = next_state
        total_reward += reward

        if done:
            break

    # Portfolio is now tracked inside env
    return total_reward, np.array(env.portfolio)


def metrics(portfolio: np.ndarray, freq: int = 252):
    """Compute cumulative return, annualised Sharpe, max drawdown."""
    rets = np.diff(portfolio) / (portfolio[:-1] + 1e-12)
    cum_return = portfolio[-1] / portfolio[0] - 1
    sharpe = (np.mean(rets) / (np.std(rets) + 1e-8)) * np.sqrt(freq)
    peak = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - peak) / (peak + 1e-12)
    max_dd = float(drawdown.min())
    return {"cumulative_return": cum_return, "sharpe": sharpe, "max_drawdown": max_dd}


def buy_and_hold(df: pd.DataFrame):
    prices = df["close"].values
    portfolio = prices / prices[0]
    return metrics(portfolio)


def greedy_episode(agent, env):
    """Run deterministic full-horizon evaluation with pure exploitation."""
    _, portfolio = run_episode(agent, env, train=False, greedy=True, reset_kwargs={"random_start": False, "start_index": 0})
    return portfolio
