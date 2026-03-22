"""
TradingEnv: Gym-compatible single-asset trading environment.

Observation : [return, dev5, dev20, rsi, macd_signal, bb_width, vol_norm, atr, obv_norm, position]
Actions     : 0 = Sell (short), 1 = Hold (flat), 2 = Buy (long)
Reward      : position * raw_daily_return - cost - drawdown_penalty * drawdown

The environment tracks a full episode history (positions, trades, portfolio value)
so that evaluation code can inspect the agent's behaviour after an episode.
"""
import numpy as np
import pandas as pd
from src.data import FEATURE_COLS

ACTION_TO_POS = {0: -1, 1: 0, 2: 1}   # Sell=-1, Hold=0, Buy=+1
POS_LABELS = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
TRANSACTION_COST = 0.001


class TradingEnv:
    """Gym-style trading environment for a single asset.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns in FEATURE_COLS plus 'raw_return' and 'close'.
    transaction_cost : float
        Cost applied each time the position changes (default 0.001 = 10 bps).
    """

    def __init__(self, df: pd.DataFrame, transaction_cost: float = TRANSACTION_COST,
                 drawdown_penalty: float = 0.5, proportional_cost: bool = False):
        self.df = df.reset_index(drop=True)
        self.n = len(df)
        self.transaction_cost = transaction_cost
        self.drawdown_penalty = drawdown_penalty
        self.proportional_cost = proportional_cost

        self.observation_space_size = len(FEATURE_COLS) + 1   # features + position
        self.action_space_size = 3

        # Pre-extract arrays for faster step()
        self._features = df[FEATURE_COLS].values.astype(np.float32)
        self._raw_returns = df["raw_return"].values.astype(np.float64)

        self.reset()

    def reset(self):
        """Reset environment to the beginning of the data."""
        self.t = 0
        self.position = 0  # start flat

        # Episode history
        self.positions = [0]
        self.trades = []          # list of (timestep, old_pos, new_pos)
        self.rewards = []
        self.portfolio = [1.0]    # normalised portfolio value

        return self._obs()

    def _obs(self) -> np.ndarray:
        """Return current observation vector."""
        feats = self._features[self.t]
        return np.append(feats, np.float32(self.position))

    def step(self, action: int):
        """Execute one trading step.

        Returns: (observation, reward, done, info)
        """
        assert 0 <= action <= 2, f"Invalid action {action}"

        new_pos = ACTION_TO_POS[action]
        traded = new_pos != self.position
        old_pos = self.position

        self.position = new_pos
        self.t += 1

        done = self.t >= self.n - 1
        daily_ret = self._raw_returns[self.t]

        # Transaction cost: flat fee or proportional to position change size
        if traded:
            if self.proportional_cost:
                cost = self.transaction_cost * abs(new_pos - old_pos)
            else:
                cost = self.transaction_cost
        else:
            cost = 0.0

        pnl = self.position * daily_ret - cost

        # Drawdown penalty: penalise drops from portfolio peak
        new_val_tmp = self.portfolio[-1] * (1 + self.position * daily_ret - cost)
        peak = max(self.portfolio)
        dd = (new_val_tmp - peak) / (peak + 1e-8) if new_val_tmp < peak else 0.0
        reward = float(pnl + self.drawdown_penalty * dd)

        # Update episode history
        self.positions.append(self.position)
        self.rewards.append(reward)
        if traded:
            self.trades.append((self.t, old_pos, new_pos))

        # Track portfolio (compounded)
        new_val = self.portfolio[-1] * (1 + self.position * daily_ret - cost)
        self.portfolio.append(max(new_val, 1e-8))

        info = {
            "daily_return": daily_ret,
            "position": self.position,
            "traded": traded,
            "portfolio_value": self.portfolio[-1],
        }

        return self._obs(), reward, done, info

    @property
    def num_trades(self) -> int:
        """Total number of position changes so far."""
        return len(self.trades)

    @property
    def current_portfolio_value(self) -> float:
        return self.portfolio[-1]

    def summary(self) -> dict:
        """Return a summary dict of the completed episode."""
        positions = np.array(self.positions)
        n_long  = int((positions == 1).sum())
        n_short = int((positions == -1).sum())
        n_flat  = int((positions == 0).sum())

        port = np.array(self.portfolio)
        rets = np.diff(port) / (port[:-1] + 1e-12)
        sharpe = float((np.mean(rets) / (np.std(rets) + 1e-8)) * np.sqrt(252))
        peak = np.maximum.accumulate(port)
        max_dd = float(((port - peak) / (peak + 1e-12)).min())

        return {
            "steps": self.t,
            "num_trades": self.num_trades,
            "trades_per_day": self.num_trades / max(self.t, 1),
            "total_reward": sum(self.rewards),
            "final_portfolio": self.portfolio[-1],
            "cumulative_return": self.portfolio[-1] - 1.0,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "pct_long": n_long / len(positions),
            "pct_short": n_short / len(positions),
            "pct_flat": n_flat / len(positions),
        }
