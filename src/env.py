from __future__ import annotations

"""
TradingEnv: Gym-compatible single-asset trading environment.

Observation : [return, dev5, dev20, rsi, macd_signal, bb_width, vol_norm, atr, obv_norm, position]
Actions     : 0 = Decrease position, 1 = Hold position, 2 = Increase position
Reward      : position * raw_daily_return - cost - drawdown_penalty * drawdown

The environment tracks a full episode history (positions, trades, portfolio value)
so that evaluation code can inspect the agent's behaviour after an episode.
"""
import numpy as np
import pandas as pd
from src.data import FEATURE_COLS

ACTION_TO_DELTA = {0: -1, 1: 0, 2: 1}   # decrease, hold, increase
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

    def __init__(
        self,
        df: pd.DataFrame,
        transaction_cost: float = TRANSACTION_COST,
        drawdown_penalty: float = 0.5,
        proportional_cost: bool = False,
        random_start: bool = True,
        start_index: int | None = None,
        episode_length: int | None = None,
        min_position: int = -1,
        max_position: int = 1,
    ):
        self.df = df.reset_index(drop=True)
        self.n = len(df)
        self.transaction_cost = transaction_cost
        self.drawdown_penalty = drawdown_penalty
        self.proportional_cost = proportional_cost
        self.random_start = random_start
        self.default_start_index = start_index
        self.episode_length = episode_length
        self.min_position = min_position
        self.max_position = max_position

        self.observation_space_size = len(FEATURE_COLS) + 1   # features + position
        self.action_space_size = 3

        # Pre-extract arrays for faster step()
        self._features = df[FEATURE_COLS].values.astype(np.float32)
        self._raw_returns = df["raw_return"].values.astype(np.float64)

        self.reset()

    def _max_start_index(self) -> int:
        max_start = self.n - 2
        if self.episode_length is not None:
            max_start = min(max_start, self.n - self.episode_length)
        if max_start < 0:
            raise ValueError("TradingEnv does not have enough rows for the configured episode length.")
        return max_start

    def _resolve_start_index(self, random_start: bool, start_index: int | None) -> int:
        if self.n < 2:
            raise ValueError("TradingEnv requires at least 2 rows of data.")

        max_start = self._max_start_index()
        if start_index is not None:
            if not 0 <= start_index <= max_start:
                raise ValueError(f"start_index must be in [0, {max_start}], got {start_index}.")
            return int(start_index)

        if random_start:
            return int(np.random.randint(0, max_start + 1))

        return 0

    def _resolve_end_index(self, start_index: int) -> int:
        if self.episode_length is None:
            return self.n - 1
        return min(start_index + self.episode_length - 1, self.n - 1)

    def reset(self, random_start: bool | None = None, start_index: int | None = None):
        """Reset environment with either a random or deterministic start index."""
        if random_start is None:
            random_start = self.random_start
        if start_index is None:
            start_index = self.default_start_index

        self.t = self._resolve_start_index(random_start=random_start, start_index=start_index)
        self.end_t = self._resolve_end_index(self.t)
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

        old_pos = self.position
        delta = ACTION_TO_DELTA[action]
        self.position = int(np.clip(self.position + delta, self.min_position, self.max_position))
        new_pos = self.position
        traded = new_pos != old_pos
        self.t += 1

        done = self.t >= self.end_t
        daily_ret = self._raw_returns[self.t]

        # Transaction cost: flat fee or proportional to position change size
        if traded:
            if self.proportional_cost:
                cost = self.transaction_cost * abs(new_pos - old_pos)
            else:
                cost = self.transaction_cost
        else:
            cost = 0.0

        prev_val = self.portfolio[-1]
        new_val_tmp = prev_val * (1 + self.position * daily_ret - cost)

        # log return reward (numerically stable)
        reward = float(np.log((new_val_tmp + 1e-12) / (prev_val + 1e-12)))

        # optional drawdown penalty
        peak = max(self.portfolio)
        dd = (new_val_tmp - peak) / (peak + 1e-8) if new_val_tmp < peak else 0.0
        reward += self.drawdown_penalty * dd
        reward = float(np.clip(reward, -1.0, 1.0))
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
