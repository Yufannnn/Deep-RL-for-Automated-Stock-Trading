"""
Visualisations for data pipeline & TradingEnv contributions (Yufan).

Generates:
  1. Feature correlation heatmap
  2. Technical indicators time-series panel (AAPL)
  3. Feature distributions across train/val/test splits
  4. Environment episode walkthrough (position, portfolio, reward)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src import load, TradingEnv, DQNAgent, DDQNAgent, greedy_episode, FEATURE_COLS
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def fig1_feature_correlation(train_df):
    """Correlation heatmap of all 9 engineered features."""
    corr = train_df[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    labels = ["Return", "MA5 Dev", "MA20 Dev", "RSI", "MACD", "BB Width",
              "Vol Norm", "ATR", "OBV Norm"]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation", fontsize=10)
    ax.set_title("Feature Correlation Matrix (AAPL Training Set)", fontsize=12, pad=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_correlation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig2_technical_indicators(train_df, val_df, test_df):
    """Time-series panel showing price + all technical indicators."""
    full = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    t1 = len(train_df)
    t2 = t1 + len(val_df)

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)

    # Panel 1: Close price
    axes[0].plot(full["close"].values, color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("Price (USD)")
    axes[0].set_title("AAPL Technical Indicators (2009-2020)", fontsize=13)

    # Panel 2: MA deviations
    axes[1].plot(full["dev5"].values, color="tab:blue", linewidth=0.6, alpha=0.8, label="MA5 Dev")
    axes[1].plot(full["dev20"].values, color="tab:orange", linewidth=0.6, alpha=0.8, label="MA20 Dev")
    axes[1].set_ylabel("MA Deviation")
    axes[1].legend(fontsize=8, loc="upper left")

    # Panel 3: RSI + MACD
    ax3 = axes[2]
    ax3.plot(full["rsi"].values, color="tab:purple", linewidth=0.6, alpha=0.8, label="RSI (scaled)")
    ax3.set_ylabel("RSI")
    ax3.legend(fontsize=8, loc="upper left")
    ax3b = ax3.twinx()
    ax3b.plot(full["macd_signal"].values, color="tab:green", linewidth=0.6, alpha=0.6, label="MACD Signal")
    ax3b.set_ylabel("MACD Signal", color="tab:green")
    ax3b.legend(fontsize=8, loc="upper right")

    # Panel 4: Bollinger Width + ATR
    axes[3].plot(full["bb_width"].values, color="tab:red", linewidth=0.6, alpha=0.8, label="BB Width")
    axes[3].set_ylabel("BB Width")
    axes[3].legend(fontsize=8, loc="upper left")
    ax4b = axes[3].twinx()
    ax4b.plot(full["atr"].values, color="tab:brown", linewidth=0.6, alpha=0.6, label="ATR (norm)")
    ax4b.set_ylabel("ATR", color="tab:brown")
    ax4b.legend(fontsize=8, loc="upper right")

    # Panel 5: Volume + OBV
    axes[4].plot(full["vol_norm"].values, color="tab:cyan", linewidth=0.5, alpha=0.7, label="Vol Norm")
    axes[4].set_ylabel("Vol Norm")
    axes[4].legend(fontsize=8, loc="upper left")
    ax5b = axes[4].twinx()
    ax5b.plot(full["obv_norm"].values, color="tab:olive", linewidth=0.5, alpha=0.7, label="OBV Norm")
    ax5b.set_ylabel("OBV Norm", color="tab:olive")
    ax5b.legend(fontsize=8, loc="upper right")

    axes[4].set_xlabel("Trading Day")

    # Add split boundaries to all panels
    for ax in axes:
        ax.axvline(t1, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axvline(t2, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "technical_indicators.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig3_feature_distributions(train_df, val_df, test_df):
    """Feature distributions across train/val/test to show normalisation effect."""
    labels = ["Return", "MA5 Dev", "MA20 Dev", "RSI", "MACD", "BB Width",
              "Vol Norm", "ATR", "OBV Norm"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for i, (col, label) in enumerate(zip(FEATURE_COLS, labels)):
        ax = axes[i]
        for split, name, color in [
            (train_df, "Train", "steelblue"),
            (val_df, "Val", "darkorange"),
            (test_df, "Test", "seagreen"),
        ]:
            vals = split[col].values
            ax.hist(vals, bins=60, alpha=0.5, color=color, density=True, label=name)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Feature Distributions: Train vs Val vs Test (z-score normalised, AAPL)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def fig4_episode_walkthrough(test_df):
    """Show agent behaviour: positions, portfolio value, and rewards over a test episode."""
    # Try to load a trained model; if not available, run a random agent
    env = TradingEnv(test_df)
    state_dim = env.observation_space_size
    action_dim = env.action_space_size

    model_path = os.path.join(RESULTS_DIR, "dqn_AAPL.pt")
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        agent.epsilon = 0.0
        label = "DQN Agent"
    except Exception:
        # Model was trained with different feature set; retrain quickly
        from src import run_episode
        train_df_tmp, _, _ = load("AAPL")
        train_env_tmp = TradingEnv(train_df_tmp)
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        print("    Retraining DQN (50 episodes) for walkthrough figure...")
        for _ in range(50):
            run_episode(agent, train_env_tmp, train=True)
        agent.epsilon = 0.0
        label = "DQN Agent"

    # Run episode
    state = env.reset()
    while True:
        action = agent.select_action(state)
        state, _, done, _ = env.step(action)
        if done:
            break

    positions = np.array(env.positions)
    portfolio = np.array(env.portfolio)
    rewards = np.array(env.rewards)
    prices = test_df["close"].values
    bnh = prices / prices[0]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)

    # Panel 1: Portfolio vs Buy-and-Hold
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(bnh, label="Buy & Hold", color="gray", linestyle="--", linewidth=1.5)
    ax0.plot(portfolio, label=label, color="tab:blue", linewidth=1.5)
    ax0.set_ylabel("Portfolio Value")
    ax0.set_title(f"Test Episode Walkthrough - {label} on AAPL", fontsize=13)
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.2)

    # Panel 2: Positions with colour bands
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    colors = {1: "green", 0: "gold", -1: "red"}
    color_labels = {1: "Long", 0: "Flat", -1: "Short"}
    for pos_val, color in colors.items():
        mask = positions[:-1] == pos_val
        indices = np.where(mask)[0]
        if len(indices) > 0:
            ax1.bar(indices, [1]*len(indices), color=color, alpha=0.6, width=1.0,
                    label=color_labels[pos_val])
    ax1.set_ylabel("Position")
    ax1.set_yticks([])
    ax1.legend(fontsize=8, loc="upper right", ncol=3)
    ax1.grid(alpha=0.2)

    # Panel 3: Cumulative reward
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    cum_reward = np.cumsum(rewards)
    ax2.plot(cum_reward, color="tab:purple", linewidth=1.0)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(alpha=0.2)

    # Panel 4: Trade markers on price
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3.plot(prices, color="steelblue", linewidth=0.8, label="Close Price")
    for t_step, old_pos, new_pos in env.trades:
        if new_pos == 1:
            ax3.scatter(t_step, prices[t_step], marker="^", color="green", s=20, zorder=5)
        elif new_pos == -1:
            ax3.scatter(t_step, prices[t_step], marker="v", color="red", s=20, zorder=5)
        else:
            ax3.scatter(t_step, prices[t_step], marker="o", color="gold", s=15, zorder=5)
    # Legend with dummy markers
    ax3.scatter([], [], marker="^", color="green", s=40, label="Buy")
    ax3.scatter([], [], marker="v", color="red", s=40, label="Sell")
    ax3.scatter([], [], marker="o", color="gold", s=30, label="Flatten")
    ax3.set_ylabel("Price (USD)")
    ax3.set_xlabel("Trading Day")
    ax3.legend(fontsize=8, loc="upper left", ncol=3)
    ax3.grid(alpha=0.2)

    summary = env.summary()
    fig.text(0.99, 0.01,
             f"Trades: {summary['num_trades']} | "
             f"Sharpe: {summary['sharpe']:.2f} | "
             f"Return: {summary['cumulative_return']:+.1%} | "
             f"Max DD: {summary['max_drawdown']:.1%}",
             ha="right", fontsize=9, color="gray")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "episode_walkthrough.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Loading AAPL data...")
    train_df, val_df, test_df = load("AAPL")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n1/4  Feature correlation heatmap...")
    fig1_feature_correlation(train_df)

    print("2/4  Technical indicators panel...")
    fig2_technical_indicators(train_df, val_df, test_df)

    print("3/4  Feature distributions...")
    fig3_feature_distributions(train_df, val_df, test_df)

    print("4/4  Episode walkthrough...")
    fig4_episode_walkthrough(test_df)

    print("\nAll figures saved to results/")
