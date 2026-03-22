"""
Training script.

Usage:
    python train.py --ticker AAPL --agent dqn --episodes 200
    python train.py --ticker AAPL --agent ddqn --episodes 200
    python train.py --ticker AAPL --agent both --episodes 200
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Fix Windows encoding for unicode output
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import load, TradingEnv, DQNAgent, DDQNAgent
from src import run_episode, greedy_episode, metrics, buy_and_hold

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def train_agent_with_data(ticker, agent_type, episodes, device, train_df, val_df, test_df):
    train_env = TradingEnv(train_df)
    val_env   = TradingEnv(val_df)
    test_env  = TradingEnv(test_df)

    state_dim  = train_env.observation_space_size
    action_dim = train_env.action_space_size

    AgentClass = DDQNAgent if agent_type == "ddqn" else DQNAgent
    agent = AgentClass(state_dim=state_dim, action_dim=action_dim, device=device)

    train_rewards, val_sharpes = [], []

    print(f"\n{'='*60}")
    print(f"  Training {agent_type.upper()} on {ticker} | {episodes} episodes | {device}")
    print(f"{'='*60}")

    for ep in range(1, episodes + 1):
        reward, _ = run_episode(agent, train_env, train=True)
        train_rewards.append(reward)

        if ep % 10 == 0:
            port = greedy_episode(agent, val_env)
            m = metrics(port)
            val_sharpes.append(m["sharpe"])
            print(
                f"  Ep {ep:>4d}/{episodes} | reward={reward:+.4f} | "
                f"val_sharpe={m['sharpe']:+.3f} | "
                f"val_return={m['cumulative_return']:+.2%} | "
                f"eps={agent.epsilon:.3f}"
            )

    # -- Test evaluation --
    port_agent = greedy_episode(agent, test_env)
    m_agent = metrics(port_agent)
    m_bnh   = buy_and_hold(test_df)

    print(f"\n  {'-'*50}")
    print(f"  TEST RESULTS ({agent_type.upper()} on {ticker})")
    print(f"  {'-'*50}")
    print(f"  {'Metric':<22} {'Agent':>10} {'Buy&Hold':>10}")
    print(f"  {'-'*50}")
    for k in ["cumulative_return", "sharpe", "max_drawdown"]:
        print(f"  {k:<22} {m_agent[k]:>+10.4f} {m_bnh[k]:>+10.4f}")

    # -- Save model --
    model_path = os.path.join(RESULTS_DIR, f"{agent_type}_{ticker}.pt")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    return agent, train_rewards, port_agent, m_agent, m_bnh


def plot_results(ticker, results, test_df):
    """Plot training curves + test portfolio for all trained agents."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training rewards
    for name, (_, train_rewards, _, _, _) in results.items():
        axes[0].plot(train_rewards, alpha=0.3, linewidth=0.5)
        ma = np.convolve(train_rewards, np.ones(20) / 20, mode="valid")
        axes[0].plot(ma, label=f"{name.upper()} (20-ep MA)", linewidth=2)
    axes[0].set_title(f"Training Reward ({ticker})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Test portfolio
    bnh_port = test_df["close"].values / test_df["close"].values[0]
    axes[1].plot(bnh_port, label="Buy & Hold", linestyle="--", color="gray", linewidth=2)
    colors = ["tab:blue", "tab:orange"]
    for i, (name, (_, _, port_agent, _, _)) in enumerate(results.items()):
        axes[1].plot(port_agent, label=name.upper(), color=colors[i % 2], linewidth=2)
    axes[1].set_title(f"Test Portfolio ({ticker})")
    axes[1].set_xlabel("Trading Day")
    axes[1].set_ylabel("Portfolio Value (normalised)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"results_{ticker}.png")
    plt.savefig(fname, dpi=150)
    print(f"\n  Plot saved: {fname}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default="AAPL")
    parser.add_argument("--agent",    default="both", choices=["dqn", "ddqn", "both"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    agents_to_train = ["dqn", "ddqn"] if args.agent == "both" else [args.agent]

    print("Downloading and preparing data...")
    train_df, val_df, test_df = load(args.ticker)
    print(f"Data ready: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    results = {}
    for agent_type in agents_to_train:
        result = train_agent_with_data(args.ticker, agent_type, args.episodes, args.device,
                                       train_df, val_df, test_df)
        results[agent_type] = result

    plot_results(args.ticker, results, test_df)

    print(f"\n{'='*60}")
    print(f"  All done! Check results/ for plots and saved models.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
