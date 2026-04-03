"""
Training script.

Usage:
    python train.py --ticker AAPL --agent dqn --episodes 200
    python train.py --ticker AAPL --agent ddqn --episodes 200
    python train.py --ticker AAPL --agent both --episodes 200
"""
import argparse
import copy
import os
import random
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Fix Windows encoding for unicode output
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import load, TradingEnv, DQNAgent, DDQNAgent
from src import run_episode, greedy_episode, metrics, buy_and_hold
from baselines import run_all_baselines

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TRAIN_EPISODE_LENGTH = 252
TRAIN_MIN_POSITION = 0
TRAIN_MAX_POSITION = 1
TRAIN_DRAWDOWN_PENALTY = 0.0


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    # if torch.mps.is_available():
    #     return "mps"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_agent_with_data(ticker, agent_type, episodes, device, train_df, val_df, test_df, baseline_results=None):
    train_episode_length = min(TRAIN_EPISODE_LENGTH, max(len(train_df) - 1, 1))
    env_kwargs = {
        "min_position": TRAIN_MIN_POSITION,
        "max_position": TRAIN_MAX_POSITION,
        "drawdown_penalty": TRAIN_DRAWDOWN_PENALTY,
    }
    train_env = TradingEnv(train_df, random_start=True, episode_length=train_episode_length, **env_kwargs)
    val_env   = TradingEnv(val_df, random_start=False, start_index=0, **env_kwargs)
    test_env  = TradingEnv(test_df, random_start=False, start_index=0, **env_kwargs)

    state_dim  = train_env.observation_space_size
    action_dim = train_env.action_space_size

    AgentClass = DDQNAgent if agent_type == "ddqn" else DQNAgent
    agent = AgentClass(state_dim=state_dim, action_dim=action_dim, device=device)

    train_rewards, val_sharpes = [], []
    best_val_score = None
    best_state_dict = copy.deepcopy(agent.policy_net.state_dict())

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
            score = (float(m["cumulative_return"]), float(m["sharpe"]))
            if best_val_score is None or score > best_val_score:
                best_val_score = score
                best_state_dict = copy.deepcopy(agent.policy_net.state_dict())
            print(
                f"  Ep {ep:>4d}/{episodes} | reward={reward:+.4f} | "
                f"val_sharpe={m['sharpe']:+.3f} | "
                f"val_return={m['cumulative_return']:+.2%} | "
                f"eps={agent.current_epsilon():.3f}"
            )

    # -- Test evaluation --
    agent.policy_net.load_state_dict(best_state_dict)
    agent.target_net.load_state_dict(best_state_dict)
    port_agent = greedy_episode(agent, test_env)
    m_agent = metrics(port_agent)
    m_bnh   = buy_and_hold(test_df)
    if baseline_results is None:
        _, baseline_results = run_all_baselines(
            train_df,
            val_df,
            test_df,
            min_position=TRAIN_MIN_POSITION,
            max_position=TRAIN_MAX_POSITION,
        )

    print(f"\n  {'-'*50}")
    print(f"  TEST RESULTS ({agent_type.upper()} on {ticker})")
    print(f"  {'-'*50}")
    print(f"  {'Metric':<22} {'Agent':>10} {'Buy&Hold':>10} {'ARIMA':>10}")
    print(f"  {'-'*50}")
    for k in ["cumulative_return", "sharpe", "max_drawdown"]:
        print(
            f"  {k:<22} {m_agent[k]:>+10.4f} "
            f"{m_bnh[k]:>+10.4f} {baseline_results['arima'][k]:>+10.4f}"
        )

    # -- Save model --
    model_path = os.path.join(RESULTS_DIR, f"{agent_type}_{ticker}.pt")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    return agent, train_rewards, port_agent, m_agent, m_bnh, baseline_results


def save_comparison_table(ticker, agent_results, baseline_results):
    rows = []
    for name, metrics_dict in agent_results.items():
        row = {"model": name.upper(), **metrics_dict}
        rows.append(row)
    for name, metrics_dict in baseline_results.items():
        row = {"model": name, **metrics_dict}
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(by="cumulative_return", ascending=False)
    output_path = os.path.join(RESULTS_DIR, f"comparison_{ticker}.csv")
    df.to_csv(output_path, index=False)
    print(f"  Comparison saved: {output_path}")
    return df


def plot_results(ticker, results, test_df, baseline_portfolios=None):
    """Plot training curves + test portfolio for all trained agents."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training rewards
    for name, result in results.items():
        _, train_rewards, _, _, _, _ = result
        axes[0].plot(train_rewards, alpha=0.3, linewidth=0.5)
        window = min(20, max(len(train_rewards), 1))
        ma = np.convolve(train_rewards, np.ones(window) / window, mode="valid")
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
    for i, (name, (_, _, port_agent, _, _, _)) in enumerate(results.items()):
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
    parser.add_argument("--device",   default=default_device())
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--refresh-baselines", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(args.seed)

    agents_to_train = ["dqn", "ddqn"] if args.agent == "both" else [args.agent]

    print("Downloading and preparing data...")
    train_df, val_df, test_df = load(args.ticker)
    print(f"Data ready: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Using seed={args.seed}")

    final_baseline_portfolios, final_baseline_results = run_all_baselines(
        train_df,
        val_df,
        test_df,
        min_position=TRAIN_MIN_POSITION,
        max_position=TRAIN_MAX_POSITION,
        results_dir=RESULTS_DIR,
        ticker=args.ticker,
        refresh=args.refresh_baselines,
    )

    results = {}
    agent_metric_table = {}
    for agent_type in agents_to_train:
        result = train_agent_with_data(
            args.ticker,
            agent_type,
            args.episodes,
            args.device,
            train_df,
            val_df,
            test_df,
            baseline_results=final_baseline_results,
        )
        results[agent_type] = result
        agent_metric_table[agent_type] = result[3]

    print(f"\n{'='*60}")
    print("  OVERALL COMPARISON")
    print(f"{'='*60}")
    comparison_df = save_comparison_table(args.ticker, agent_metric_table, final_baseline_results)
    print(comparison_df.to_string(index=False))

    plot_results(args.ticker, results, test_df, baseline_portfolios=final_baseline_portfolios)

    print(f"\n{'='*60}")
    print(f"  All done! Check results/ for plots and saved models.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
