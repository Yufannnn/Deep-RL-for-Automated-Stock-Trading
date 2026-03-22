"""Generate data exploration figures for the report."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src import load

train_df, val_df, test_df = load("AAPL")
import pandas as pd
full = pd.concat([train_df, val_df, test_df])

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Left: price with split boundaries
axes[0].plot(full["close"].values, color="steelblue", linewidth=0.8)
t1 = len(train_df)
t2 = t1 + len(val_df)
axes[0].axvline(t1, color="red", linestyle="--", alpha=0.7, label=f"Train/Val split (day {t1})")
axes[0].axvline(t2, color="orange", linestyle="--", alpha=0.7, label=f"Val/Test split (day {t2})")
axes[0].set_title("AAPL Close Price (2009-2024)")
axes[0].set_xlabel("Trading Day")
axes[0].set_ylabel("Price (USD)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Right: return distribution
raw_rets = full["raw_return"].values
axes[1].hist(raw_rets, bins=100, color="steelblue", edgecolor="none", alpha=0.7, density=True)
axes[1].set_title("Distribution of Daily Returns (AAPL)")
axes[1].set_xlabel("Daily Return")
axes[1].set_ylabel("Density")
mu, sigma = np.mean(raw_rets), np.std(raw_rets)
axes[1].axvline(mu, color="red", linestyle="--", label=f"Mean={mu:.4f}")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/data_exploration.png", dpi=150)
print("Saved results/data_exploration.png")
