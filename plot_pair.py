# plot_pair.py
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/ko_pep.csv", index_col=0, parse_dates=True)

# Normalize both series to start at 100, so they're comparable regardless
# of the fact that KO and PEP trade at very different price levels
normalized = data / data.iloc[0] * 100

# Simple spread: the gap between the two normalized series.
# This isn't the "real" spread you'll use later (that comes from the
# regression in step 4) — it's just enough to eyeball reversion.
spread = normalized["KO"] - normalized["PEP"]
rolling_mean = spread.rolling(60).mean()
rolling_std = spread.rolling(60).std()

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(normalized.index, normalized["KO"], label="KO")
axes[0].plot(normalized.index, normalized["PEP"], label="PEP")
axes[0].set_title("Normalized prices (both start at 100)")
axes[0].legend()

axes[1].plot(spread.index, spread, label="Spread (KO - PEP, normalized)", color="black")
axes[1].plot(rolling_mean.index, rolling_mean, label="60-day rolling mean", color="orange")
axes[1].fill_between(
    spread.index,
    rolling_mean - 2 * rolling_std,
    rolling_mean + 2 * rolling_std,
    color="orange", alpha=0.15, label="±2 rolling std"
)
axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.8)
axes[1].set_title("Spread")
axes[1].legend()

plt.tight_layout()
plt.savefig("data/ko_pep_spread.png")
plt.show()