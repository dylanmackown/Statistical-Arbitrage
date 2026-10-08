import matplotlib.pyplot as plt

from pull_data import TICKER_A, TICKER_B, load_data

data = load_data(TICKER_A, TICKER_B)

# Normalize both series to start at 100, so they're comparable regardless
# of the fact that Ticker A and Ticker B trade at very different price levels
normalized = data / data.iloc[0] * 100

# Simple spread: the gap between the two normalized series.
spread = normalized[TICKER_A] - normalized[TICKER_B]
rolling_mean = spread.rolling(60).mean()
rolling_std = spread.rolling(60).std()

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(normalized.index, normalized[TICKER_A], label=TICKER_A)
axes[0].plot(normalized.index, normalized[TICKER_B], label=TICKER_B)
axes[0].set_title("Normalized prices (both start at 100)")
axes[0].legend()

axes[1].plot(spread.index, spread, label=f"Spread ({TICKER_A} - {TICKER_B}, normalized)", color="black")
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
plt.savefig(f"data/{TICKER_A}_{TICKER_B}_spread.png")
plt.show()