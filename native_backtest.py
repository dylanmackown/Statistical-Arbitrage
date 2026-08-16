# naive_backtest.py
import pandas as pd
import statsmodels.api as sm

from pull_data import TICKER_A, TICKER_B

data = pd.read_csv(f"data/{TICKER_A}_{TICKER_B}.csv", index_col=0, parse_dates=True)

# --- Fit hedge ratio on the ENTIRE sample (this is the naive part) ---
X = sm.add_constant(data[TICKER_B])
hedge_ratio = sm.OLS(data[TICKER_A], X).fit().params[TICKER_B]

spread = data[TICKER_A] - hedge_ratio * data[TICKER_B]

# --- Rolling z-score (this part is already trailing-only, no leak here) ---
window = 60
rolling_mean = spread.rolling(window).mean()
rolling_std = spread.rolling(window).std()
zscore = (spread - rolling_mean) / rolling_std

# --- Stateful position rule: enter at |z|>2, exit near 0 ---
entry_z, exit_z = 2.0, 0.5
positions = pd.Series(0.0, index=data.index)
position = 0
for i in range(len(data)):
    z = zscore.iloc[i]
    if pd.isna(z):
        continue
    if position == 0:
        if z > entry_z:
            position = -1
        elif z < -entry_z:
            position = 1
    else:
        if abs(z) < exit_z:
            position = 0
    positions.iloc[i] = position

# --- P&L: yesterday's position times today's change in spread ---
pnl = (positions.shift(1) * spread.diff()).fillna(0)
cumulative_pnl = pnl.cumsum()

print(f"Hedge ratio (fit on full sample): {hedge_ratio:.4f}")
print(f"Total P&L: {cumulative_pnl.iloc[-1]:.2f}")
print(f"Sharpe (daily, not annualized): {pnl.mean() / pnl.std():.3f}")
cumulative_pnl.to_csv(f"data/{TICKER_A}_{TICKER_B}_naive_pnl.csv")