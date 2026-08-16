# walk_forward_backtest.py
import pandas as pd
import statsmodels.api as sm

from pull_data import TICKER_A, TICKER_B

data = pd.read_csv(f"data/{TICKER_A}_{TICKER_B}.csv", index_col=0, parse_dates=True)
n = len(data)

lookback = 252    # ~1 trading year to estimate the hedge ratio
refit_every = 63  # ~1 quarter before re-estimating
window, entry_z, exit_z = 60, 2.0, 0.5

# --- THE ACTUAL FIX: many hedge ratios, each fit on trailing data only ---
hedge_ratios = pd.Series(index=data.index, dtype=float)
for start in range(lookback, n, refit_every):
    train = data.iloc[start - lookback:start]          # strictly PAST data
    X = sm.add_constant(train[TICKER_B])
    hr = sm.OLS(train[TICKER_A], X).fit().params[TICKER_B]
    end = min(start + refit_every, n)
    hedge_ratios.iloc[start:end] = hr                   # apply FORWARD only

valid = hedge_ratios.notna()
spread = (data[TICKER_A] - hedge_ratios * data[TICKER_B])[valid]   # no trading during warmup

rolling_mean = spread.rolling(window).mean()
rolling_std = spread.rolling(window).std()
zscore = (spread - rolling_mean) / rolling_std

positions = pd.Series(0.0, index=spread.index)
position = 0
for i in range(len(spread)):
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

pnl = (positions.shift(1) * spread.diff()).fillna(0)
cumulative_pnl = pnl.cumsum()

print(f"Trading days used (after {lookback}-day warmup): {len(spread)}")
print(f"Total P&L: {cumulative_pnl.iloc[-1]:.2f}")
print(f"Sharpe (daily, not annualized): {pnl.mean() / pnl.std():.3f}")
cumulative_pnl.to_csv(f"data/{TICKER_A}_{TICKER_B}_walk_forward_pnl.csv")