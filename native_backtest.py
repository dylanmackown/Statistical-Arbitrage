import pandas as pd

from pull_data import TICKER_A, TICKER_B, load_data
from cointegration_test import compute_hedge_ratio

# Contains look-ahead bias intentionally, it is a control to show how much the walk-forward version's results differ once that leak is removed
def run_naive(data, ticker_a=TICKER_A, ticker_b=TICKER_B,
              window=60, entry_z=2.0, exit_z=0.5):
    hedge_ratio = compute_hedge_ratio(data[ticker_a], data[ticker_b])
    spread = data[ticker_a] - hedge_ratio * data[ticker_b]

    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std

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

    pnl = (positions.shift(1) * spread.diff()).fillna(0)

    return pd.DataFrame({
        "hedge_ratio": hedge_ratio,  # constant, broadcasts to every row
        "spread": spread,
        "zscore": zscore,
        "position": positions,
        "pnl": pnl,
        "cumulative_pnl": pnl.cumsum(),
    })


if __name__ == "__main__":
    data = load_data()
    result = run_naive(data)

    print(f"Hedge ratio (fit on full sample): {result['hedge_ratio'].iloc[0]:.4f}")
    print(f"Total P&L: {result['cumulative_pnl'].iloc[-1]:.2f}")
    print(f"Sharpe (daily, not annualized): {result['pnl'].mean() / result['pnl'].std():.3f}")
    result.to_csv(f"data/{TICKER_A}_{TICKER_B}_naive_pnl.csv")
