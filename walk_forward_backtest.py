import pandas as pd

from pull_data import TICKER_A, TICKER_B, load_data
from cointegration_test import compute_hedge_ratio


def run_walk_forward(data, ticker_a=TICKER_A, ticker_b=TICKER_B,
                      lookback=252, refit_every=63,
                      window=60, entry_z=2.0, exit_z=0.5):
    """
    The hedge ratio is re-estimated every `refit_every` days using only the
    trailing `lookback` days of data, then applied forward-only
    """
    n = len(data)

    hedge_ratios = pd.Series(index=data.index, dtype=float)
    for start in range(lookback, n, refit_every):
        train = data.iloc[start - lookback:start]           # strictly PAST data
        hr = compute_hedge_ratio(train[ticker_a], train[ticker_b])
        end = min(start + refit_every, n)
        hedge_ratios.iloc[start:end] = hr                     # applied FORWARD only

    valid = hedge_ratios.notna()
    spread = (data[ticker_a] - hedge_ratios * data[ticker_b])[valid]  # no trading during warmup

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

    return pd.DataFrame({
        "hedge_ratio": hedge_ratios[valid],
        "spread": spread,
        "zscore": zscore,
        "position": positions,
        "pnl": pnl,
        "cumulative_pnl": pnl.cumsum(),
    })


if __name__ == "__main__":
    data = load_data()
    result = run_walk_forward(data)

    print(f"Trading days used (after {252}-day warmup): {len(result)}")
    print(f"Total P&L: {result['cumulative_pnl'].iloc[-1]:.2f}")
    print(f"Sharpe (daily, not annualized): {result['pnl'].mean() / result['pnl'].std():.3f}")
    result.to_csv(f"data/{TICKER_A}_{TICKER_B}_walk_forward_pnl.csv")
