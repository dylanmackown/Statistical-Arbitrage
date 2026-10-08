import pandas as pd

from pull_data import TICKER_A, TICKER_B, load_data
from cointegration_test import compute_hedge_ratio
from performance import to_returns, perf_stats, COST_BPS


def run_walk_forward(data, ticker_a=TICKER_A, ticker_b=TICKER_B,
                      lookback=252, refit_every=63,
                      window=60, entry_z=2.0, exit_z=0.5):
    """Walk-forward pairs backtest.

    The hedge ratio is re-estimated every `refit_every` days using only the
    trailing `lookback` days of data, then applied forward-only — this is
    the fix for the look-ahead bias baked into native_backtest.py.
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


def grid_search_thresholds(data, ticker_a=TICKER_A, ticker_b=TICKER_B,
                            train_end=None,
                            entry_grid=(1.5, 2.0, 2.5, 3.0),
                            exit_grid=(0.0, 0.25, 0.5, 0.75),
                            lookback=252, refit_every=63, window=60,
                            cost_bps=COST_BPS, metric="Sharpe"):
    """Grid-search entry/exit z-score thresholds on an IN-SAMPLE (train) split.

    Returns (best_entry_z, best_exit_z, grid_results_df) where grid_results_df
    is sorted best-first by `metric`
    """
    train = data if train_end is None else data.loc[:train_end]

    rows = []
    for entry_z in entry_grid:
        for exit_z in exit_grid:
            if exit_z >= entry_z:
                continue  # exit must be tighter than entry or a position never closes
            result = run_walk_forward(train, ticker_a, ticker_b,
                                       lookback=lookback, refit_every=refit_every,
                                       window=window, entry_z=entry_z, exit_z=exit_z)
            if result.empty:
                continue
            _, net_returns, n_trades = to_returns(result, train, ticker_a, ticker_b, cost_bps)
            stats = perf_stats(net_returns)
            stats.update({"entry_z": entry_z, "exit_z": exit_z, "Trades": n_trades})
            rows.append(stats)

    grid = pd.DataFrame(rows)
    if grid.empty:
        return 2.0, 0.5, grid  # fall back to the original defaults
    grid = grid.sort_values(metric, ascending=False).reset_index(drop=True)
    best = grid.iloc[0]
    return best["entry_z"], best["exit_z"], grid


if __name__ == "__main__":
    data = load_data()
    result = run_walk_forward(data)

    print(f"Trading days used (after {252}-day warmup): {len(result)}")
    print(f"Total P&L: {result['cumulative_pnl'].iloc[-1]:.2f}")
    print(f"Sharpe (daily, not annualized): {result['pnl'].mean() / result['pnl'].std():.3f}")
    result.to_csv(f"data/{TICKER_A}_{TICKER_B}_walk_forward_pnl.csv")
