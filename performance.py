import numpy as np
import pandas as pd

TRADING_DAYS = 252
COST_BPS = 5.0          # one-way cost per $ notional traded, in basis points
RISK_FREE_RATE = 0.0    # annualized, used in Sharpe/Sortino


def to_returns(result, data, ticker_a, ticker_b, cost_bps=COST_BPS):
    """Convert a backtest's raw spread P&L into daily % returns.

    Raw `pnl` is in spread price-points, which isn't comparable across
    pairs or annualizable on its own. We divide by the gross notional
    required to hold the position (long 1 unit of A, short hedge_ratio
    units of B) to get a return, then subtract a simple bps-of-notional
    transaction cost whenever the position changes.
    """
    notional = data[ticker_a] + result["hedge_ratio"].abs() * data[ticker_b]
    notional = notional.reindex(result.index)

    trade = result["position"].diff().abs().fillna(0)
    cost = (cost_bps / 1e4) * notional * trade

    gross_returns = result["pnl"] / notional.shift(1)
    net_returns = (result["pnl"] - cost) / notional.shift(1)

    n_trades = int(((result["position"] != 0) & (result["position"].shift(1) == 0)).sum())

    return gross_returns.dropna(), net_returns.dropna(), n_trades


def perf_stats(returns, freq=TRADING_DAYS, rf=RISK_FREE_RATE):
    returns = returns.dropna()
    if len(returns) < 2:
        return {k: np.nan for k in
                ["CAGR", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar", "WinRate"]}

    equity = (1 + returns).cumprod()
    n_years = len(returns) / freq
    total_return = equity.iloc[-1] - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    ann_vol = returns.std() * np.sqrt(freq)
    ann_mean = returns.mean() * freq
    sharpe = (ann_mean - rf) / ann_vol if ann_vol > 0 else np.nan

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(freq) if len(downside) > 1 else np.nan
    sortino = (ann_mean - rf) / downside_vol if downside_vol and downside_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    win_rate = (returns > 0).mean()

    return {
        "CAGR": cagr, "AnnVol": ann_vol, "Sharpe": sharpe, "Sortino": sortino,
        "MaxDD": max_dd, "Calmar": calmar, "WinRate": win_rate,
    }


def alpha_beta(strategy_returns, benchmark_returns, freq=TRADING_DAYS):
    df = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    df.columns = ["strategy", "benchmark"]
    if len(df) < 2 or df["benchmark"].var() == 0:
        return np.nan, np.nan, np.nan
    beta = df["benchmark"].cov(df["strategy"]) / df["benchmark"].var()
    alpha_daily = df["strategy"].mean() - beta * df["benchmark"].mean()
    alpha_annual = alpha_daily * freq
    corr = df["strategy"].corr(df["benchmark"])
    return alpha_annual, beta, corr
