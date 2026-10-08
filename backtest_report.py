import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pull_data import TICKER_A, TICKER_B, START_DATE, END_DATE, load_data
from screen_pairs import CANDIDATES, screen
from native_backtest import run_naive
from walk_forward_backtest import run_walk_forward

TRADING_DAYS = 252
COST_BPS = 5.0          # one-way cost per $ notional traded, in basis points
RISK_FREE_RATE = 0.0    # annualized, used in Sharpe/Sortino
BENCHMARK_TICKER = "SPY"
SIGNIFICANCE = 0.05     # p-value threshold


# Pair selection

def select_best_pair(candidates=CANDIDATES, start=START_DATE, end=END_DATE,
                      alpha=SIGNIFICANCE):
    """Run the cointegration screen and return the most cointegrated pair.

    Returns (ticker_a, ticker_b, pvalue) for the lowest p-value in the
    candidate list. Prints a warning (but still proceeds) if even the best
    candidate doesn't clear the significance threshold -- that means the
    candidate list doesn't currently contain a statistically defensible
    pair, and whatever gets backtested should be treated as exploratory.
    """
    results = screen(candidates, start=start, end=end)
    best = results.iloc[0]
    if best["p_value"] > alpha:
        print(f"[warning] Best candidate {best['A']}/{best['B']} has "
              f"p-value {best['p_value']:.4f}, above the {alpha} "
              f"significance threshold. No candidate passed the "
              f"cointegration screen -- treat results with caution.\n"
              f"{results.to_string(index=False)}\n")
    else:
        print(f"Selected pair from screen: {best['A']}/{best['B']} "
              f"(p-value = {best['p_value']:.4f})\n"
              f"{results.to_string(index=False)}\n")
    return best["A"], best["B"], best["p_value"]


# Returns, not raw spread P&L

def to_returns(result, data, ticker_a=TICKER_A, ticker_b=TICKER_B, cost_bps=COST_BPS):
    """Convert a backtest's raw spread P&L into daily % returns.

    Divide by the gross notional required to hold the position 
    to get a return, then subtract a simple bps-of-notional
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


# Performance statistics

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


# Benchmarks

def naive_buy_and_hold(data, ticker_a=TICKER_A, ticker_b=TICKER_B):
    """Daily-rebalanced 50/50 buy-and-hold of the two legs."""
    returns = 0.5 * data[ticker_a].pct_change() + 0.5 * data[ticker_b].pct_change()
    return returns.dropna()


def fetch_benchmark(start, end, ticker=BENCHMARK_TICKER):
    """Best-effort SPY download. Returns None (with a warning) if unavailable
    so the rest of the report still runs without internet access."""
    try:
        import yfinance as yf
        spy = yf.download(ticker, start=start, end=end)["Close"].dropna()
        if isinstance(spy, pd.DataFrame):
            spy = spy[ticker]
        return spy.pct_change().dropna()
    except Exception as e:
        print(f"[warning] Could not fetch {ticker} benchmark ({e}). "
              f"Continuing with the 50/50 buy-and-hold benchmark only.")
        return None


# Report

def build_report(ticker_a=None, ticker_b=None, cost_bps=COST_BPS,
                  candidates=CANDIDATES):

    if ticker_a is None or ticker_b is None:
        ticker_a, ticker_b, _ = select_best_pair(candidates)

    data = load_data(ticker_a, ticker_b)

    naive_result = run_naive(data, ticker_a, ticker_b)
    wf_result = run_walk_forward(data, ticker_a, ticker_b)

    naive_gross, naive_net, naive_trades = to_returns(naive_result, data, ticker_a, ticker_b, cost_bps)
    wf_gross, wf_net, wf_trades = to_returns(wf_result, data, ticker_a, ticker_b, cost_bps)

    # Evaluate benchmarks over the walk-forward's live trading window only,
    # since that's the period the strategy could actually have traded
    # (the first ~252 days are a warmup with no walk-forward signal yet).
    eval_start, eval_end = wf_result.index[0], wf_result.index[-1]
    bh_returns = naive_buy_and_hold(data, ticker_a, ticker_b).loc[eval_start:eval_end]
    spy_returns = fetch_benchmark(eval_start, eval_end)
    if spy_returns is not None:
        spy_returns = spy_returns.loc[eval_start:eval_end]

    rows = {
        "Walk-fwd strategy (gross)": perf_stats(wf_gross),
        "Walk-fwd strategy (net)": perf_stats(wf_net),
        "Naive strategy (gross, look-ahead biased)": perf_stats(naive_gross),
        "50/50 buy & hold (benchmark)": perf_stats(bh_returns),
    }
    if spy_returns is not None:
        rows[f"{BENCHMARK_TICKER} buy & hold (benchmark)"] = perf_stats(spy_returns)

    summary = pd.DataFrame(rows).T
    summary["Trades"] = [wf_trades, wf_trades, naive_trades, np.nan] + (
        [np.nan] if spy_returns is not None else [])

    # Alpha/beta of the (tradeable) net walk-forward strategy vs. each benchmark
    alpha_bh, beta_bh, corr_bh = alpha_beta(wf_net, bh_returns)
    print(f"\nWalk-forward strategy (net) vs. 50/50 buy & hold:")
    print(f"  annualized alpha = {alpha_bh:+.2%}   beta = {beta_bh:.3f}   corr = {corr_bh:.3f}")

    if spy_returns is not None:
        alpha_spy, beta_spy, corr_spy = alpha_beta(wf_net, spy_returns)
        print(f"Walk-forward strategy (net) vs. {BENCHMARK_TICKER}:")
        print(f"  annualized alpha = {alpha_spy:+.2%}   beta = {beta_spy:.3f}   corr = {corr_spy:.3f}")

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(f"\n=== Performance summary: {ticker_a}/{ticker_b} "
          f"({eval_start.date()} to {eval_end.date()}, {cost_bps:.1f}bps cost) ===")
    print(summary.to_string())

    summary.to_csv(f"data/{ticker_a}_{ticker_b}_performance_summary.csv")

    # Equity curve chart
    fig, ax = plt.subplots(figsize=(10, 6))
    (1 + wf_gross).cumprod().plot(ax=ax, label="Walk-forward (gross)")
    (1 + wf_net).cumprod().plot(ax=ax, label=f"Walk-forward (net, {cost_bps:.0f}bps)")
    (1 + bh_returns).cumprod().plot(ax=ax, label="50/50 buy & hold", linestyle="--")
    if spy_returns is not None:
        (1 + spy_returns).cumprod().plot(ax=ax, label=f"{BENCHMARK_TICKER}", linestyle=":")
    ax.set_title(f"{ticker_a}/{ticker_b} pairs strategy vs. benchmarks (growth of $1)")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"data/{ticker_a}_{ticker_b}_equity_curve.png")
    plt.close(fig)

    return summary


if __name__ == "__main__":
    # Default: screen CANDIDATES and backtest whichever pair wins.
    #   python backtest_report.py
    # Override: force a specific pair and skip screening entirely.
    #   python backtest_report.py V MA
    if len(sys.argv) == 3:
        build_report(sys.argv[1], sys.argv[2])
    else:
        build_report()
