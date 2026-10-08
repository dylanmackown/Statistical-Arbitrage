# Three ways to run it:
#   python backtest_report.py                 screen CANDIDATES, backtest the
#                                              single best pair (default thresholds)
#   python backtest_report.py V MA             skip screening, force a specific pair
#   python backtest_report.py --tune           best pair, but grid-search entry/exit
#                                              thresholds on a train split and report
#                                              performance out-of-sample on the rest
#   python backtest_report.py --portfolio      backtest EVERY pair that clears the
#                                              cointegration screen and report an
#                                              equal-weighted portfolio across them
#   python backtest_report.py --portfolio --tune   portfolio, each pair's thresholds
#                                                   tuned on its own train split

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pull_data import TICKER_A, TICKER_B, START_DATE, END_DATE, load_data
from screen_pairs import CANDIDATES, screen
from native_backtest import run_naive
from walk_forward_backtest import run_walk_forward, grid_search_thresholds
from performance import to_returns, perf_stats, alpha_beta, TRADING_DAYS, COST_BPS, RISK_FREE_RATE

BENCHMARK_TICKER = "SPY"
SIGNIFICANCE = 0.05      # p-value threshold for "actually cointegrated"
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.5
DEFAULT_TRAIN_END = "2023-06-30"   # ~65/35 train/test split over 2020-2025


# Pair selection

def select_best_pair(candidates=CANDIDATES, start=START_DATE, end=END_DATE,
                      alpha=SIGNIFICANCE):
    """Run the cointegration screen and return the most cointegrated pair.

    Returns (ticker_a, ticker_b, pvalue) for the lowest p-value in the
    candidate list. Prints a warning (but still proceeds) if even the best
    candidate doesn't clear the significance threshold.
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


# Single-pair report (unchanged behavior from before)

def build_report(ticker_a=None, ticker_b=None, cost_bps=COST_BPS,
                  candidates=CANDIDATES, entry_z=DEFAULT_ENTRY_Z, exit_z=DEFAULT_EXIT_Z):
    if ticker_a is None or ticker_b is None:
        ticker_a, ticker_b, _ = select_best_pair(candidates)

    data = load_data(ticker_a, ticker_b)

    naive_result = run_naive(data, ticker_a, ticker_b)
    wf_result = run_walk_forward(data, ticker_a, ticker_b, entry_z=entry_z, exit_z=exit_z)

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
          f"({eval_start.date()} to {eval_end.date()}, {cost_bps:.1f}bps cost, "
          f"entry_z={entry_z}, exit_z={exit_z}) ===")
    print(summary.to_string())

    summary.to_csv(f"data/{ticker_a}_{ticker_b}_performance_summary.csv")

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


# Threshold tuning: grid-search on train, report on held-out test only

def tuned_vs_default_report(ticker_a, ticker_b, cost_bps=COST_BPS,
                             train_end=DEFAULT_TRAIN_END):
    """Tune entry/exit thresholds on data up to `train_end`, then compare
    the tuned thresholds against the original defaults (2.0 / 0.5) on the
    held-out period AFTER train_end -- so any Sharpe improvement shown here
    is out-of-sample, not just curve-fit to the whole backtest.
    """
    data = load_data(ticker_a, ticker_b)

    best_entry, best_exit, grid = grid_search_thresholds(
        data, ticker_a, ticker_b, train_end=train_end, cost_bps=cost_bps)

    test_start = data.loc[train_end:].index[1]  # first day after the split
    test_end = data.index[-1]

    default_result = run_walk_forward(data, ticker_a, ticker_b,
                                       entry_z=DEFAULT_ENTRY_Z, exit_z=DEFAULT_EXIT_Z)
    tuned_result = run_walk_forward(data, ticker_a, ticker_b,
                                     entry_z=best_entry, exit_z=best_exit)

    _, default_net, default_trades = to_returns(
        default_result.loc[test_start:test_end], data, ticker_a, ticker_b, cost_bps)
    _, tuned_net, tuned_trades = to_returns(
        tuned_result.loc[test_start:test_end], data, ticker_a, ticker_b, cost_bps)

    rows = {
        f"Default thresholds (entry={DEFAULT_ENTRY_Z}, exit={DEFAULT_EXIT_Z})": perf_stats(default_net),
        f"Tuned thresholds (entry={best_entry}, exit={best_exit})": perf_stats(tuned_net),
    }
    summary = pd.DataFrame(rows).T
    summary["Trades"] = [default_trades, tuned_trades]

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(f"\n=== Threshold tuning: {ticker_a}/{ticker_b} "
          f"(trained through {train_end}, evaluated out-of-sample "
          f"{test_start.date()} to {test_end.date()}) ===")
    print("Top of grid search (ranked by train-period Sharpe):")
    print(grid.head(5).to_string(index=False))
    print()
    print(summary.to_string())

    return best_entry, best_exit, summary


# Portfolio report: every cointegrated pair, equal-weighted

def build_portfolio_report(candidates=CANDIDATES, alpha=SIGNIFICANCE, cost_bps=COST_BPS,
                            tune=False, train_end=DEFAULT_TRAIN_END):
    """Run the walk-forward strategy on every pair that clears the
    cointegration screen (not just the single best one) and combine them
    into an equal-weighted portfolio.

    A single pair's Sharpe is dominated by idiosyncratic noise. Running a
    basket of cointegrated pairs and equal-weighting them is the standard
    way a stat arb book raises portfolio-level Sharpe through diversification,
    without claiming any one pair individually has a bigger edge than it does.
    """
    screened = screen(candidates)
    qualifying = screened[screened["p_value"] < alpha]
    if qualifying.empty:
        print(f"[warning] No candidate cleared p < {alpha}; using the single "
              f"best candidate instead so the portfolio isn't empty.")
        qualifying = screened.iloc[:1]

    print(f"Pairs entering the portfolio (p < {alpha}):")
    print(qualifying.to_string(index=False))

    per_pair_returns = {}
    per_pair_stats = {}
    pair_data = {}
    for _, row in qualifying.iterrows():
        a, b = row["A"], row["B"]
        data = load_data(a, b)
        pair_data[(a, b)] = data

        entry_z, exit_z = DEFAULT_ENTRY_Z, DEFAULT_EXIT_Z
        if tune:
            entry_z, exit_z, _ = grid_search_thresholds(
                data, a, b, train_end=train_end, cost_bps=cost_bps)

        result = run_walk_forward(data, a, b, entry_z=entry_z, exit_z=exit_z)
        _, net_returns, n_trades = to_returns(result, data, a, b, cost_bps)
        per_pair_returns[f"{a}/{b}"] = net_returns
        per_pair_stats[f"{a}/{b}"] = {
            **perf_stats(net_returns), "Trades": n_trades,
            "entry_z": entry_z, "exit_z": exit_z,
        }

    returns_df = pd.DataFrame(per_pair_returns)
    # Equal-weight combine. Days before a pair's own walk-forward warmup ends
    # are treated as 0 (flat / no position) rather than dropping the row, so
    # one pair's longer warmup doesn't shrink the whole portfolio's history.
    portfolio_returns = returns_df.fillna(0).mean(axis=1)
    portfolio_returns = portfolio_returns.loc[returns_df.dropna(how="all").index]

    per_pair_stats["EQUAL-WEIGHT PORTFOLIO"] = {
        **perf_stats(portfolio_returns), "Trades": np.nan,
        "entry_z": np.nan, "exit_z": np.nan,
    }

    bh_combo = pd.concat(
        [naive_buy_and_hold(pair_data[(row["A"], row["B"])], row["A"], row["B"])
         for _, row in qualifying.iterrows()], axis=1
    ).mean(axis=1)
    per_pair_stats["50/50 buy & hold, averaged across same pairs"] = {
        **perf_stats(bh_combo.reindex(portfolio_returns.index)), "Trades": np.nan,
        "entry_z": np.nan, "exit_z": np.nan,
    }

    summary = pd.DataFrame(per_pair_stats).T
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(f"\n=== Portfolio performance across {len(qualifying)} cointegrated pair(s) "
          f"({cost_bps:.1f}bps cost, thresholds {'tuned per-pair' if tune else 'default'}) ===")
    print(summary.to_string())

    summary.to_csv("data/portfolio_performance_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in per_pair_returns.items():
        (1 + r.fillna(0)).cumprod().plot(ax=ax, alpha=0.4, label=name)
    (1 + portfolio_returns).cumprod().plot(ax=ax, color="black", linewidth=2.5,
                                            label="Equal-weight portfolio")
    ax.set_title("Per-pair strategies vs. equal-weighted portfolio (growth of $1)")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/portfolio_equity_curve.png")
    plt.close(fig)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairs trading backtest report.")
    parser.add_argument("pair", nargs="*", help="Optional explicit pair, e.g. V MA")
    parser.add_argument("--tune", action="store_true",
                         help="Grid-search entry/exit thresholds on a train split, "
                              "then report performance out-of-sample using the tuned thresholds.")
    parser.add_argument("--portfolio", action="store_true",
                         help="Backtest every pair that clears the cointegration screen "
                              "and report an equal-weighted portfolio, instead of one pair.")
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END,
                         help=f"Date splitting train (tuning) from test (reporting) data. "
                              f"Default {DEFAULT_TRAIN_END}.")
    args = parser.parse_args()

    if args.portfolio:
        build_portfolio_report(tune=args.tune, train_end=args.train_end)
    elif args.tune:
        if len(args.pair) == 2:
            ticker_a, ticker_b = args.pair
        else:
            ticker_a, ticker_b, _ = select_best_pair()
        best_entry, best_exit, _ = tuned_vs_default_report(
            ticker_a, ticker_b, train_end=args.train_end)
        # Also print the standard full-history report, using the tuned thresholds.
        build_report(ticker_a, ticker_b, entry_z=best_entry, exit_z=best_exit)
    elif len(args.pair) == 2:
        build_report(args.pair[0], args.pair[1])
    else:
        build_report()
