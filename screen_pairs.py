import pandas as pd
from statsmodels.tsa.stattools import coint

from pull_data import load_data, START_DATE, END_DATE

CANDIDATES = [
    ("V", "MA"),      # Visa / Mastercard
    ("HD", "LOW"),    # Home Depot / Lowe's
    ("XOM", "CVX"),   # Exxon / Chevron
    ("JPM", "BAC"),   # JPMorgan / Bank of America
    ("KO", "PEP"),    # Coca-Cola / Pepsi - control
]


def screen(candidates=CANDIDATES, start=START_DATE, end=END_DATE):
    """Run the Engle-Granger cointegration test over a list of (A, B) candidate pairs.

    Uses the same cached loader as the rest of the pipeline (pull_data.load_data),
    so every candidate's data/{A}_{B}.csv ends up cached under data/ exactly like
    a single-pair run would -- nothing has to be re-downloaded later if that pair
    turns out to be the one carried forward into the backtest.

    Returns a DataFrame sorted by p-value (most cointegrated first).
    """
    results = []
    for a, b in candidates:
        data = load_data(a, b, start, end)
        score, pvalue, _ = coint(data[a], data[b])
        results.append((a, b, pvalue))
    return pd.DataFrame(results, columns=["A", "B", "p_value"]).sort_values("p_value")


if __name__ == "__main__":
    print(screen())
