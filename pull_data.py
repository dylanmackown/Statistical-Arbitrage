import os
import pandas as pd
import yfinance as yf

TICKER_A = "V"
TICKER_B = "MA"
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"
DATA_DIR = "data"


def _csv_path(ticker_a=TICKER_A, ticker_b=TICKER_B):
    return f"{DATA_DIR}/{ticker_a}_{ticker_b}.csv"


def download(ticker_a=TICKER_A, ticker_b=TICKER_B, start=START_DATE, end=END_DATE):
    """Download daily closes for both tickers and cache them to data/{A}_{B}.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = yf.download([ticker_a, ticker_b], start=start, end=end)["Close"].dropna()
    raw = raw[[ticker_a, ticker_b]]  # lock in column order regardless of yfinance sorting
    raw.index.name = "Date"
    raw.to_csv(_csv_path(ticker_a, ticker_b))
    print(f"Saved {len(raw)} rows to {_csv_path(ticker_a, ticker_b)}")
    return raw


def load_data(ticker_a=TICKER_A, ticker_b=TICKER_B, start=START_DATE, end=END_DATE):
    """Shared loader used by every other script in the pipeline.

    Reads the cached CSV if it exists; downloads it first if it doesn't,
    so you never have to remember to run this file before the others.
    """
    path = _csv_path(ticker_a, ticker_b)
    if not os.path.exists(path):
        return download(ticker_a, ticker_b, start, end)
    return pd.read_csv(path, index_col=0, parse_dates=True)[[ticker_a, ticker_b]]


if __name__ == "__main__":
    download()
