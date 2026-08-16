# screen_pairs.py
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint

candidates = [
    ("V", "MA"),     # Visa / Mastercard
    ("HD", "LOW"),    # Home Depot / Lowe's
    ("XOM", "CVX"),   # Exxon / Chevron
    ("JPM", "BAC"),   # JPMorgan / Bank of America
    ("KO", "PEP"),    # keep as a control — you already know this one fails
]

results = []
for a, b in candidates:
    data = yf.download([a, b], start="2020-01-01", end="2025-01-01")["Close"].dropna()
    score, pvalue, _ = coint(data[a], data[b])
    results.append((a, b, pvalue))

results_df = pd.DataFrame(results, columns=["A", "B", "p_value"]).sort_values("p_value")
print(results_df)