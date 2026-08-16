import yfinance as yf
import os

os.makedirs("data", exist_ok=True)

TICKER_A = "V"
TICKER_B = "MA"

tickers = [TICKER_A, TICKER_B]  # Visa / Mastercard
data = yf.download(tickers, start="2020-01-01", end="2025-01-01")["Close"]

print(data.head())
print(data.shape)

data.to_csv(f"data/{TICKER_A}_{TICKER_B}.csv")