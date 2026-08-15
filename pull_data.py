import yfinance as yf
import os

os.makedirs("data", exist_ok=True)

tickers = ["KO", "PEP"]  # Coca-Cola / Pepsi
data = yf.download(tickers, start="2020-01-01", end="2025-01-01")["Close"]

print(data.head())
print(data.shape)

data.to_csv("data/ko_pep.csv")