# cointegration_test.py
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from pull_data import TICKER_A, TICKER_B


data = pd.read_csv(f"data/{TICKER_A}_{TICKER_B}.csv", index_col=0, parse_dates=True)

score, pvalue, crit_values = coint(data[TICKER_A], data[TICKER_B])
print(f"Engle-Granger statistic: {score:.4f}")
print(f"p-value: {pvalue:.4f}")
print(f"Critical values (1%, 5%, 10%): {crit_values}")
print("Cointegrated" if pvalue < 0.05 else "Not cointegrated")

# coint() only gives you the test statistic — you still need the hedge
# ratio itself for the actual trading rule in step 5, so get that separately
X = sm.add_constant(data[TICKER_B])
hedge_ratio = sm.OLS(data[TICKER_A], X).fit().params[TICKER_B]
print(f"Hedge ratio: {hedge_ratio:.4f}")

spread = data[TICKER_A] - hedge_ratio * data[TICKER_B]
spread.to_csv(f"data/{TICKER_A}_{TICKER_B}_true_spread.csv")