# cointegration_test.py
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

data = pd.read_csv("data/ko_pep.csv", index_col=0, parse_dates=True)

score, pvalue, crit_values = coint(data["KO"], data["PEP"])
print(f"Engle-Granger statistic: {score:.4f}")
print(f"p-value: {pvalue:.4f}")
print(f"Critical values (1%, 5%, 10%): {crit_values}")
print("Cointegrated" if pvalue < 0.05 else "Not cointegrated")

# coint() only gives you the test statistic — you still need the hedge
# ratio itself for the actual trading rule in step 5, so get that separately
X = sm.add_constant(data["PEP"])
hedge_ratio = sm.OLS(data["KO"], X).fit().params["PEP"]
print(f"Hedge ratio: {hedge_ratio:.4f}")

spread = data["KO"] - hedge_ratio * data["PEP"]
spread.to_csv("data/ko_pep_true_spread.csv")