import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from pull_data import TICKER_A, TICKER_B, load_data


def compute_hedge_ratio(y, x):
    """OLS hedge ratio: y = alpha + beta * x + eps. Returns beta.
    """
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit().params.iloc[1]


def test_cointegration(data, ticker_a=TICKER_A, ticker_b=TICKER_B):
    score, pvalue, crit_values = coint(data[ticker_a], data[ticker_b])
    hedge_ratio = compute_hedge_ratio(data[ticker_a], data[ticker_b])
    return {
        "score": score,
        "pvalue": pvalue,
        "crit_values": crit_values,
        "hedge_ratio": hedge_ratio,
    }


if __name__ == "__main__":
    data = load_data()
    result = test_cointegration(data)

    print(f"Engle-Granger statistic: {result['score']:.4f}")
    print(f"p-value: {result['pvalue']:.4f}")
    print(f"Critical values (1%, 5%, 10%): {result['crit_values']}")
    print("Cointegrated" if result["pvalue"] < 0.05 else "Not cointegrated")
    print(f"Hedge ratio: {result['hedge_ratio']:.4f}")

    spread = data[TICKER_A] - result["hedge_ratio"] * data[TICKER_B]
    spread.to_csv(f"data/{TICKER_A}_{TICKER_B}_true_spread.csv")
