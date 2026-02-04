import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","LT.NS","ITC.NS"
]

end = pd.Timestamp.today()
start = end - pd.DateOffset(years=1)

data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
data = data.dropna(axis=1)

returns = data.pct_change().dropna()
ma50 = data.rolling(50).mean()
momentum = data / data.shift(63) - 1
signal = (data > ma50).shift(1)
vol = returns.rolling(20).std()

# ---------- SINGLE STOCK PERFORMANCE ----------
results = []

for stock in data.columns:
    strat_ret = returns[stock] * signal[stock]
    strat_ret = strat_ret.dropna()

    if len(strat_ret) == 0:
        continue

    ann_ret = strat_ret.mean() * 252
    sharpe = (strat_ret.mean()/strat_ret.std()) * np.sqrt(252)
    dd = ((1+strat_ret).cumprod() /
          (1+strat_ret).cumprod().cummax() - 1).min()

    results.append([stock, round(ann_ret,3), round(sharpe,3), round(dd,3)])

stock_report = pd.DataFrame(results, columns=["Stock","Annual Return","Sharpe","Max Drawdown"])
stock_report = stock_report.sort_values("Sharpe", ascending=False)

print("\n=== STRATEGY PERFORMANCE BY STOCK ===")
print(stock_report)

# ---------- PORTFOLIO VERSION ----------
topN = 5
portfolio = []
dates = signal.index[63:]

for d in dates:
    mom_rank = momentum.loc[d].dropna().sort_values(ascending=False)
    selected = mom_rank.head(topN).index

    sig = signal.loc[d, selected]
    rets = returns.loc[d, selected]
    vols = vol.loc[d, selected]

    sig = sig[sig == True]

    if len(sig) == 0:
        portfolio.append(0)
    else:
        inv_vol = 1 / vols[sig.index]
        weights = inv_vol / inv_vol.sum()
        portfolio.append((rets[sig.index] * weights).sum())

port_returns = pd.Series(portfolio, index=dates)

# ---------- METRICS ----------
def sharpe(x):
    return (x.mean()/x.std()) * np.sqrt(252)

def max_dd(x):
    cum = (1+x).cumprod()
    peak = cum.cummax()
    return ((cum-peak)/peak).min()

print("\n=== PORTFOLIO STRATEGY REPORT ===")
print("Annual Return:", round(port_returns.mean()*252, 3))
print("Volatility:", round(port_returns.std()*np.sqrt(252), 3))
print("Sharpe:", round(sharpe(port_returns), 3))
print("Max Drawdown:", round(max_dd(port_returns), 3))

# ---------- PLOT ----------
plt.figure(figsize=(10,5))
plt.plot((1+port_returns).cumprod(), label="Strategy Portfolio")
plt.title("Your Strategy – Real Portfolio Performance")
plt.legend()
plt.grid()
plt.show()
