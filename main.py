import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","LT.NS","ITC.NS",
    "HINDUNILVR.NS","BAJFINANCE.NS","BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS","ULTRACEMCO.NS","TITAN.NS","NESTLEIND.NS","POWERGRID.NS",
    "NTPC.NS","ONGC.NS","COALINDIA.NS","ADANIENT.NS","ADANIPORTS.NS",
    "JSWSTEEL.NS","TATASTEEL.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
    "DRREDDY.NS","DIVISLAB.NS","CIPLA.NS","HEROMOTOCO.NS","BAJAJFINSV.NS",
    "GRASIM.NS","SHREECEM.NS","INDUSINDBK.NS","M&M.NS","TATAMOTORS.NS",
    "BPCL.NS","IOC.NS","GAIL.NS","EICHERMOT.NS","BRITANNIA.NS",
    "HDFCLIFE.NS","SBILIFE.NS","ICICIPRULI.NS","PIDILITIND.NS","UPL.NS",
    "DMART.NS","IRCTC.NS","TATACONSUM.NS","ADANIGREEN.NS","ADANIPOWER.NS",
    "SIEMENS.NS","ABB.NS","HAL.NS","BEL.NS","IRFC.NS"
]

end = pd.Timestamp.today()
start = end - pd.DateOffset(years=2)

data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
data = data.dropna(axis=1)

returns = data.pct_change().dropna()

ma50 = data.rolling(50).mean()
momentum = data / data.shift(63) - 1

signal = (data > ma50).shift(1)

vol = returns.rolling(20).std()

topN = 10
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

def sharpe(x):
    return (x.mean()/x.std()) * np.sqrt(252)

def max_dd(x):
    cum = (1+x).cumprod()
    peak = cum.cummax()
    return ((cum-peak)/peak).min()

print("\n=== ALPHA STRATEGY REPORT ===")
print("Annual Return:", round(port_returns.mean()*252, 3))
print("Volatility:", round(port_returns.std()*np.sqrt(252), 3))
print("Sharpe:", round(sharpe(port_returns), 3))
print("Max Drawdown:", round(max_dd(port_returns), 3))

plt.figure(figsize=(10,5))
plt.plot((1+port_returns).cumprod(), label="Alpha Strategy")
plt.title("Momentum + Trend + Volatility Weighting")
plt.legend()
plt.show()
