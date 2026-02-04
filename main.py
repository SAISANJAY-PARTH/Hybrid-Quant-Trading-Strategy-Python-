import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- UNIVERSE ----------------
TICKERS = [
"RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","LT.NS","ITC.NS",
"HINDUNILVR.NS","BAJFINANCE.NS","BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","ULTRACEMCO.NS","TITAN.NS","NESTLEIND.NS","POWERGRID.NS",
"NTPC.NS","ONGC.NS","COALINDIA.NS","ADANIENT.NS","ADANIPORTS.NS","JSWSTEEL.NS","TATASTEEL.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
"DRREDDY.NS","DIVISLAB.NS","CIPLA.NS","HEROMOTOCO.NS","BAJAJFINSV.NS","GRASIM.NS","SHREECEM.NS","INDUSINDBK.NS","M&M.NS","ETERNAL.NS",
"BPCL.NS","IOC.NS","GAIL.NS","EICHERMOT.NS","BRITANNIA.NS","HDFCLIFE.NS","SBILIFE.NS","ICICIPRULI.NS","PIDILITIND.NS","UPL.NS",
"DMART.NS","IRCTC.NS","TATACONSUM.NS","ADANIGREEN.NS","ADANIPOWER.NS","SIEMENS.NS","ABB.NS","HAL.NS","BEL.NS","IRFC.NS",
"PNB.NS","BANKBARODA.NS","CANBK.NS","IDFCFIRSTB.NS","FEDERALBNK.NS","AUBANK.NS","BANDHANBNK.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","MANAPPURAM.NS",
"JINDALSTEL.NS","SAIL.NS","HINDALCO.NS","VEDL.NS","NMDC.NS","AMBUJACEM.NS","ACC.NS","DALBHARAT.NS","SRF.NS","TATACHEM.NS",
"LUPIN.NS","BIOCON.NS","GLENMARK.NS","APOLLOHOSP.NS","FORTIS.NS","TORNTPHARM.NS","ALKEM.NS","AUROPHARMA.NS","ABBOTINDIA.NS","PAGEIND.NS"
]

# ---------------- DATA ----------------
end = pd.Timestamp.today()
start = end - pd.DateOffset(years=1)

data = yf.download(TICKERS, start=start, end=end,
                   auto_adjust=True, progress=False)["Close"]

data = data.dropna(axis=1)  # keep only stocks with full data

returns = data.pct_change().dropna()
ma50 = data.rolling(30).mean()
momentum = data / data.shift(42) - 1
signal = (data > ma50).shift(1)
vol = returns.rolling(20).std()

# ---------------- SINGLE STOCK PERFORMANCE ----------------
results = []

for stock in data.columns:
    strat_ret = returns[stock] * signal[stock]
    strat_ret = strat_ret.dropna()

    if len(strat_ret) < 50:
        continue

    ann_ret = strat_ret.mean() * 252
    shp = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    dd = ((1 + strat_ret).cumprod() /
          (1 + strat_ret).cumprod().cummax() - 1).min()

    results.append([stock, round(ann_ret,3), round(shp,3), round(dd,3)])

stock_report = pd.DataFrame(results,
            columns=["Stock","Annual Return","Sharpe","Max Drawdown"]) \
            .sort_values("Sharpe", ascending=False)

print("\n=== STRATEGY PERFORMANCE BY STOCK ===")
print(stock_report)

# ---------------- PORTFOLIO ----------------
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

# ---------------- MARKET BENCHMARK ----------------
market_returns = returns[data.columns].mean(axis=1)

# ---------------- METRICS ----------------
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

# ---------------- VISUALIZATION ----------------
strategy_equity = (1 + port_returns).cumprod()
market_equity = (1 + market_returns.loc[strategy_equity.index]).cumprod()

plt.figure(figsize=(11,6))
plt.plot(strategy_equity, label="My Strategy", linewidth=2)
plt.plot(market_equity, label="100-Stock Market Portfolio", linestyle="--", alpha=0.8)
plt.title("Strategy vs 100-Stock Market Portfolio")
plt.ylabel("Growth of ₹1")
plt.xlabel("Date")
plt.legend()
plt.grid()
plt.show()
