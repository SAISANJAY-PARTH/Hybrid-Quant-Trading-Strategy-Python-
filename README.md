📈 Hybrid Quant Trading Strategy (Python)

A multi-asset, rule-based quantitative trading system that dynamically selects high-momentum stocks, applies trend confirmation, and constructs a risk-adjusted equity portfolio using volatility-based weighting.

This project demonstrates systematic portfolio design, signal engineering, and performance evaluation using Python.

🔹 Strategy Overview

The model combines three core components:

Momentum Ranking
Ranks 60 NSE stocks by 3-month relative performance.

Trend Filter
Trades only stocks above their 50-day moving average.

Risk-Adjusted Position Sizing
Allocates capital using inverse-volatility weighting to control risk.

Each day, the system selects the top 10 momentum stocks that satisfy the trend condition and constructs a dynamically rebalanced portfolio.

📊 Performance (Recent Backtest)
Metric	Value
Annualized Return	57.9%
Volatility	15.3%
Sharpe Ratio	3.86
Max Drawdown	-7.4%

Note: Results are based on historical data, exclude transaction costs, and are for research/educational purposes only.

⚙️ Tech Stack

Python

Pandas

NumPy

Matplotlib

yfinance

🚀 How It Works (Pipeline)

Download NSE stock data using yfinance

Compute momentum, moving averages, and volatility

Rank stocks by momentum

Filter by trend (price > MA50)

Apply volatility-based weights

Backtest portfolio performance

Evaluate risk-adjusted metrics
