# %%
import numpy as np
import pandas as pd
import yfinance
from pandas.tseries.offsets import BDay


tickers = ['^GSPC', '^IXIC', '^VIX', '^RUT', '^IRX', '^TYX']
start = '2008-01-01'
end = '2024-07-01'
end = None
periods = [0,1,2,3,4,5,45]
periods = 45
pre_periods_min = 0


if periods:
    try:
        period_max = max(periods)
    except TypeError:
        period_max = periods

# %%
start=pd.to_datetime(start)
end=pd.to_datetime(end)
start -= BDay(period_max+10)
if end:
    end += BDay(period_max+10)

# %%
Tickers = yfinance.Tickers(tickers)
prices = Tickers.download(start=start, end=end)
prices = prices['Open'].bfill()

# %%
try:
    post_periods = list(range(1, periods+1))
except TypeError:
    pass

# %%
pre_periods = np.maximum(post_periods, pre_periods_min)

# %%
dfs = {}
for pre, post in zip(pre_periods, post_periods):
    change = prices.pct_change(-post) - prices.pct_change(pre)
    dfs[post] = change

# %%
pd.concat(dfs, names=['Period', 'Ticker'], axis=1).to_parquet('../data/prices.parquet')