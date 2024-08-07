import pandas as pd
import yfinance
from pandas.tseries.offsets import BDay


def get_prices(symbols, periods, pre_period, start=None, end=None):

    try:
       symbols = pd.read_csv(symbols, header=None, index_col=0)
       symbols = symbols.to_dict()[1]
    except ValueError:
        try:
            symbols.keys()
        except AttributeError:
            symbols = dict(zip(symbols, symbols))
            
    if periods:
        try:
            period_max = max(periods)
        except TypeError:
            period_max = periods


    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    start -= BDay(period_max+10)
    if end:
        end += BDay(period_max+10)

    
    Tickers = yfinance.Tickers(list(symbols))
    prices = Tickers.download(start=start, end=end, progress=False)
    prices = prices['Open'].bfill()
    prices = prices.rename(columns=symbols)
    
    try:
        post_periods = list(range(1, periods+1))
    except TypeError:
        pass

    dfs = {}
    for post in post_periods:
        pre_change = prices.pct_change(pre_period)
        post_change = prices.pct_change(-post) 
        change = post_change - pre_change
        dfs[post] = change

    df = pd.concat(dfs, names=['Period', 'Ticker'], axis=1)
    df = df.dropna(how='all')
    
    symbols = pd.Series(range(len(symbols)), index=symbols.values())
    mapper = lambda index: symbols[index]
    df = df.sort_index(axis=1, level=1, key=mapper)

    return df