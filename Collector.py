# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 08:59:50 2016

@author: jeremyfix
"""

#import numpy as np
import talib as t
import pandas as pd
import os
#import datetime as dt

def symbol_to_path(symbol, base_dir=os.path.join("data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPX=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPX and 'SPX' not in symbols:  # add SPX for reference, if absent
        symbols = ['SPX'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPX':  # drop dates SPX did not trade
            df = df.dropna(subset=["SPX"])
    return df


# MAIN
dates = pd.date_range('1990-01-01', '2016-06-06') 

# ADD PRICE DATA
prices = get_data(['SPX','VIX'], dates)
prices = prices.join(pd.read_csv("data/SPX.csv", index_col='Date', parse_dates=True, usecols=['Date','High','Low','Volume']))
prices = prices.join(pd.read_csv("data/GC.csv", index_col='Date', parse_dates=True, usecols=['Date','Settle']).rename(columns={'Settle': 'GOLD'}))
prices = prices.join(pd.read_csv("data/CL.csv", index_col='Date', parse_dates=True, usecols=['Date','Settle']).rename(columns={'Settle': 'OIL'}))
prices['Vol/AvgVol'] = prices['Volume'] / pd.rolling_mean(prices['Volume'],30)

# ADD ADVANCE and DECLINE DATA
prices = prices.join(pd.read_csv("data/ADV.csv", index_col='Date', parse_dates=True).rename(columns={'Numbers of Stocks': 'ADV'}))
prices = prices.join(pd.read_csv("data/DEC.csv", index_col='Date', parse_dates=True).rename(columns={'Numbers of Stocks': 'DEC'}))
prices['Adv/Dec'] = prices['ADV']/prices['DEC']

prices = prices.join(pd.read_csv("data/AVOL.csv", index_col='Date', parse_dates=True).rename(columns={'Numbers of Stocks': 'AVOL'}))
prices = prices.join(pd.read_csv("data/DVOL.csv", index_col='Date', parse_dates=True).rename(columns={'Numbers of Stocks': 'DVOL'}))
prices['AVol/DVol'] = prices['AVOL']/prices['DVOL']

# ADD INDICATORS
prices['RSI'] = t.RSI(prices['SPX'].values, timeperiod=3)
prices['ROC'] = t.ROC(prices['SPX'].values, timeperiod=14)
prices['MACD'], macdsignal, macdhist = t.MACD(prices['SPX'].values, fastperiod=12, slowperiod=26, signalperiod=9)
prices['ATR'] = t.ATR(prices['High'].values, prices['Low'].values, prices['SPX'].values, timeperiod=14)
prices['ADOSC'] = t.ADOSC(prices['High'].values, prices['Low'].values, prices['SPX'].values, prices['Volume'].values, fastperiod=3, slowperiod=10)

# ADD IN SEASONAL COMPONENTS like day of week, month
prices['DayOfWeek'] = prices.index.weekday
prices['Month'] = prices.index.month

# ADD MAX RETURN OVER PERIOD
period = 30
prices['DRMax'] = pd.rolling_max(prices['SPX'].shift(-period),period)
prices.ix[prices['DRMax'] > prices['SPX'], 'HiLo'] = 1
prices.ix[prices['DRMax'] == prices['SPX'], 'HiLo'] = 0
prices.ix[prices['DRMax'] < prices['SPX'], 'HiLo'] = -1

prices.dropna(subset=["VIX"],inplace=True)
prices.dropna(subset=["MACD"],inplace=True)

prices.to_csv('data/output.csv', index_label='DATE')

print prices.head(5)



