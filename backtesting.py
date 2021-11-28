import re
import pandas as pd
import datetime as dt
import numpy as np

import pandas_ta as ta
import ccxt
from datetime import datetime

from requests.utils import requote_uri

# Plotting graphs 
#import matplotlib.pyplot as plt 

#Import your data
# Collect the candlestick data from Binance. we must first create the Binance object that helps  us manage the requests to the exchange.
binance = ccxt.binance()
trading_pairs = ['BTC/USDT', 'ETH/BTC', 'BNB/BTC', 'ADA/BTC', 'XTZ/BTC']

#To store the final results
final=[]

for i in trading_pairs:
    #possibility to input several dataframes (1h, 4h, 1D)
    # candles_4h = binance.fetch_ohlcv(trading_pair, '4h')
    candles_1d = binance.fetch_ohlcv(str(i), '1d')

    #daily
    df_1d=pd.DataFrame(candles_1d, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    # df_1d['Date'] = pd.to_datetime(df_1d['Date'],unit='ms')
    print(df_1d)
    df_1d['Date'] = pd.to_datetime(df_1d['Date'],unit='ms',errors='coerce', utc=True)
    df_1d.set_index('Date', inplace=True, drop=True)

    # print(df_1d)
    # #Ichimoku calculations, we add one column to df_1d.
    # # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    high_prices = df_1d['High']
    low_prices = df_1d['Low']
    period26_high = high_prices.rolling(window=26).max()
    period26_low = low_prices.rolling(window=26).min()
    period9_high = high_prices.rolling(window=9).max()
    period9_low = low_prices.rolling(window=9).min()
    df_1d['kijun_sen'] = (period26_high + period26_low) / 2
    df_1d['tenkan_sen'] = (period9_high + period9_low) / 2
    df_1d['spread'] = ((df_1d['tenkan_sen']-df_1d['kijun_sen'])/df_1d['kijun_sen'])
    spread=0.1

    adx=df_1d.ta.adx(length=18)

    print("Daily ADX is", adx)

    #Calculation of the daily CMF
    cmf = df_1d.ta.cmf(length=20)
    #Conversion from a pandas series to a pandas dataframe
    cmf_frame=cmf.to_frame()

    print("df_1d is", df_1d)

    frames = pd.concat([df_1d, adx, cmf_frame], axis=1, join='inner')

    print("Here is is the concatenated dataframes", frames)

    # Take long positions 
    frames['long_positions1a'] = np.where(frames['spread'] < -spread, 1, 0)
    frames['long_positions1b']=np.where((frames['Close'] < 0.9*df_1d['kijun_sen']) & (frames['Close'] > 0.8*df_1d['kijun_sen']), 1, 0)
    frames['long_positions1c']=np.where((frames['Close'] < 0.8*df_1d['kijun_sen']) & (frames['Close'] > 0.7*df_1d['kijun_sen']), 1, 0)

    frames['positions_long']=frames['long_positions1a'] + frames['long_positions1b'] + frames['long_positions1c']
    print("Frame positions long is", frames['positions_long'])
    #No two consecutive trade
    frames['signal_long']=frames['positions_long'].diff()
    frames['positions2'] = np.where(frames['signal_long'] == 1, 1, 0)
    
    frames['positions_long2']=frames['positions_long']*frames['positions2'] #Filters on the frames['positions_long']

    #STOP-LOSS implementation
    frames['cor_price'] = frames['Close'].where((frames['signal_long'] == 1) & (frames['positions_long'] == 1), np.nan)
    print("cor_price is", frames['cor_price'])
    frames['cor_price'] = frames['cor_price'].ffill().astype(frames['Close'].dtype)
    print("cor_price is", frames['cor_price'])
    frames['diff_perc'] = (frames['Close'] - frames['cor_price']) / frames['cor_price']
    print("diff_price is", frames['cor_price'])
    #as long as we are below 5%, we keep the position ("1") otherwise we trash it ("0"), it is like a new mask. It can either be 0 or 1.
    #Frames for stop-losses. We add it to the total short posiitons, so "0" if there is nothing happening, "-1" when we have to sell.
    frames['positions3'] = np.where(frames['diff_perc'] <= -0.05, -1, 0)

    # Take short positions
    frames['short_positions1a'] = np.where(frames['spread'] > spread, -1, 0)
    #Selling around the equilibrium point (kijun), counter trend strategy.
    frames['short_positions1b'] = np.where((frames['Close'] < 1.02*df_1d['kijun_sen']) & (frames['Close'] > 0.98*df_1d['kijun_sen']), -1, 0)
    #Selling if the price (daily close) is too far from Kijun.
    frames['short_positions1c']=0#np.where((frames['close'] > 1.1*df_1d['kijun_sen']) & (frames['close'] < 1.2*df_1d['kijun_sen']), 1, 0)
    frames['short_positions1d']=np.where((frames['Close'] > 1.2*df_1d['kijun_sen']) & (frames['Close'] < 1.3*df_1d['kijun_sen']), 1, 0)
    
    frames['short_positions']=frames['short_positions1a'] + frames['positions3'] + frames['short_positions1b'] + frames['short_positions1c'] + frames['short_positions1d']
    print("Frame positions short is", frames['short_positions'])

    #No two consecutive trade
    frames['signal_short']=frames['short_positions'].diff()
    frames['positions4'] = np.where(frames['signal_short'] == -1, 1, 0)
    frames['short_positions2']=frames['short_positions']*frames['positions4'] #A mask not too have two consecutive selling

    frames['positions'] = frames['positions_long2'] + frames['short_positions2'] 

    print("Frame positions total is", frames['positions'])

    frames=frames.drop(['positions2', 'positions4'], axis = 1)
    # print("Here is is the frame with the positions column", frames)

    print("Here is is the frame with the positions column", frames)
    frames.to_csv('test.csv', sep='\t', encoding='utf-8')

    #buy and hold strategy = Calculate daily returns. 2 options:
    frames['Daily return'] = np.log(frames['Close']/frames['Close'].shift(1))
    #Alternative way:
    # frames['Daily return'] = frames['close'].pct_change()

    #STRATEGY RETURNS
    #Shift(1) to avoid the "forward looking bias".
    frames['Return'] = frames['Daily return'] * frames['positions'].shift(1) 

    # Calculate the cumulative returns:
    frames=frames[['Daily return', 'Return']].cumsum().apply(np.exp)

    print("Here is is the dataframes with all the returns", frames)

    #Profitability of each strategy
    calc=frames[['Daily return', 'Return']].mean()
    final.append(calc)

    print("Final results are", calc)

    #Volatility associated with each strategy
    vol=frames[['Daily return', 'Return']].std()

    print("Volatility are", vol)

print('Final results are:', final)
