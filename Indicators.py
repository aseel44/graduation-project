from scipy.signal import argrelextrema
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt

def getAtrExtreme(df, atrPeriod=14, slowPeriod=30, fastPeriod=3):
        df = df.copy()
        atr = ta.ATR(df.High, df.Low, df.Close, timeperiod=atrPeriod)

        highsMean = ta.EMA(df.High, 5)
        lowsMean = ta.EMA(df.Low, 5)
        closesMean = ta.EMA(df.Close, 5)

        df['atrExtremes'] = np.where(df.Close > closesMean,
                               ((df.High - highsMean)/df.Close * 100) * (atr/df.Close * 100),
                               ((df.Low - lowsMean)/df.Close * 100) * (atr/df.Close * 100)
                               )

        df['fasts'] = ta.MA(df['atrExtremes'], fastPeriod)
        df['slows'] = ta.EMA(df['atrExtremes'], slowPeriod)
        df['extrems_std']= df['atrExtremes'].rolling(slowPeriod).std()

        return df['fasts'],df['slows'],df['extrems_std']

def get_min_max(data, argrel_window):

    # Use the argrelextrema to compute the local minima and maxima points
    local_min = argrelextrema(
        data.iloc[:-argrel_window]['Low'].values, np.less, order=argrel_window)[0]
    local_max = argrelextrema(
        data.iloc[:-argrel_window]['High'].values, np.greater, order=argrel_window)[0]

    # Store the minima and maxima values in a dataframe
    minima = data.iloc[local_min].Low
    maxima = data.iloc[local_max].High

    # Return dataframes containing minima and maxima values
    return minima, maxima
    
def get_support(df, argrel_window = 15):   
    price_data=df.copy()
    # Check whether the length of data is greater than the argel_window
    assert price_data.shape[0] > argrel_window, "Length of data is less then argel_window"        

    # Determine the indices of all support levels
    support_list = argrelextrema(
        price_data['Low'].values, np.less, order=argrel_window)[0]

    
    price_data['support'] = price_data.iloc[support_list, 2]

    # Fetch the last traded price
    ltp = price_data.Close.iloc[-1]
    
    # Get the nearest support level
    price_data['support'] = np.where(price_data['support'] < ltp, price_data['support'], np.nan)

    try:
        return price_data.loc[price_data.support.dropna().index.max(), 'support']    
    except:
        return np.nan

# Function to get the nearest resistance level
def get_resistance(df, argrel_window = 15):   
    price_data=df.copy()
    # Check whether the length of data is greater than the argel_window
    resistance_list = argrelextrema(
        price_data['High'].values, np.greater, order=argrel_window)[0]

    # Determine the indices of all resistance levels
    price_data['resistance'] = price_data.iloc[resistance_list, 1]       

    # Fetch the last traded price
    ltp = price_data.Close.iloc[-1]   
    
    # Get the nearest resistance level
    price_data['resistance'] = np.where(price_data['resistance'] > ltp, price_data['resistance'], np.nan)         

    try:
        return price_data.loc[price_data.resistance.dropna().index.max(), 'resistance']    
    except:
        return np.nan   
     
def compute_indicators(df):
    #tse parameters based on basic optimization
    close_returns_ema_len =96
    volatility_window=1344
    atr_len = 21
    cs_lookback = 38
    stats_window= 42
    cs_channel_sd_mul=0.73
    
    #df = df.copy()
    
    #Candle Strength Calculations
    df['up_bar'] =  ((df.High -df.Open)/(df.High-df.Low)) + ((df.Close -df.Open)/(df.High-df.Low))
    df['down_bar'] =  ((df.Low -df.Open)/(df.High-df.Low)) + ((df.Close -df.Open)/(df.High-df.Low))
    
    conditions = [
              (df.High==df.Low),
              (df.Close>df.Open),
              (df.Close<=df.Open)
              
        ]
    
    values = [0,
                df['up_bar'],
                df['down_bar']
              ]
    df['cs'] = np.select(conditions,values)

    #Sum of candle strengths over cs_lookback period (sum of cs for 38 Bar) 
    df['cs_sum'] = df['cs'].rolling(window=cs_lookback).sum()
    
    #Min cs sum over 38 bar
    df['lch']=df['cs_sum'].shift(1).rolling(cs_lookback).min()
    #Max cs sum over 38 bar
    df['uch']=df['cs_sum'].shift(1).rolling(cs_lookback).max()
    #Mean and Standard Deviation for the cs sum channel
    df['mp']  = df.loc[df['cs_sum'] >= 0, 'cs_sum'].rolling(stats_window).mean()
    df['mn']  = df.loc[df['cs_sum'] <  0, 'cs_sum'].rolling(stats_window).mean() 
    df['sdp'] = df.loc[df['cs_sum'] >= 0, 'cs_sum'].rolling(stats_window).std()
    df['sdn'] = df.loc[df['cs_sum'] <  0, 'cs_sum'].rolling(stats_window).std()
    
    df.fillna(method='ffill', inplace=True)
    
    #Constructing channel based on the calculated mean and cs_channel_sd_mul sd
    df['uf'] = df['mp'] + cs_channel_sd_mul*df['sdp']
    df['lf'] = df['mn'] - cs_channel_sd_mul*df['sdn']
    
    #strength signals : 
    #if current cs_sum (strength for the last 38 candle) 
    #   breaks the channel to the upper side -> this means buyers power and hence the signal is 1
    #if current cs_sum (strength for the last 38 candle) 
    #   breaks the channel to the down side -> this means sellers power and hence the signal is -1
    conditions = [
              (df["uch"]<df["cs_sum"]) & (df['uch']>=df['uf']),
              (df["lch"]>df["cs_sum"]) & (df['lch']<=df['lf'])    
        ]
    
    values = [1,-1]
    df['signal'] = np.select(conditions,values,default=0)
    
    #close ratio indicator
    df['close_min_ratio']=df['Close'] / df['Close'].rolling(close_returns_ema_len).min()
    
    close_ret_windows = [int(48*x) for x in range(1,10,1)]
    
    #more close returns important indicators to be used for training AI 
    for w in close_ret_windows:
        df[f'close_returns_{w}'] = df['Close'].pct_change(w)*100
        df[f'close_returns_skew_{w}'] = df[f'close_returns_{w}'].rolling(close_returns_ema_len).skew()
        df[f'close_returns_kurt_{w}'] = df[f'close_returns_{w}'].rolling(close_returns_ema_len).kurt()
        df[f'close_returns_std_{w}']= df[f'close_returns_{w}'].ewm(span=close_returns_ema_len,adjust=True).std()

    #volitility indicators   
    df['fasts1'],df['slows1'],df['extrems_std1'] = getAtrExtreme(df,atr_len,14,3)
    df['fasts2'],df['slows2'],df['extrems_std2'] = getAtrExtreme(df,atr_len,7,3)
    df['fasts'],df['slows'],df['extrems_std'] = getAtrExtreme(df,atr_len,48,12)
    
    #adx indicators 
    df['adx200'] = ta.ADXR(df.High,df.Low,df.Close, timeperiod=200)
    
    #more other common indicators 
    df['atr'] = ta.ATR(df.High, df.Low, df.Close, timeperiod=atr_len)
    pre_atr = df['atr'].shift(1)
    df['atr_change'] = np.where(pre_atr == 0, 0, np.log(df['atr'] / pre_atr))
    
    df['std_ewm'] = df['atr_change'].ewm(span=atr_len-1,adjust=True).std() 
    df['std_ewm_skew'] = df['std_ewm'].rolling(close_returns_ema_len).skew()
    df['std_ewm_kurt'] = df['std_ewm'].rolling(close_returns_ema_len).kurt()
    
    N1 = max(int(close_returns_ema_len/2),5)
    N2 = max(int(close_returns_ema_len/10),5)
    #df['std_ewm_skew1'] = df['std_ewm'].rolling(N1).skew()
    df['std_ewm_kurt1'] = df['std_ewm'].rolling(N1).kurt()
    
    df['std_ewm_skew2'] = df['std_ewm'].rolling(N2).skew()
    df['std_ewm_kurt2'] = df['std_ewm'].rolling(N2).kurt()
    
    df['target_vol'] =  df['std_ewm'].rolling(volatility_window).median()
    df['target_vol_288'] =  df['std_ewm'].rolling(288).median()
    
    df['leverage'] = df['target_vol']  / df['close_returns_std_48']
    
    df['slope1']  = ta.LINEARREG_SLOPE(df.Close, timeperiod=24)
    df['slope2']  = ta.LINEARREG_SLOPE(df.Close, timeperiod=48)
    df['slope3']  = ta.LINEARREG_SLOPE(df.Close, timeperiod=96)
    
    df['close_returns_daily1'] = df['Close'].pct_change(672)*100
    df['close_returns_daily2'] = df['Close'].pct_change(672*2)*100
    
    df['rsi3'] = ta.RSI(df.Close, timeperiod=96)
    df['rsi4'] = ta.RSI(df.Close, timeperiod=200)
    df['rsi5'] = ta.RSI(df.Close, timeperiod=672)
    df['rsi6'] = ta.RSI(df.Close, timeperiod=672*2)
    
    #volume indicators
    df['f051'] = df.Volume.pct_change()
    df['f055'] = df.Volume.pct_change(144)
    df['f171'] = df['f051'].apply(np.sign)
    df['f175'] = df['f055'].apply(np.sign)
    
    return df  


if __name__=='__main__':
    import BitcoinDataReader as btc_reader
    import datetime as dt
    from_date = dt.datetime(2017,1,1,0,0,0)#Inclusive
    to_date = dt.datetime(2022,1,1,0,0,0) #Exclusive 
    df = btc_reader.read_data(from_date,to_date)
    df = compute_indicators(df)
    print(df)

    plot_df = df.iloc[-3000:]
    #columns_to_plot = ['Close','extrems_std','close_returns_std_48','std_ewm','fasts','slows','rsi3']
    columns_to_plot = ['atr']

    plot_df[columns_to_plot].plot(subplots=True, figsize=(16, 12), grid=True)

    plt.show()