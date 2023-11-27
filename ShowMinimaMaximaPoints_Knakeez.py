
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import argrelextrema
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import BitcoinDataReader as reader
import Indicators as indicators

from_date = dt.datetime(2018,1,1,0,0,0)#Inclusive
to_date = dt.datetime(2021,1,1,0,0,0) #Exclusive 
df= reader.read_data(from_date,to_date)

#plot_data = df[-17000:-15500]
plot_data = df[-400:]
w = 20

minima, maxima=indicators.get_min_max(plot_data,w)
lp = plot_data.Close.iloc[-1] 

my_support = indicators.get_support(plot_data,w)
my_resistance = indicators.get_resistance(plot_data,w)

s,r = [my_support,my_resistance]

# List of data points that fall under the minima category
min_points = [
    minima.loc[k] if k in minima.index else np.nan for k in plot_data.index]

# List of data points that fall under the maxima category
max_points = [
    maxima.loc[k] if k in maxima.index else np.nan for k in plot_data.index]

# Additional plots for marking the local minima and maxima
apd = [mpf.make_addplot(min_points, type='scatter', color="green", markersize=50),
       mpf.make_addplot(max_points, type='scatter', color="red", markersize=50)]


# Plot the OHLC data along with the local minima and maxima
mpf.plot(plot_data, type='candle', style='classic',
         addplot=apd, title='Local Minima and Maxima', figsize=(15, 7),hlines=dict(hlines=[s,lp,r],
                     colors=['g','y','r'],linestyle='-.'))





