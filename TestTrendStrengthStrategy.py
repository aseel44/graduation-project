
import BitcoinDataReader as btc_reader
import datetime as dt
from backtesting import Backtest  
import pandas as pd
from backtesting import Strategy
import numpy as np
import talib as ta
import Indicators as indicators
import matplotlib.dates as mdates 
import matplotlib.pyplot as plt
import warnings
import AIModel_TPM as tpm
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

from_date = dt.datetime(2021,11,1,0,0,0)#Inclusive
to_date = dt.datetime(2023,2,1,0,0,0) #Exclusive 
run_date = dt.datetime(2022,1,1,0,0,0) #start of testing data
initial_cash = 100_000
out_dir = './CorrectiveAI/TestOutput'
write_data = True

use_corrective_ai  =True
minimum_trade_profatibility_thresh=0.3 #minimum trade profitability to consider and enter the market

log_trades = True

"""
as shown , ai risk (transformed and scaled trade profitability) is not constant (ex : 0.02) ; it changes according to the ai model predictions 
when trade risk(ai risk) < minimum_trade_profatibility_thresh we set it to 0 (skip and stay away from the market)


"""

def compute_pip_value(ep,sl,risk,equity):
    return (risk*equity)/abs(ep-sl)



class TradingStrategy(Strategy):
    
    bars = 0
    
    def locate_stoploss(self,signal):
        if signal ==-1:
            resistance = indicators.get_resistance(self.data.df[-1500:],self.argrel_window)
            #in rare cases
            if np.isnan(resistance):
                #print('nan resistance')
                return self.data.Close[-1] + self.atr[-1] * 3
            return resistance 
        elif signal==1:
            support = indicators.get_support(self.data.df[-1500:],self.argrel_window)
            #in rare cases
            if np.isnan(support):
                #print('nan support')
                return self.data.Close[-1] - self.atr[-1] * 3
            return support
    def get_trades_df(self):
        trades = self.closed_trades
        if len(trades)<1:
            return None
        
        trades_df = pd.DataFrame({
                'Size': [t.size for t in trades],
                'EntryBar': [t.entry_bar for t in trades],
                'ExitBar': [t.exit_bar for t in trades],
                'EntryPrice': [t.entry_price for t in trades],
                'ExitPrice': [t.exit_price for t in trades],
                'PnL': [t.pl for t in trades],# profit and loss (PnL) for each trade.
                'ReturnPct': [t.pl_pct for t in trades],#return percentage for each trade
                'EntryTime': [t.entry_time for t in trades],
                'ExitTime': [t.exit_time for t in trades],
            })
        trades_df['EntryTime'] = pd.to_datetime(trades_df['EntryTime'])
        trades_df.set_index("EntryTime",inplace=True,drop=False)
        del trades_df[trades_df.columns[0]]
        return trades_df
        
    def init(self):
        self.last_bar_date = self.data.index[-1]
        self.run_date=run_date
        self.timeframe=15
        self.argrel_window = 50
        
        self.out_dir = out_dir
        self.write_data = write_data

        indicators.compute_indicators(self.data.df)
        
        self.initial_equity = self._broker.equity

        self.risk = 0.01
        self.atr_len = 21
        self.atr = ta.ATR(self.data.High,self.data.Low,self.data.Close,timeperiod=self.atr_len) 

        if use_corrective_ai: 
            self.ai = tpm.TPM()
            self.ai = joblib.load('./CorrectiveAI/TrainOutput/tpm.ai')
            self.profit_probability_scalers= joblib.load('./CorrectiveAI/TrainOutput/scalers.sav')

        
    def predict_risk_ai(self):
        indicator_df = self.data.df[-1:]
        #predict last 2 instances
        df_ai = self.ai.predict_profit_probability(indicator_df , inplace=False)
          
        
        profit_prob_df = df_ai['profit_probability'].values.reshape(-1,1)
           
         
        X = self.profit_probability_scalers['uniform'].transform(profit_prob_df)
        X = self.profit_probability_scalers['robust'].transform(X)
        trade_risk_arr = self.profit_probability_scalers['minmax'].transform(X)
       
        trade_risk = trade_risk_arr[-1][0]
        
        self.data.df.loc[self.data.index[-1],'profit_propability']= df_ai['profit_probability'].iloc[-1]
        self.data.df.loc[self.data.index[-1],'trade_risk']= trade_risk

        if df_ai['profit_probability'].iloc[-1] < minimum_trade_profatibility_thresh:
            trade_risk = 0
        print(f'ai risk = {trade_risk}')
        return trade_risk

    def next(self):
        #start trading from run_date (we keep 2 months for calculating indicators)
        if self.data.index[-1]<self.run_date:
             return
        df = self.data.df
        signal = df['signal'][-1]
        #simple logic : 
        #if signal is 1 and no other buy positions (long positions) , then open buy position
        if signal==1 and not self.position.is_long:
            sl = self.locate_stoploss(signal=1)
            if use_corrective_ai:
                self.risk = self.predict_risk_ai()
                if self.risk==0:
                    self.position.close()
                    return
                
              
            size = round(compute_pip_value(self.data.Close[-1],sl,self.risk,self.initial_equity))
            
            if size<1:
                size = 1
             
            if log_trades:    
                print(f'{self.data.index[-1]}:opening buy position #price ={self.data.Close[-1]} , stoploss={sl} , size = {size} and equity = {self._broker.equity}')
            self.buy(sl=sl,size=size)
            
            
        #if signal is -1 and no other sell positions (short positions) , then open sell position
        elif signal==-1 and not self.position.is_short:
            sl = self.locate_stoploss(signal=-1)
            if use_corrective_ai:
                self.risk = self.predict_risk_ai()
                if self.risk==0:
                    self.position.close()
                    return
                
            size = round(compute_pip_value(self.data.Close[-1],sl,self.risk,self.initial_equity))
            
            if size<1:
                size = 1
              
            if log_trades:    
                print(f'{self.data.index[-1]}:opening sell position #price ={self.data.Close[-1]} , stoploss={sl} , size = {size} and equity = {self._broker.equity}')      
            self.sell(sl=sl,size=size)  
            

        if self.data.index[-1]==self.last_bar_date:
            if self.write_data:
                #1.save indicator data
                print(f'saving (df_indicator.csv) ...')
                self.data.df.to_csv(self.out_dir + "/df_indicator.csv")


                


def run_backtest():
    global out_dir
    if use_corrective_ai:

        print('using corrective ai to correct trade risk ...')
        out_dir = out_dir + '/WithCorrectiveAI'
    else:
        print('using fixed risk = 0.02 fr each trade ...')
        out_dir = out_dir + '/WithoutCorrectiveAI'   
    os.makedirs(out_dir,exist_ok=True)
    df = btc_reader.read_data(from_date,to_date)
    print('running backtest on date range {}-{} and initial equity = {}'.format(from_date,to_date,initial_cash))
    bt = Backtest(df, TradingStrategy, cash=initial_cash, commission=0,exclusive_orders=True,trade_on_close=False,margin=0.001)
    stats = bt.run()

    return stats

def clear_plot(plt):
    plt.clf()
    plt.legend('Guide',frameon=False)
    dtFmt = mdates.DateFormatter('%m/%Y') # define the formatting
    plt.gca().xaxis.set_major_formatter(dtFmt) 
    plt.xticks(rotation=45, fontweight='light',  fontsize='x-small',) 

if __name__=='__main__':
    stats = run_backtest()
    equity_curve = stats['_equity_curve']
    #print(equity_curve)
    
    tdf = stats['_trades']
    print(stats)
    plt.plot(equity_curve['Equity'],color = 'blue', linewidth = 1, label = 'equity curve')
    plt.savefig(out_dir+"/equity_curve",transparent=False,dpi=300,bbox_inches='tight')   
    clear_plot(plt)

    
    
    #print(tdf)
    tdf.to_csv(out_dir+"/trades.csv")
    stats.to_csv(out_dir+"/stats.csv")
    