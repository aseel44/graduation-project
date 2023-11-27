
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

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

from_date = dt.datetime(2018,1,1,0,0,0)#Inclusive
to_date = dt.datetime(2022,1,1,0,0,0) #Exclusive 
run_date = dt.datetime(2018,2,1,0,0,0)
initial_cash = 100_000
out_dir = './CorrectiveAI/TrainOutput'
write_data = True
train_ai_flg  = False
log_trades = True

def compute_pip_value(ep,sl,risk,equity):#هاد الفنكشن بحدد قيمة البيتكوين بالدولار
    return (risk*equity)/abs(ep-sl)

class TradingStrategy(Strategy):
    
    bars = 0
    
    def locate_stoploss(self,signal):#اذا الاشارة تساوي 1- يعني بيعو وبرح بجيب النقاط الحمراء
        if signal ==-1:
            resistance = indicators.get_resistance(self.data.df[-1500:],self.argrel_window)
            #in rare cases
            if np.isnan(resistance):
                #print('nan resistance')
                return self.data.Close[-1] + self.atr[-1] * 3
            return resistance 
        elif signal==1:#اذا الاشارة تساوي 1 يعني شراء وبروح بجيب النقاط الخضراء
            support = indicators.get_support(self.data.df[-1500:],self.argrel_window)
            #in rare cases
            if np.isnan(support):
                #print('nan support')
                return self.data.Close[-1] - self.atr[-1] * 3
            return support
    def get_trades_df(self):#هاد الفنكشن بعطينا الداتا الي للصفقة بس بطريقة ثانية
        trades = self.closed_trades
        if len(trades)<1:
            return None
        
        trades_df = pd.DataFrame({
                'Size': [t.size for t in trades],
                'EntryBar': [t.entry_bar for t in trades],
                'ExitBar': [t.exit_bar for t in trades],
                'EntryPrice': [t.entry_price for t in trades],
                'ExitPrice': [t.exit_price for t in trades],
                'PnL': [t.pl for t in trades],#الربح والخسارة بالصفقة
                'ReturnPct': [t.pl_pct for t in trades],#نسبة الربح والخسارة بالصفقة
                'EntryTime': [t.entry_time for t in trades],
                'ExitTime': [t.exit_time for t in trades],
            })
        trades_df['EntryTime'] = pd.to_datetime(trades_df['EntryTime'])
        trades_df.set_index("EntryTime",inplace=True,drop=False)
        del trades_df[trades_df.columns[0]]
        return trades_df
        
    def init(self):#هاد الفنكشن لنعرف المتغيرات الي عنا
        self.last_bar_date = self.data.index[-1]
        self.run_date=run_date
        self.timeframe=15
        self.argrel_window = 50
        
        self.out_dir = out_dir
        self.write_data = write_data

        indicators.compute_indicators(self.data.df)
        
        self.initial_equity = self._broker.equity

        self.risk = 0.02
        self.minimum_risk = 0.005
        self.maximm_risk = 0.03
        self.atr_len = 21
        self.atr = ta.ATR(self.data.High,self.data.Low,self.data.Close,timeperiod=self.atr_len)  
        self.ai = tpm.TPM()  

        self.profit_probability_scalers = {'uniform':QuantileTransformer(output_distribution='uniform'),
                         'normal':QuantileTransformer(output_distribution='normal'),
                         'robust': RobustScaler(quantile_range=(20, 80)),
                         'minmax':MinMaxScaler(feature_range=(self.minimum_risk, self.maximm_risk))
                         }
        
   #لندرب ال ااا اي بعد ما خلص الصفقات كلها
    def train_ai(self):
        
        #prepare and train
        indicator_df = self.data.df
        #remove all non signal instances 
        indicator_df.dropna(inplace=True)
        #only select signal (1,-1) data instances
        indicator_df=indicator_df[(indicator_df.signal==1) | (indicator_df.signal==-1)]

        #save trades data
        trades_df = self.get_trades_df()
        #train ai model
        self.ai.train(self.data.df,trades_df)

        #predict train
        df_ai = self.ai.predict_profit_probability(indicator_df , inplace=False)  
        #transfer shape of the output matrix  
        profit_prob_df = df_ai['profit_probability'] .values.reshape(-1,1)

        #fitting scalers
        X = self.profit_probability_scalers['uniform'].fit_transform(profit_prob_df)
        X = self.profit_probability_scalers['robust'].fit_transform(X)
        self.profit_probability_scalers['minmax'].fit_transform(X)

    def next(self):
        #start trading from run_date (we keep 2 months for calculating indicators)
        if self.data.index[-1]<self.run_date:
             return
        df = self.data.df
        signal = df['signal'][-1]
        #simple logic : 
        #if signal is 1 and no other buy positions (long positions) , then open buy position
        if signal==1 and not self.position.is_long:##اذا كان ما في اي صفقة شراء من قبل ,بفتح صفقة جديدة بس اول بحسب هاي الاشياء
            sl = self.locate_stoploss(signal=1)
            size = round(compute_pip_value(self.data.Close[-1],sl,self.risk,self.initial_equity))
            if size<1:
                size = 1
            if log_trades:    
                print(f'{self.data.index[-1]}:opening buy position #price ={self.data.Close[-1]} , stoploss={sl} , size = {size} and equity = {self._broker.equity}')
            self.buy(sl=sl,size=size)
            
            
        #if signal is -1 and no other sell positions (short positions) , then open sell position
        elif signal==-1 and not self.position.is_short:
            sl = self.locate_stoploss(signal=-1)
            size = round(compute_pip_value(self.data.Close[-1],sl,self.risk,self.initial_equity))
            if size<1:
                size = 1
            if log_trades:    
                print(f'{self.data.index[-1]}:opening sell position #price ={self.data.Close[-1]} , stoploss={sl} , size = {size} and equity = {self._broker.equity}')      
            self.sell(sl=sl,size=size)  
            

        if self.data.index[-1]==self.last_bar_date:
            if self.write_data:
                #save indicator data
                print(f'saving (df_indicator.csv) ...')
                self.data.df.to_csv(self.out_dir+"/df_indicator.csv")

                #training ai model
                if train_ai_flg:
                    print(f'training ai model ...')
                    self.train_ai()
                    print(f'saving ai model (tpm.ai) ...')
                    #dumping ai model (save it)
                    joblib.dump(self.ai,self.out_dir+"/tpm.ai" ,compress=3)
                    #dumping scalers (save it)
                    print(f'saving scalers (scalers.sav) ...')
                    joblib.dump(self.profit_probability_scalers,self.out_dir+"/scalers.sav" ,compress=3)
                    self.ai.dataset.to_csv(self.out_dir+"/train_dataset.csv")

def run_backtest():
    import os
    os.makedirs(out_dir,exist_ok=True)
    df = btc_reader.read_data(from_date,to_date)
    print('running backtest on date range {}-{} and initial equity = {}'.format(from_date,to_date,initial_cash))
    bt = Backtest(df, TradingStrategy, cash=initial_cash, commission=0,exclusive_orders=True,#انو اي صفقة رح ندخلها بتكنسل الصفقات الي قبل
                  trade_on_close=False,margin=0.001)
    stats = bt.run()

    return stats

def clear_plot(plt):
    plt.clf()
    plt.legend('Guide',frameon=False)
    dtFmt = mdates.DateFormatter('%m/%Y') # define the formatting
    plt.gca().xaxis.set_major_formatter(dtFmt) 
    plt.xticks(rotation=45, fontweight='light',  fontsize='x-small',) 

if __name__=='__main__':
   #pip_value = compute_pip_value(26800,26600,0.50,1000)
    #print(f'pip_value={pip_value}$')
    #quit()
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
    