import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import numpy as np

class TPM: 
    
    dataset : pd.DataFrame
    scaler : MinMaxScaler
    
    seed = 42
    model = None
    
    printed = False
    def generate_features(self,df):
        
        #filter signals 
        df_signals = df[(df.signal==1) | (df.signal==-1)].copy()
        
        df_signals = df_signals[self.features]
        
        df_signals.dropna(inplace=True)
        
        return df_signals
    
    def prepare_data(self,indicator_df, trades_df):
        timeframe=15
        
        features_df = self.generate_features(indicator_df)
        features_df['Class'] = -1
        
        trades_df_index_shifted = trades_df.index - dt.timedelta(minutes=timeframe)
        trades_df.set_index(trades_df_index_shifted,inplace=True)
        
       
        #merge two dataframes (include all executed signals)
        df_new = pd.merge(features_df, trades_df, left_index=True, right_index=True, how='right',
                  suffixes=("", 'sal_'))
        df_new['Class'] = np.where(df_new.PnL>0,1,0)
        
        #drop all columns related to the trades
        class_index = df_new.columns.get_loc("Class")
        df_new = df_new.drop(df_new.iloc[:,class_index+1:],axis=1)
        
        #drop na values
        df_new.dropna(inplace=True)
        
        return df_new  
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        
        self.features = [
                    'close_returns_std_48',
                    'close_returns_std_96',
                    'close_returns_std_144',
                    'close_returns_std_240',
                    'close_returns_std_432',
                    'close_returns_skew_48',
                    'close_returns_skew_144',
                    'close_returns_skew_192',
                    'close_returns_skew_240',
                    'close_returns_skew_384',
                    'close_returns_skew_432',
                    'close_returns_kurt_144',
                    'close_returns_kurt_192',
                    'close_returns_kurt_240',
                    'close_returns_kurt_384',
                    'close_returns_kurt_432',
                    'target_vol_288',
                    'fasts2','slows2',
                    'fasts1','slows1',
                    'close_min_ratio',
                    'fasts',
                    'std_ewm_skew',
                    'std_ewm_kurt',
                    'std_ewm_kurt1',
                    'std_ewm_skew2',
                    'std_ewm_kurt2',
                    'slope1',
                    'slope2',
                    'slope3',
                    'close_returns_daily1',
                    'close_returns_daily2',

                    'rsi3',
                    'rsi4',
                    'rsi5',
                    'rsi6',
                    
                    'f171','f175',
                    ]
        
       
    def train(self,df_indicator , trades_df) -> bool : 
        
        model = RandomForestClassifier(
                                criterion='entropy',
                                n_estimators = 200,
                                bootstrap = False,
                                max_depth=35,
                                max_features =1,
                                min_samples_leaf = 5,
                                random_state = 42,class_weight='balanced'
                          )
        
        
        dataset = self.prepare_data(df_indicator,trades_df)

        
       
        target_class = dataset.columns[len(dataset.columns)-1]
        predictors_list = dataset.columns[:len(dataset.columns)-1].values
        
        X = dataset[predictors_list]
        y = dataset[target_class]
        
        X = self.scaler.fit_transform(X)
        
        self.model = model.fit(X, y)
        self.dataset = dataset
        
        return True
    
    def predict_profit_probability(self,inference_df, inplace=False)-> pd.DataFrame :
      
        if self.model is not None : 
            
            features_df = inference_df[self.features]
            
            X_test = features_df
            X_test = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test[:,:])
            
            
            y_pred_prop = self.model.predict_proba(X_test[:,:])
            
            if inplace:
                features_df['predicted_class'] = y_pred
                features_df['predicted_class_prop'] = np.max(y_pred_prop, axis=1)
                features_df['profit_probability'] = np.where(features_df['predicted_class']==1 , features_df['predicted_class_prop'] , 1-features_df['predicted_class_prop'])
                
                return features_df
            else:
                ndf = features_df.copy()
                ndf['predicted_class'] = y_pred
                ndf['predicted_class_prop'] = np.max(y_pred_prop, axis=1)
                ndf['profit_probability'] = np.where(ndf['predicted_class']==1 , ndf['predicted_class_prop'] , 1-ndf['predicted_class_prop'])
                
                return ndf
        
        else:
            raise ValueError('no trained model is avaliable') 
 