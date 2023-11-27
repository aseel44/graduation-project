import pandas as pd
import datetime as dt

#to disable python warnnings 
pd.options.mode.chained_assignment = None  # default='warn'

#This is method to read 15 Min BTC data into data frame 
def read_data(from_date,to_date = None):
    #read data as plain csv data into dataframe
    df = pd.read_csv('./CorrectiveAI/btc_data_2018_2023.csv')
   
    #convert string date column to pandas datetime object 
    df = df.apply(lambda col: pd.to_datetime(col, errors='ignore') 
                  if col.dtypes == object 
                  else col, 
                  axis=0)
    #set index to Date 
    df.set_index("Date",inplace=True,drop=True)
    #sort index (Date) asc (oldest candles to newest candles)
    df.sort_index(axis=0,inplace=True)
    
    #filter data and include only data that belongs to the given input date ranges
    if from_date is not None and to_date is not None:
        df = df[(df.index>=from_date) & (df.index<to_date)]
    elif from_date is not None:
        df = df[(df.index >= from_date)]
    
    #drop na records if exists
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    from_date = dt.datetime(2018,1,1,0,0,0)#Inclusive
    to_date = dt.datetime(2022,1,1,0,0,0) #Exclusive 

    train_data = read_data(from_date,to_date) #Read train data from 1/2017 to 1/2022
    test_data = read_data(to_date) #Read test data from 1/2022 to 1/2023

    print(test_data)