import nasdaqdatalink
import os
import json
import quandl
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime

#store my API key
with open('C:/Users/meich/.nasdaq/data_link_apikey.json') as f:
    data=json.load(f)
    key=data['api_key']
quandl.ApiConfig.api_key = key

#Get latest market date
nyse = mcal.get_calendar('NYSE')
lastdate = datetime.today().date() - pd.tseries.offsets.CustomBusinessDay(1, holidays = nyse.holidays().holidays)


#GET SHARADAR EQUITY PRICES (SEP) FOR 2017 THRU PRESENT, STORE FILE
def sharadarSEP():
    sep = pd.read_csv('Data/SHARADAR_SEP_2017_TODAY.csv')
    sep = sep[['ticker','date','closeadj']].copy()

    # CHECK FOR NEW DATA, APPEND IF NEW, AND OVERWRITE CSV IF NEW.
    if  (lastdate.date() - pd.to_datetime(sep['date'].max()).date()).days > 0:
        print('New Data---')
        septoday = quandl.get_table('SHARADAR/SEP',date=lastdate,
                         paginate=True)
        sep = sep.append(septoday[['ticker','date','closeadj']])
        sep['date'] = pd.to_datetime(sep['date'])
    
        sep.to_csv('Data/SHARADAR_SEP_2017_TODAY.csv',index=False)
    else:
        print('Data up to date')
        
    print(sep['date'].max())

#GET FILTERED TICKERS (USED FOR PASSING TO OTHER CALLS & BASIC STOCK CATEGORICAL INFO)
def sharadarTICKERS():
    
    tickers = quandl.get_table('SHARADAR/TICKERS',paginate=True)
    
    # limit to common stock, may need to change this at some point
    tickers = tickers[tickers['category'].str.contains('Common',na=False)]
    tickers = tickers[~tickers['industry'].str.contains('Shell Companies',na=False)]
    
    #limit to last price date after 2016
    tickers = tickers[tickers['lastpricedate'] > '2017-03-01']
    
    #limit to nyse / nasdaq for now
    tickers = tickers[tickers['exchange'].isin(['NYSE','NASDAQ'])]
    
    #limit to real tickers w/o period
    tickers = tickers[~tickers['ticker'].str.contains('\.')]
    
    #tickers are duplicated by table for some reason -- get the most recent from SEP
    tickers = tickers[tickers['table'] == 'SEP']
    
    filtered_tickers = tickers[['ticker','isdelisted','sector','industry','location','category']].copy()
    
    filtered_tickers['location'] = filtered_tickers['location'].str.split(';').str[-1].str.strip().str.replace('.','')

    return filtered_tickers

def sharadarDAILY():
    
    quandl.get_table('SHARADAR/DAILY',date=lastdate)