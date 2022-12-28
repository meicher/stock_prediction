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


########################################################################################################
# DATA PULL & REFRESH FUNCTIONS #
########################################################################################################

#GET SHARADAR EQUITY PRICES (SEP) FOR 2017 THRU PRESENT, STORE FILE
def sharadarSEP():
    
    sep = pd.read_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_SEP.csv')
    sep = sep[['ticker','date','closeadj']].copy()

    # CHECK FOR NEW DATA, APPEND IF NEW, AND OVERWRITE CSV IF NEW.
    if  (lastdate.date() - pd.to_datetime(sep['date'].max()).date()).days > 0:
        print('New Data---')
        
        septoday = quandl.get_table('SHARADAR/SEP',date=lastdate,
                         paginate=True)
        sep = sep.append(septoday[['ticker','date','closeadj']])
        sep['date'] = pd.to_datetime(sep['date'])
    
        sep.to_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_SEP.csv',index=False)
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
    
    daily = pd.read_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_DAILY.csv')
    
    # CHECK FOR NEW DATA, APPEND IF NEW, AND OVERWRITE CSV IF NEW.
    if  (lastdate.date() - pd.to_datetime(daily['date'].max()).date()).days > 0:
        print('New Data---')
        
        dailytoday = quandl.get_table('SHARADAR/DAILY',date=lastdate,paginate=True)
        daily = daily.append(dailytoday)
        daily['date'] = pd.to_datetime(daily['date'])
    
        daily.to_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_DAILY.csv',index=False)
    else:
        print('Data up to date')
        
    print(daily['date'].max())
    
    
def nasdaqRTAT():
    
    rtat = pd.read_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/NDAQ_RTAT.csv')
    
    # CHECK FOR NEW DATA, APPEND IF NEW, AND OVERWRITE CSV IF NEW.
    if  (lastdate.date() - pd.to_datetime(rtat['date'].max()).date()).days > 0:
        print('New Data---')
        
        rtat_today = quandl.get_table('NDAQ/RTAT', date=lastdate,paginate=True)
        rtat = rtat.append(rtat_today)
        rtat['date'] = pd.to_datetime(rtat['date'])
    
        rtat.to_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/NDAQ_RTAT.csv',index=False)
    else:
        print('Data up to date:')
        
    print(rtat['date'].max())

    
########################################################################################################
# PROCESSING / FEATURE CREATION FUNCTIONS #
########################################################################################################
    
def fts_rtat(df):
    # want to get single dataset features calculated here (i.e. rolling metrics, any within data metrics)
    # USE A STANDARDIZED FEATURE CODING (datasource_OGMetric_generatedmetric_timewindow)
    # sector/industry market wide metrics (after getting equities bundle)
    
    #TICKER BASED WINDOW METRICS @ 5, 15, 30
    df['activity_5'] = df.rolling(5).mean()['activity']
    df['sentiment_5'] = df.rolling(5).mean()['sentiment']
    df['activity_15'] = df.rolling(15).mean()['activity']
    df['sentiment_15'] = df.rolling(15).mean()['sentiment']
    df['activity_30'] = df.rolling(30).mean()['activity']
    df['sentiment_30'] = df.rolling(30).mean()['sentiment']
    df['activity_recent_ratio'] = df['activity_5'] / df['activity_30']
    df['sentiment_recent_ratio'] = df['sentiment_5'] / df['sentiment_30']
    
#     #MARKET WIDE (DATE BASED) WINDOW METRICS -- CAN USE TO ADJUST
#     df = df.join(df.groupby('date').mean()[['sentiment','activity']].rolling(5).mean(),on='date',rsuffix='_5_mkt')
#     df = df.join(df.groupby('date').mean()[['sentiment','activity']].rolling(15).mean(),on='date',rsuffix='_15_mkt')
#     df = df.join(df.groupby('date').mean()[['sentiment','activity']].rolling(30).mean(),on='date',rsuffix='_30_mkt')
    
    #CREATE METRIC THAT COMBINED ACTIVITY + SENTIMENT (ADD # TO ACTIVITY TO PREVENT 0)
    df['prod_sent_act'] = (df['activity']+.00001)*df['sentiment']*100
    df['prod_sent_act_5'] = (df['activity_5']+.00001)*df['sentiment_5']*100
    df['prod_sent_act_15'] = (df['activity_15']+.00001)*df['sentiment_15']*100
    df['prod_sent_act_30'] = (df['activity_30']+.00001)*df['sentiment_30']*100
    
    # ADD Z SCORES FOR METRICS BY TICKER?? SIMILAR TO VOLATILITY FOR ACTIVITY/SENTIMENT
    
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df
    