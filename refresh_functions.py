import nasdaqdatalink
import os
import time
import json
import quandl
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import requests
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


#store my API key
with open('C:/Users/meich/.nasdaq/data_link_apikey.json') as f:
    data=json.load(f)
    key=data['api_key']
quandl.ApiConfig.api_key = key

#Get latest market date
nyse = mcal.get_calendar('NYSE')
lastdate = datetime.today().date() - pd.tseries.offsets.CustomBusinessDay(1, holidays = nyse.holidays().holidays)

# Creating a funtion that will measure execution time
def timeit(method):
    # Provide time elapsed for function to complete
    def timed(*args, **kw):
        # define time paramters
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r %2.2f mins' %                  
            (method.__name__, (((te - ts) * 1000))/60000))
        return result
    #add this to maintain availability of docstrings w/ .__doc__
    timed.__doc__ = "{}".format(
        str(method.__doc__)
    )
    return timed


########################################################################################################
# DATA PULL & REFRESH FUNCTIONS #
########################################################################################################

#GET SHARADAR EQUITY PRICES (SEP) FOR 2017 THRU PRESENT, STORE FILE
@timeit
def sharadarSEP(date=lastdate):
    
    sep = pd.read_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_SEP.csv')
    sep = sep[['ticker','date','closeadj']].copy()

    # CHECK FOR NEW DATA, APPEND IF NEW, AND OVERWRITE CSV IF NEW.
    if  (lastdate.date() - pd.to_datetime(sep['date'].max()).date()).days > 0:
        print('New Data---')
        
        septoday = quandl.get_table('SHARADAR/SEP',date=date,
                         paginate=True)
        sep = sep.append(septoday[['ticker','date','closeadj']])
        sep['date'] = pd.to_datetime(sep['date'])
    
        sep.to_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_SEP.csv',index=False)
    else:
        print('Data up to date')
        
    print(sep['date'].max())

#GET FILTERED TICKERS (USED FOR PASSING TO OTHER CALLS & BASIC STOCK CATEGORICAL INFO)
@timeit
def sharadarTICKERS():
    
    tickers = quandl.get_table('SHARADAR/TICKERS',paginate=True)
    
    # limit to common stock, may need to change this at some point
    tickers = tickers[tickers['category'].str.contains('Common',na=False)]
    tickers = tickers[~tickers['industry'].str.contains('Shell Companies',na=False)]
    
    #limit to last price date after 2016
    tickers = tickers[tickers['lastpricedate'] > '2017-01-01']
    
    #limit to nyse / nasdaq for now
    tickers = tickers[tickers['exchange'].isin(['NYSE','NASDAQ'])]
    
    #limit to real tickers w/o period
    tickers = tickers[~tickers['ticker'].str.contains('\.')]
    
    #tickers are duplicated by table for some reason -- get the most recent from SEP
    tickers = tickers[tickers['table'] == 'SEP']
    
    filtered_tickers = tickers[['ticker','isdelisted','sector','industry','location','category']].copy()
    filtered_tickers['location'] = filtered_tickers['location'].str.split(';').str[-1].str.strip().str.replace('.','')

    return filtered_tickers

@timeit
def sharadarDAILY(date=lastdate):
    
    daily = pd.read_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_DAILY.csv')
    
    # CHECK FOR NEW DATA, APPEND IF NEW, AND OVERWRITE CSV IF NEW.
    if  (lastdate.date() - pd.to_datetime(daily['date'].max()).date()).days > 0:
        print('New Data---')
        
        dailytoday = quandl.get_table('SHARADAR/DAILY',date=date,paginate=True)
        daily = daily.append(dailytoday)
        daily['date'] = pd.to_datetime(daily['date'])
    
        daily.to_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_DAILY.csv',index=False)
    else:
        print('Data up to date')
        
    print(daily['date'].max())
    
    
@timeit
def nasdaqRTAT(date=lastdate):
    
    rtat = pd.read_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/NDAQ_RTAT.csv')
    
    # CHECK FOR NEW DATA, APPEND IF NEW, AND OVERWRITE CSV IF NEW. (CAN RUN A PREVIOUS DATE IF NEEDED)
    if  (lastdate.date() - pd.to_datetime(rtat['date'].max()).date()).days > 0:
        print('New Data---')
        
        rtat_today = quandl.get_table('NDAQ/RTAT', date=date,paginate=True)
        rtat = rtat.append(rtat_today)
        rtat['date'] = pd.to_datetime(rtat['date'])
    
        rtat.to_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/NDAQ_RTAT.csv',index=False)
    else:
        print('Data up to date:')
        
    print(rtat['date'].max())
    
@timeit    
def finraSHORTS(date=lastdate):
    
    new = pd.read_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/FINRA_SI.csv')
    
    #NEST EVERYTHING IN CHECK FOR NEW DATA
    if  (lastdate.date() - pd.to_datetime(new['date'].max()).date()).days > 0:
        print('New Data---')
        
        url = 'https://api.finra.org/data/group/otcMarket/name/regShoDaily'
        headers = {
            'Content-Type':'application/json',
            'Accept': 'application/json'
        }

        records=5000
        offset=0
        si = []
        while records == 5000: #this actually needs to be the return output

            customFilter = {
                'limit':5000,
                'offset':offset,
                'compareFilters':[
                    {
                        'compareType':'equal',
                        'fieldName': 'tradeReportDate',
                        'fieldValue': str(date)
                    }
                ]
            }
            request = requests.post(url,headers=headers,json=customFilter)
            df = pd.DataFrame.from_dict(request.json())
            si.append(df)

            #update offset by 5000
            offset += 5000

            #update records with rows returned
            records = df.shape[0]

        #rename columns
        si = pd.concat(si)
        si.drop(['reportingFacilityCode','marketCode','shortExemptParQuantity'],axis=1,inplace=True)
        si.rename({
            'totalParQuantity':'TotalVolume',
            'shortParQuantity':'ShortVolume',
            'securitiesInformationProcessorSymbolIdentifier':'ticker',
            'tradeReportDate':'date'
            },axis=1,inplace=True)
        
        #sum all SI from different TRFs
        si = si.groupby(['ticker','date']).sum().reset_index()
        
        #append new data, write full data
        new = new.append(si)
        new.to_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/FINRA_SI.csv',index=False)
        
    else:
        print('Data up to date:')
    
    print(new['date'].max())

    
@timeit    
def sharadarSF2(date=lastdate):
    
    #PULLS INSIDER TRADING INFO
    sf2 = pd.read_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_SF2.csv')
    
    # CHECK FOR NEW DATA, APPEND IF NEW, AND OVERWRITE CSV IF NEW.
    if  (lastdate.date() - pd.to_datetime(sf2['date'].max()).date()).days > 0:
        print('New Data---')
        sf2today = quandl.get_table('SHARADAR/SF2',filingdate=date,paginate=True)
        
        #PROCESSING
        #FILTER TO TRUE SALES/PURCHASES + EXCLUDE NON-TRANSACTIONS
        sf2today = sf2today[(sf2today['transactionvalue']>0) & (sf2today['transactioncode'].isin(['P','S']))].copy()
        sf2today['transactionvalue'] = (sf2today['transactionshares']/abs(sf2today['transactionshares']))*sf2today['transactionvalue']
        
        ownergrouped = sf2today.groupby(['ticker','filingdate','ownername']).sum()
        ownergrouped.replace([np.inf, -np.inf], np.nan, inplace=True)

        #get mean stake sold by date and ticker -- down the line features can aggregate in diff ways.
        ownergrouped['pctstakechange'] = (ownergrouped['sharesownedfollowingtransaction']-ownergrouped['sharesownedbeforetransaction'])/ownergrouped['sharesownedbeforetransaction']

        #get count of buys, count of sells
        ownergrouped['sellcount'] = np.where(ownergrouped['transactionvalue']<0,1,0)
        ownergrouped['buycount'] = np.where(ownergrouped['transactionvalue']>0,1,0)

        #agg by day
        ownermoves = ownergrouped.groupby(['ticker','filingdate']).sum()[['sellcount','buycount','transactionvalue']]
        ownermoves = pd.concat([ownermoves,ownergrouped.groupby(['ticker','filingdate']).mean()['pctstakechange']],axis=1)
        ownermoves = ownermoves.reset_index()
        ownermoves.rename({'filingdate':'date'},axis=1,inplace=True)

        sf2 = sf2.append(ownermoves)
        sf2['date'] = pd.to_datetime(sf2['date'])
    
        sf2.to_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/SHARADAR_SF2.csv',index=False)
    else:
        print('Data up to date')
        
    print(sf2['date'].max())

    
# ##### IF I EVER NEED TO RUN HISTORIC SI AGAIN -- THIS WILL GET IT FROM NASDAQ DATA LINK
# si_historic = {}
# for count,ticker in enumerate(tickers['ticker'].unique()):
#     print(count,ticker)
#     try:
#         si_historic[ticker] = quandl.get(f'FINRA/FNYX_{ticker}') + quandl.get(f'FINRA/FNSQ_{ticker}')
#     except:
#         si_historic[ticker] = np.nan
# filtered_si = {k:v for (k,v) in si_historic.items() if type(v) is pd.core.frame.DataFrame}
# final_si  = pd.concat(filtered_si)

# final_si = final_si.reset_index()
# final_si.rename({
#     'Date':'date',
#     'level_0':'ticker'
# },axis=1,inplace=True)
# final_si.drop('ShortExemptVolume',axis=1,inplace=True)

# final_si.to_csv('C:/Users/meich/CareerDocs/projects/stock_prediction/Data/FINRA_SI_historic.csv',index=False)
    
  
    
########################################################################################################
# PROCESSING / FEATURE CREATION FUNCTIONS #
########################################################################################################

@timeit
def short_features(df):
    
    #change this to sum of short / total
    df['ShortRatio'] = df['ShortVolume']/df['TotalVolume']
    df['ShortRatio_5'] = df.groupby(['ticker']).apply(
            lambda x: x['ShortVolume'].rolling(5).sum()/x['TotalVolume'].rolling(5).sum()).reset_index(level=0,drop=True)
    df['ShortRatio_15'] = df.groupby(['ticker']).apply(
            lambda x: x['ShortVolume'].rolling(15).sum()/x['TotalVolume'].rolling(15).sum()).reset_index(level=0,drop=True)
    df['ShortRatio_30'] = df.groupby(['ticker']).apply(
            lambda x: x['ShortVolume'].rolling(30).sum()/x['TotalVolume'].rolling(30).sum()).reset_index(level=0,drop=True)
    
    #volume drops/spikes
    df['TotalVolume_5'] = df.groupby(['ticker']).apply(lambda x: x['TotalVolume'].rolling(5).mean()).reset_index(level=0,drop=True)
    df['TotalVolume_15'] = df.groupby(['ticker']).apply(lambda x: x['TotalVolume'].rolling(15).mean()).reset_index(level=0,drop=True)
    df['TotalVolume_30'] = df.groupby(['ticker']).apply(lambda x: x['TotalVolume'].rolling(30).mean()).reset_index(level=0,drop=True)
    
    return df

@timeit
def fundamentals_features(df):
    
    #some feature change measures -- need to be careful about including straight up daily (will just end up predicting ticker)
    
    # mktcap - ev = cash - debt (i.e. negative is more debt than cash, positive is more cash than debt)
    # this wouldn't change until quarterly earnings -- since the price will move mktcap & ev in same way.
    df['cashdebt'] = (df['marketcap']-df['ev'])/df['marketcap']
    
    df['pe_sector'] = df.groupby(['sector','date'])['pe'].transform(np.mean)
    df['cashdebt_sector'] = df.groupby(['sector','date'])['cashdebt'].transform(np.mean)
    df['ps_sector'] = df.groupby(['sector','date'])['ps'].transform(np.mean)
    df['pb_sector'] = df.groupby(['sector','date'])['pb'].transform(np.mean)
    df['evebitda_sector'] = df.groupby(['sector','date'])['evebitda'].transform(np.mean)
    
    df['pe_industry'] = df.groupby(['industry','date'])['pe'].transform(np.mean)
    df['cashdebt_industry'] = df.groupby(['industry','date'])['cashdebt'].transform(np.mean)
    df['ps_industry'] = df.groupby(['industry','date'])['ps'].transform(np.mean)
    df['pb_industry'] = df.groupby(['industry','date'])['pb'].transform(np.mean)
    df['evebitda_industry'] = df.groupby(['industry','date'])['evebitda'].transform(np.mean)    

    return df


@timeit    
def lagged_features(df,ft='closeadj'):
    
    #takes a single feature and produces lagged targets -- default is to calc for price
    df[f'{ft}_lag1'] = df.groupby(['ticker']).apply(lambda x: x[ft].shift(-1)).reset_index(level=0,drop=True)
    df[f'{ft}_lag5'] = df.groupby(['ticker']).apply(lambda x: x[ft].shift(-5)).reset_index(level=0,drop=True)
    df[f'{ft}_lag30'] = df.groupby(['ticker']).apply(lambda x: x[ft].shift(-30)).reset_index(level=0,drop=True)
    df[f'{ft}_lag90'] = df.groupby(['ticker']).apply(lambda x: x[ft].shift(-90)).reset_index(level=0,drop=True)
    df[f'{ft}_lag180'] = df.groupby(['ticker']).apply(lambda x: x[ft].shift(-180)).reset_index(level=0,drop=True)
    df[f'{ft}_lag360'] = df.groupby(['ticker']).apply(lambda x: x[ft].shift(-360)).reset_index(level=0,drop=True)
    
    
    df[f'{ft}_lagpct1'] = (df[f'{ft}_lag1'] - df[ft]) / df[ft]*100
    df[f'{ft}_lagpct5'] = (df[f'{ft}_lag5'] - df[ft]) / df[ft]*100
    df[f'{ft}_lagpct30'] = (df[f'{ft}_lag30'] - df[ft]) / df[ft]*100
    df[f'{ft}_lagpct90'] = (df[f'{ft}_lag90'] - df[ft]) / df[ft]*100
    df[f'{ft}_lagpct180'] = (df[f'{ft}_lag180'] - df[ft]) / df[ft]*100
    df[f'{ft}_lagpct360'] = (df[f'{ft}_lag360'] - df[ft]) / df[ft]*100
    
    return df

@timeit
def pctchange_features(df,fts):
    
    #takes df and list(features) and returns the df with pct chng versions added at 1,5,15,30
    for col in fts:    
        df[f'{col}_pct5'] = df.groupby(['ticker']).apply(
            lambda x: x[col].pct_change(5)).reset_index(level=0,drop=True)
        df[f'{col}_pct15'] = df.groupby(['ticker']).apply(
            lambda x: x[col].pct_change(15)).reset_index(level=0,drop=True)
        df[f'{col}_pct30'] = df.groupby(['ticker']).apply(
            lambda x: x[col].pct_change(30)).reset_index(level=0,drop=True)
        df[f'{col}_pct60'] = df.groupby(['ticker']).apply(
            lambda x: x[col].pct_change(60)).reset_index(level=0,drop=True)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
@timeit    
def rtat_features(df):
    # want to get single dataset features calculated here (i.e. rolling metrics, any within data metrics)
    # USE A STANDARDIZED FEATURE CODING (datasource_OGMetric_generatedmetric_timewindow)
    # sector/industry market wide metrics (after getting equities bundle)

    #TICKER BASED WINDOW METRICS @ 5, 15, 30
    df['activity_5'] = df.groupby(['ticker']).apply(lambda x: x['activity'].rolling(5).mean()).reset_index(level=0,drop=True)
    df['sentiment_5'] = df.groupby(['ticker']).apply(lambda x: x['sentiment'].rolling(5).mean()).reset_index(level=0,drop=True)
    
    df['activity_15'] = df.groupby(['ticker']).apply(lambda x: x['activity'].rolling(15).mean()).reset_index(level=0,drop=True)
    df['sentiment_15'] = df.groupby(['ticker']).apply(lambda x: x['sentiment'].rolling(15).mean()).reset_index(level=0,drop=True)
    
    df['activity_30'] = df.groupby(['ticker']).apply(lambda x: x['activity'].rolling(30).mean()).reset_index(level=0,drop=True)
    df['sentiment_30'] = df.groupby(['ticker']).apply(lambda x: x['sentiment'].rolling(30).mean()).reset_index(level=0,drop=True)
    
    # MARKET ADJUSTED 
    df['sentiment_mkt'] = df['sentiment'] - df.groupby('date')['sentiment'].transform(np.mean)
    df['sentiment_5_mkt'] = df['sentiment_5'] - df.groupby('date')['sentiment_5'].transform(np.mean)
    df['sentiment_15_mkt'] = df['sentiment_15'] - df.groupby('date')['sentiment_15'].transform(np.mean)
    df['sentiment_30_mkt'] = df['sentiment_30'] - df.groupby('date')['sentiment_30'].transform(np.mean)
    
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df


def priceFeatures():
    pass

def InsiderFeatures():
    #rolling long sum of features...
    # buycount, sell count, transaction value, pct stake change
    pass

def InstitutionalFeatures():
    pass


########################################################################################################
# PREDICTION / MODELLING FUNCTIONS #
########################################################################################################

#SETUP DATA FOR ANY MODELLING (CLASSIFICATION OR REGRESSION)
def model_setup(df,features,y,testsize=0.2,gap=False):
    
    #predict on recent dates where no target value... -- need to set to 0 if a stock was delisted?
    
    #DROPS NULLS BASED ON WHERE TARGET IS NULL ONLY
    df.dropna(subset=[y],inplace=True)
    
    # IF GAP VALUE SET, DO A TIME BASED SPLIT (W/ GAP SIZE OF PREDICTION LAG WINDOW)
    if gap:
        tss = TimeSeriesSplit(n_splits = 4,gap=gap) #this is about 20%
        df.sort_values('date',inplace=True)
        X = df[features]
        y = df[y]
        
        for train_index, test_index in tss.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    else:
        #normal random cv splitting
        X_train, X_test, y_train, y_test = train_test_split(
             df[features], df[y], test_size=testsize)
    
    
    
    return X_train, X_test, y_train, y_test




