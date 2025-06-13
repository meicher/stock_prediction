import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from refresh_functions import *

### Get Latest VIX Price History File (Refreshed EOD by CBOE) #########################
url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
response = requests.get(url)
csv_text = response.content.decode("utf-8")
vix = pd.read_csv(StringIO(csv_text))
vix = vix[['DATE','CLOSE']]
vix.columns = ['date','close']
vix['date'] = pd.to_datetime(vix['date'])

#### Feature Engineering####################################################
#Alternative to close lag -- better metric for options windows.
vix = future_max_gain_drop(vix,'close',windows=[5,10,15])

lag_predictor(vix,'close',time=[3,5,10,15,20,30,50],date='date')
moving_avgs(vix,['close'],date='date',time=[5,10,15,20,25,50,100])

## ema <-4 (similar to a 5_25 diff sma, doesn't hold on to past values as hard
vix['close_EMA50'] = vix['close'].ewm(span=50).mean()
vix['close_EMA5'] = vix['close'].ewm(span=5).mean()
vix['close_5_50_diff_ema'] = vix['close_EMA5'] - vix['close_EMA50']
vix['close_5_50_diff_ema_norm'] = vix['close_5_50_diff_ema']/vix['close_5_50_diff_ema'].rolling(5).std() #not entirely sure how this works

vix['close_5_20_diff'] = vix['close_MA5'] - vix['close_MA20']
vix['close_1_50_diff'] = vix['close'] - vix['close_MA50']
vix['close_1_5_diff'] = vix['close'] - vix['close_MA5']
vix['close_5_50_diff'] = vix['close_MA5']-vix['close_MA50']
vix['close_100_50_diff'] = vix['close_MA100']-vix['close_MA50']
vix['close_5_25_diff'] = vix['close_MA5']-vix['close_MA25']

#quantiles are inherently forward looking...
quantile_inds(vix,cols=['close_5_20_diff','close_5_50_diff','close_100_50_diff','close_5_50_diff','close_1_5_diff','close_5_50_diff_ema',
                       'close_5_50_diff_ema_norm'],quantiles=[.05,.95])

#buy indicator vars
vix['close_5_50_diff_neg6'] = np.where(vix['close_5_50_diff']<=-6,1,0)
vix['close_5_50_diff_neg4'] = np.where(vix['close_5_50_diff']<=-4,1,0)
vix['close_5_50_diff_neg8'] = np.where(vix['close_5_50_diff']<=-8,1,0)
vix['close_5_50_diff_ema_neg4'] = np.where(vix['close_5_50_diff_ema']<=-4,1,0)
vix['close_5_50_diff_ema_norm_neg30'] = np.where(vix['close_5_50_diff_ema_norm']<=-30,1,0)
vix['close_5_50_diff_ema_norm_neg20'] = np.where(vix['close_5_50_diff_ema_norm']<=-20,1,0)
vix['close_5_50_diff_ema_norm_neg10'] = np.where(vix['close_5_50_diff_ema_norm']<=-10,1,0)

#trigger + under mean vix (could use 75th percentile vix at 23 as well)
vix['buy_ind6'] = np.where((vix['close_5_50_diff_neg6'])  & (vix['close']<20),1,0)
vix['buy_ind4'] = np.where((vix['close_5_50_diff_neg4'])  & (vix['close']<20),1,0)
vix['buy_ind_ema10'] = np.where((vix['close_5_50_diff_ema_norm_neg10'])  & (vix['close']<20),1,0)
vix['buy_ind_ema20'] = np.where((vix['close_5_50_diff_ema_norm_neg20'])  & (vix['close']<20),1,0)

# ##NULL THE PANDEMIC ROWS
# mask = (vix['date'] >= '2020-02-01') & (vix['date'] <= '2020-12-31')
# vix.loc[mask, :] = np.nan



#### Streamlit ###################################################################################
st.title("Vix Opportunity Gauge")

st.subheader(":rainbow[Is it VIX calls time?]")

#add guage of buying opportunity as 0-sum(buying indicators) for most recent day
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = vix[['close_5_50_diff_neg6','close_5_50_diff_ema_norm_neg10',
                         'close_5_50_diff_ema_norm_neg20','close_5_50_diff_ema_norm_neg30']][vix['date']==yesterday_market_date].sum().sum(),
    title = {'text': "Daily VIX Opportunity"},
    gauge = {
        'axis': {'range': [0, 4]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 1], 'color': "lightpink"},
            {'range': [1, 2], 'color': "lightyellow"},
            {'range': [2, 4], 'color': "lightgreen"}
        ]
    }
))
st.plotly_chart(fig)

# Show recent values
tail = vix[['date','close','close_5_50_diff','close_5_50_diff_neg6','close_5_50_diff_ema_norm',
     'close_5_50_diff_ema_norm_neg10','close_5_50_diff_ema_norm_neg20','close_5_50_diff_ema_norm_neg30']].tail(5)
tail['date'] = pd.to_datetime(tail['date']).dt.strftime('%Y-%m-%d')
st.subheader("Recent Values")
st.dataframe(tail)

#Show indicator lines
data = vix[['date', "close", "close_5_50_diff", "close_5_50_diff_ema_norm"]][vix['date']>'2023']
tab1, tab2 = st.tabs(["Chart", "Dataframe"])
tab1.line_chart(data,x='date', height=250)
tab2.dataframe(data, height=250, use_container_width=True)


#appendix: predictive power of each indicator / coverage
#plotly box plots here
st.divider()
st.subheader('Appendix')

st.write('close_5_50_diff_neg6 vs max 15d gain')
st.plotly_chart(px.box(vix,x='max_15_gain',color='close_5_50_diff_neg6'))

st.write('close_5_50_diff_ema_norm_neg10 vs max 15d gain')
st.plotly_chart(px.box(vix,x='max_15_gain',color='close_5_50_diff_ema_norm_neg10'))

st.write('close_5_50_diff_ema_norm_neg30 vs max 15d gain')
st.plotly_chart(px.box(vix,x='max_15_gain',color='close_5_50_diff_ema_norm_neg30'))