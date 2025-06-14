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
buy_inds = ['close_5_50_diff_neg8','buy_ind_sma4','buy_ind_sma6','buy_ind_ema10','buy_ind_ema20','buy_ind_ema30',
            'buy_ind_ema_50std1.5','buy_ind_ema_50std2','buy_ind_ema_50std2.5']
thresh = 19

def feature_eng(vix):
    
    vix = add_max_gain_and_drawdown(vix,'close',windows=[5,15])
    
    lag_predictor(vix,'close',time=[3,5,10,15,20,30,50],date='date')
    moving_avgs(vix,['close'],date='date',time=[5,10,15,20,25,50,100])
    
    ## ema <-4 (similar to a 5_25 diff sma, doesn't hold on to past values as hard
    vix['close_EMA50'] = vix['close'].ewm(span=50).mean()
    vix['close_EMA5'] = vix['close'].ewm(span=5).mean()
    vix['close_5_50_diff_ema'] = vix['close_EMA5'] - vix['close_EMA50']
    vix['close_5_50_diff_ema_norm'] = vix['close_5_50_diff_ema']/vix['close_5_50_diff_ema'].rolling(5).std()
    vix['close_5_50_diff_ema_norm_50std'] = vix['close_5_50_diff_ema']/vix['close_5_50_diff_ema'].rolling(50).std()
    
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
    vix['close_5_50_diff_ema_norm_neg30'] = np.where(vix['close_5_50_diff_ema_norm']<=-30,1,0)
    vix['close_5_50_diff_ema_norm_neg20'] = np.where(vix['close_5_50_diff_ema_norm']<=-20,1,0)
    vix['close_5_50_diff_ema_norm_neg10'] = np.where(vix['close_5_50_diff_ema_norm']<=-10,1,0)
    vix['close_5_50_diff_ema_norm_50std_neg1.5'] = np.where(vix['close_5_50_diff_ema_norm_50std']<=-1.5,1,0)
    vix['close_5_50_diff_ema_norm_50std_neg2'] = np.where(vix['close_5_50_diff_ema_norm_50std']<=-2,1,0)
    vix['close_5_50_diff_ema_norm_50std_neg2.5'] = np.where(vix['close_5_50_diff_ema_norm_50std']<=-2.5,1,0)
    
    #trigger + under mean vix (could use 75th percentile vix at 23 as well)
    vix['buy_ind_sma6'] = np.where((vix['close_5_50_diff_neg6'])  & (vix['close']<thresh),1,0)
    vix['buy_ind_sma4'] = np.where((vix['close_5_50_diff_neg4'])  & (vix['close']<thresh),1,0)
    vix['buy_ind_ema10'] = np.where((vix['close_5_50_diff_ema_norm_neg10'])  & (vix['close']<thresh),1,0)
    vix['buy_ind_ema20'] = np.where((vix['close_5_50_diff_ema_norm_neg20'])  & (vix['close']<thresh),1,0)
    vix['buy_ind_ema30'] = np.where((vix['close_5_50_diff_ema_norm_neg30'])  & (vix['close']<thresh),1,0)
    vix['buy_ind_ema_50std1.5'] = np.where((vix['close_5_50_diff_ema_norm_50std_neg1.5'])  & (vix['close']<thresh),1,0)
    vix['buy_ind_ema_50std2'] = np.where((vix['close_5_50_diff_ema_norm_50std_neg2'])  & (vix['close']<thresh),1,0)
    vix['buy_ind_ema_50std2.5'] = np.where((vix['close_5_50_diff_ema_norm_50std_neg2.5'])  & (vix['close']<thresh),1,0)
    
    vix['buy_ind_count'] = vix[buy_inds].sum(axis=1)

    return vix

vix = feature_eng(vix)

#### Streamlit ###################################################################################
st.set_page_config(
    layout="wide"  # <-- This sets the app to wide mode by default
)

st.title("Vix Opportunity Gauge")

st.subheader(":rainbow[Is it VIX calls time?]")

# --- USER INPUT TRIGGER ---
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = False

# Trigger input mode
if st.button("Add today's intraday VIX price"):
    st.session_state.input_mode = True

# Show input field if in input mode
if st.session_state.input_mode:
    new_value = st.number_input("Enter current VIX value:", min_value=0.0)
    if st.button("Submit New Value"):
        # Get next date
        new_date = vix['date'].max() + pd.Timedelta(days=1)

        # Create new row and recalculate features (you can adapt these)
        new_row = pd.DataFrame({
            'date': [new_date],
            'close': [new_value]
        })

        # Append and reprocess indicators
        vix = pd.concat([vix, new_row], ignore_index=True)

        #re-process features
        vix = feature_eng(vix)
        
        st.session_state.input_mode = False  # Reset


col1, col2 = st.columns(2)

with col1:
    #add guage of buying opportunity as 0-sum(buying indicators) for most recent day
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = vix[vix['date'] == vix['date'].max()][buy_inds].sum().sum(),
        title = {'text': "Daily VIX Opportunity"},
        gauge = {
            'axis': {'range': [0, max(vix['buy_ind_count'])]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1], 'color': "lightpink"},
                {'range': [1, 3], 'color': "lightyellow"},
                {'range': [3, 9], 'color': "lightgreen"}
            ]
        }
    ))
    st.plotly_chart(fig)

with col2:
    def highlight_buy_count(val):
        if val >= 4:
            return 'background-color: green'
        elif val in (2,3):
            return 'background-color: yellow'
        elif val == 1:
            return 'background-color: lightpink'
        else:
            return ''
    tail = vix[['date','close','buy_ind_count','close_5_50_diff','close_5_50_diff_ema_norm','close_5_50_diff_ema_norm_50std']].tail(10)
    tail['date'] = pd.to_datetime(tail['date']).dt.strftime('%Y-%m-%d')
    styled_df = tail.style.applymap(highlight_buy_count, subset=['buy_ind_count'])
    
    st.subheader("Recent Values")
    st.write(styled_df)


#Show indicator lines
data = vix[['date', "close",'buy_ind_count',"close_5_50_diff", "close_5_50_diff_ema_norm",'close_5_50_diff_ema_norm_50std']][vix['date']>'2023']
tab1, tab2 = st.tabs(["Chart", "Dataframe"])
tab1.line_chart(data,x='date', height=500)
tab2.dataframe(data, height=500, use_container_width=True)


#appendix: predictive power of each indicator / coverage
#plotly box plots here
st.divider()
st.subheader('Appendix')

for i in buy_inds:
    st.write(f'{i} vs max_gain_15' )
    st.dataframe(summarize_signal_performance(vix, i, 'max_15_gain', threshold=.1))


# st.write('close_5_50_diff_neg8 vs max 15d gain')
# st.plotly_chart(px.box(vix,x='max_15_gain',color='close_5_50_diff_neg8'))
