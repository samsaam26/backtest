import pandas as pd
from datetime import timedelta, datetime, date
import time
import requests
import numpy as np
import requests
import pyarrow.feather as feather
import streamlit as st
from streamlit import components
import plotly.graph_objects as go
import plotly.subplots as ms
import paramiko
import datetime

# df = pd.read_feather("D:/TRADING/Backtesting/master file/data_2019_current.feather")
# # df = pd.read_feather("data_2022_plus.feather")
# # df_main_fundamentals = pd.DataFrame()
# df_main_fundamentals = pd.read_feather("df_main_fundamentals.feather")


def read_file_vps(filename):
    try:
        # filename = 'data_2022_plus.feather'
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('31.220.95.105', username='root', password='Murcia2000!')

        sftp = ssh.open_sftp()
        remote_file_path = '/root/backtest_webpage/'+filename 
        local_file_path = filename  
        sftp.get(remote_file_path, local_file_path)

        sftp.close()
        ssh.close()
        
        df = pd.read_feather(local_file_path)
    except:
        df = pd.DataFrame()#""# pd.read_feather("data_2022_plus.feather")

    return df
def save_file_vps():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('31.220.95.105', username='root', password='Murcia2000!')

        sftp = ssh.open_sftp()

        # Define your local and remote file paths
        local_file_path = 'df_main_fundamentals.feather' 
        remote_file_path = '/root/backtest_webpage/df_main_fundamentals.feather'
        
        # Use the put method to upload the file to the remote system
        sftp.put(local_file_path, remote_file_path)

        sftp.close()
        ssh.close()
        
    except Exception as e:
        print("An error occurred: ", str(e))


st.set_page_config(page_title="Backtest", page_icon=":bar_chart", layout="wide")
st.title(" :bar_chart: Backtest")

API_KEY = '4r6MZNWLy2ucmhVI7fY8MrvXfXTSmxpy'

df = read_file_vps('data_2022_plus.feather')
print(df)
try:
    df_main_fundamentals = read_file_vps('df_main_fundamentals.feather')
    df = df.merge(df_main_fundamentals, on=['date', 'ticker'], how='left')
except:
    df = df



def filter_data(df, start_date, end_date, min_gap, max_gap, min_pm_vol, max_pm_vol, change_day_prior, volume_day_prior):
    # args = df, min_gap, max_gap, min_pm_vol, max_pm_vol
    print(df)
    print(df.columns)
    df['high_pct_open'] = df['high_a'] / df['open_a'] -1
    df['low_pct_open'] = df['low_a'] / df['open_a'] -1
    df['low_pct_high'] = df['low_a'] / df['high_a'] -1
    df['change_day_prior'] = df.groupby('ticker')['prev_close'].pct_change()
    df['volume_day_prior'] = df.groupby('ticker')['volume'].shift()

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date) & 
        (df['gap'] >= min_gap) &
        (df['gap'] < max_gap) &
        (df['pm_vol'] >= min_pm_vol) &
        (df['pm_vol'] < max_pm_vol) &
        (df['change_day_prior'] >= change_day_prior) &
        (df['volume_day_prior'] >= volume_day_prior) 
    ]

    # (df['change_percent_open'] < 0) & 

    df = df.reset_index(drop=True)  

    return df
def rounding(data):
    # Create a DataFrame from the dictionary
    df = pd.DataFrame([data])

    # Format the values to two decimal places
    df = df.round(2)

    # Format the time values to seconds
    time_columns = ['avg_rth_high_time', 'avg_rth_low_time', 'avg_pm_high_time', 'avg_pm_low_time']
    for column in time_columns:
        df[column] = df[column].apply(lambda x: x.strftime('%H:%M:%S'))

    # df = df.to_string(index=False, line_width=1000)
    # df = df.T
    return df
def get_stats(df):
    total_sample = len(df)

    # close red/green
    close_red = df[(df['change_percent_open'] < 0)]
    close_red_pct = len(close_red)/len(df)

    close_green = df[(df['change_percent_open'] > 0)]
    close_green_pct = len(close_green)/len(df)

    # hod/lod before time
    df['rth_high_time'] = pd.to_datetime(df['rth_high_time'], format='%H:%M:%S')
    hod_before_time = df[df['rth_high_time'].dt.time < pd.to_datetime('10:00').time()]
    hod_before_time_pct = len(hod_before_time)/len(df)

    df['rth_low_time'] = pd.to_datetime(df['rth_low_time'], format='%H:%M:%S')
    lod_before_time = df[df['rth_low_time'].dt.time < pd.to_datetime('10:00').time()]
    lod_before_time_pct = len(lod_before_time)/len(df)


    # avg hod/lod 
    df['rth_high_time_minutes'] = df['rth_high_time'].dt.hour * 60 + df['rth_high_time'].dt.minute
    avg_rth_high_time = df['rth_high_time_minutes'].mean()

    df['rth_low_time_minutes'] = df['rth_low_time'].dt.hour * 60 + df['rth_low_time'].dt.minute
    avg_rth_low_time = df['rth_low_time_minutes'].mean()

    base_date = datetime.datetime(1900, 1, 1)
    avg_rth_high_time_timedelta = datetime.timedelta(minutes=avg_rth_high_time)
    avg_rth_high_time = (base_date + avg_rth_high_time_timedelta).time()

    base_date = datetime.datetime(1900, 1, 1)
    avg_rth_low_time_timedelta = datetime.timedelta(minutes=avg_rth_low_time)
    avg_rth_low_time = (base_date + avg_rth_low_time_timedelta).time()

    # avg PM hod/lod 
    df['pm_high_time'] = pd.to_datetime(df['pm_high_time'], format='%H:%M:%S')
    df['pm_high_time_minutes'] = df['pm_high_time'].dt.hour * 60 + df['pm_high_time'].dt.minute
    avg_pm_high_time = df['pm_high_time_minutes'].mean()

    df['pm_low_time'] = pd.to_datetime(df['pm_low_time'], format='%H:%M:%S')
    df['pm_low_time_minutes'] = df['pm_low_time'].dt.hour * 60 + df['pm_low_time'].dt.minute
    avg_pm_low_time = df['pm_low_time_minutes'].mean()

    base_date = datetime.datetime(1900, 1, 1)
    avg_pm_high_time_timedelta = datetime.timedelta(minutes=avg_pm_high_time)
    avg_pm_high_time = (base_date + avg_pm_high_time_timedelta).time()

    base_date = datetime.datetime(1900, 1, 1)
    avg_pm_low_time_timedelta = datetime.timedelta(minutes=avg_pm_low_time)
    avg_pm_low_time = (base_date + avg_pm_low_time_timedelta).time()

    # median high/low/cahnge pct
    median_high_pct_open = df['high_pct_open'].median()
    median_low_pct_open = df['low_pct_open'].median()
    median_low_pct_high = df['low_pct_high'].median()
    median_change_percent_open = df['change_percent_open'].median()

    # chance pmh break
    pmh_break = df[df['high'] >= df['pm_high']]
    pmh_break_pct = len(pmh_break)/len(df)

    #  touch pdc / ssr
    touch_pdc = df[df['low'] <= df['prev_close']]
    touch_pdc_pct = len(touch_pdc)/len(df)

    touch_ssr = df[df['low'] <= df['prev_close']*0.9]
    touch_ssr_pct = len(touch_ssr)/len(df)


    print(df)
    # print("total_sample: ", len(df))
    # print("close_red_pct: ", close_red_pct)
    # print("close_green_pct: ", close_green_pct)
    # print("hod_before_time_pct: ", hod_before_time_pct)
    # print("lod_before_time_pct: ", lod_before_time_pct)
    # print("avg_rth_high_time: ", avg_rth_high_time)
    # print("avg_rth_low_time: ", avg_rth_low_time)
    # print("avg_pm_high_time: ", avg_pm_high_time)
    # print("avg_pm_low_time: ", avg_pm_low_time)
    # print("median_high_pct_open: ", median_high_pct_open)
    # print("median_low_pct_open: ", median_low_pct_open)
    # print("median_low_pct_high: ", median_low_pct_high)
    # print("median_change_percent_open: ", median_change_percent_open)
    # print("pmh_break_pct: ", pmh_break_pct)
    # print("touch_pdc_pct: ", touch_pdc_pct)
    # print("touch_ssr_pct: ", touch_ssr_pct)

    return pd.Series([total_sample, close_red_pct, close_green_pct, hod_before_time_pct, lod_before_time_pct, \
                      avg_rth_high_time, avg_rth_low_time, avg_pm_high_time, avg_pm_low_time, \
                        median_high_pct_open, median_low_pct_open, median_low_pct_high,  median_change_percent_open, \
                            pmh_break_pct, touch_pdc_pct, touch_ssr_pct], 
                    index=['total_sample', 'close_red_pct', 'close_green_pct', 'hod_before_1000_pct', 'lod_before_1000_pct', \
                           'avg_rth_high_time', 'avg_rth_low_time', 'avg_pm_high_time', 'avg_pm_low_time', \
                            'median_high_pct_open', 'median_low_pct_open', 'median_low_pct_high', 'median_change_percent_open', \
                                'pmh_break_pct', 'touch_pdc_pct', 'touch_ssr_pct'])
def stats_results(df):
    st.title("Statistics")    
    st.header("Statistics:")
    columns = ['total_sample', 'close_red_pct', 'close_green_pct',
               'hod_before_1000_pct', 'lod_before_1000_pct',
               'avg_rth_high_time', 'avg_rth_low_time',
               'avg_pm_high_time', 'avg_pm_low_time',
               'median_high_pct_open', 'median_low_pct_open',
               'median_low_pct_high', 'median_change_percent_open',
               'pmh_break_pct', 'touch_pdc_pct', 'touch_ssr_pct']

    cols_per_row = 4

    row_str = ''
    for idx, column in enumerate(columns):
        if idx % cols_per_row == 0:
            row_str += "<div style='display: flex;'>"

        row_str += "<div style='border: 1px solid #ccc; padding: 10px; flex: 1;'>"
        row_str += f"<b>{column}</b>: {df[column].iloc[0]}"
        row_str += "</div>"

        if (idx + 1) % cols_per_row == 0 or idx == len(columns) - 1:
            row_str += "</div>"

    st.markdown(row_str, unsafe_allow_html=True)

def vwap_func(df):
    H = df["h"]
    L = df["l"]
    C = df["c"]
    V = df["v"]
    res = (V * (H+L+C) / 3).cumsum() / V.cumsum()
    return res.to_frame()
def adjust_intraday(df):
    #df=pd.DataFrame(results)
    df['date_time']=pd.to_datetime(df['t']*1000000).dt.tz_localize('UTC')
    df['date_time']=df['date_time'].dt.tz_convert('US/Eastern')

    df['date_time'] = df['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # df=df.set_index(['date_time']).asfreq('1min')
    # df.v = df.v.fillna(0)
    # df[['c']] = df[['c']].ffill()
    # df['h'].fillna(df['c'], inplace=True)
    # df['l'].fillna(df['c'], inplace=True)
    # df['o'].fillna(df['c'], inplace=True)
    # df=df.between_time('04:00', '20:00')
    df = df.reset_index(level=0)

    df['time'] = pd.to_datetime(df['date_time']).dt.time
    df['date'] = pd.to_datetime(df['date_time']).dt.date

    daily_v_sum = df.groupby(df['date_time'].dt.date)['v'].sum()
    valid_dates = daily_v_sum[daily_v_sum > 0].index
    df = df[df['date_time'].dt.date.isin(valid_dates)]
    df = df.reset_index(drop=True)


    df['ema9'] = df['c'].ewm(span=9,adjust=False).mean()
    df['ema20'] = df['c'].ewm(span=20,adjust=False).mean()
    df['ema50'] = df['c'].ewm(span=50,adjust=False).mean()
    df['ema200'] = df['c'].ewm(span=200,adjust=False).mean()

    df = df.assign(vwap= df.groupby('date', group_keys=False).apply(vwap_func))
    return df
def adjust_daily(df):
    # df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['t'], unit='ms')#.dt.date
    # df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int) 
    df['date'] = df['date'].dt.date      

    return df

def get_stock_data(ticker, start_date, end_date, candle_type, timeframe, adjusted):
    base_url = 'https://api.polygon.io/v2/aggs/ticker/{}/range/{}/{}/{}/{}?adjusted={}&apiKey={}'

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    ticker = ticker.upper()
    adjusted = str(adjusted)
    adjusted = adjusted.lower()
    data = []

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(ticker, timeframe, candle_type, start_date_str, end_date_str, adjusted)
    url = base_url.format(ticker, timeframe, candle_type, start_date_str, end_date_str, adjusted, API_KEY)

    # Send the API request
    response = requests.get(url)

    # If the response was successful, parse the data
    if response.status_code == 200:
        json_response = response.json()
        df = pd.DataFrame(json_response['results'])
        if candle_type == "day":
            df = adjust_daily(df)
        else:
            df = adjust_intraday(df)


        return df

    else:
        df = ""
        return df

def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['o'],
        high=df['h'],
        low=df['l'],
        close=df['c'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    # fig.show()
    st.pyplot(fig)




def stats_page():
    st.header('Stats')
    st.sidebar.header("Data Filtering")
    current_date = date.today()
    start_date = st.sidebar.date_input("Start Date", current_date - timedelta(days=365))     
    end_date = st.sidebar.date_input("End Date", current_date)

    min_gap = st.sidebar.number_input("Min Gap", value=0.2, step=0.01)
    max_gap = st.sidebar.number_input("Max Gap", value=1.00, min_value=0.1, max_value=10.0, step=0.01)
    min_pm_vol = st.sidebar.number_input("Min PM Volume", value=100000)
    max_pm_vol = st.sidebar.number_input("Max PM Volume", value=100000000)
    change_day_prior = st.sidebar.number_input("Change Day Prior", value=0.2, step=0.01)
    volume_day_prior = st.sidebar.number_input("Volume Day Prior", value=0)

    filtered_df = filter_data(df, start_date, end_date, min_gap, max_gap, min_pm_vol, max_pm_vol, change_day_prior, volume_day_prior)
    stats = get_stats(filtered_df)
    stats = rounding(stats)
    stats_results(stats)
    st.write(filtered_df)
# def input_fundies_page():
    st.header('Input Fundamentals')
    st.title("Fundamental Data Entry")
    date_input = st.date_input("Date", value=None, key=None)
    ticker = st.text_input("Ticker")
    months_cash_remaining = st.number_input("Months of Cash Remaining", value=0.0)
    
    if st.button("Submit"):
        st.title("Results")
        st.write("Date:", date_input)
        st.write("Ticker:", ticker)
        st.write("Months of Cash Remaining:", months_cash_remaining)

def input_fundies_page():
    global df_main_fundamentals
    st.header('Input Fundamentals')
    st.title("Fundamental Data Entry")
    
    # Date input
    date_input  = st.date_input("Date", value=None, key=None)
    date_str = str(date_input)
    date1 = datetime.datetime.strptime(date_str, "%Y-%m-%d")

    # Ticker input
    ticker = st.text_input("Ticker")
    ticker = ticker.upper()
    
    # Float input
    float_value = st.number_input("Float (m)", value=0.0)
    
    # Market Cap input
    market_cap = st.number_input("Market Cap (m)", value=0.0)
    
    months_cash_remaining = st.number_input("Months of Cash Remaining", value=0.0)
    
    # Country input with tag options
    country_options = ["US", "China", "Singapore", "Malaysia", "other"]
    country = st.multiselect("Country", country_options)
    if "other" in country:
        country.append(st.text_input("Enter Country Name"))
    
    # Sector input
    sector_options = ['Information Technology', 'Health Care', 'Financials', 'Consumer Cyclical', 'Communication Services', 'Industrials', 'Energy', 'Real Estate', 'Other']
    sector = st.multiselect("sector", sector_options)
    sector = st.text_input("Sector")
    if "Other" in sector:
        sector.append(st.text_input("Enter Sector Name"))
    
    # ITM dilution input with tag options
    itm_dilution_options = ["warrants", "options", "atm", "equity line", "common stock", "other"]
    itm_dilution = st.multiselect("ITM Dilution", itm_dilution_options)
    itm_dilution_data = {}
    for itm in itm_dilution:
        if itm == "other":
            itm = st.text_input("Enter ITM Dilution")
            itm_dilution_data[itm] = {
                "Amount": st.text_input(f"Enter amount of shares or $ amount (m) for {itm}"),
                "Registered Month": st.number_input(f"Enter registered month for {itm}", min_value=1, max_value=12),
                "Registered Year": st.number_input(f"Enter registered year for {itm}", min_value=1900, max_value=2100)
            }
        else:
            itm_dilution_data[itm] = {
                "Amount": st.text_input(f"Enter amount of shares or $ amount (m) for {itm}"),
                "Registered Month": st.number_input(f"Enter registered month for {itm}", min_value=1, max_value=12),
                "Registered Year": st.number_input(f"Enter registered year for {itm}", min_value=1900, max_value=2100)
            }
    
    ways_to_raise_options = ["s-1 R2G", "s-1 not R2G", "s-3", "other"]
    ways_to_raise = st.multiselect("Ways to Raise", ways_to_raise_options)
    ways_to_raise_data = {}

    for way in ways_to_raise:
        if way == "other":
            way = st.text_input("Enter Ways to Raise")
        underwriter_options = ["HC-Wainright", "maxim", "jefferies", "bousted", "armistice", "think equity", "other"]
        underwriter = st.multiselect(f"Underwriter for {way}", underwriter_options)

        if way == "other":
            ways_to_raise_data[way] = {
                "Amount": st.text_input(f"Enter $ amount (m) for {way}"),
                "Registered Month": st.number_input(f"Enter registered month for {way}", min_value=1, max_value=12),
                "Registered Year": st.number_input(f"Enter registered year for {way}", min_value=1900, max_value=2100),
                "Underwriter": underwriter
            }
        else:
            ways_to_raise_data[way] = {
                "Amount": st.text_input(f"Enter $ amount (m) for {way}"),
                "Registered Month": st.number_input(f"Enter registered month for {way}", min_value=1, max_value=12),
                "Registered Year": st.number_input(f"Enter registered year for {way}", min_value=1900, max_value=2100),
                "Underwriter": underwriter
            }
    
    
    if st.button("Submit"):
        data = {
            "date": [date1],
            "ticker": [ticker],
            "Float": [float_value],
            "Market Cap": [market_cap],
            "Months of Cash Remaining": [months_cash_remaining],
            "Country": [", ".join(country)],
            "Sector": [sector]
        }
        for itm, values in itm_dilution_data.items():
            data[itm] = [values["Amount"]]
            data[f"{itm} Registered Month"] = [values["Registered Month"]]
            data[f"{itm} Registered Year"] = [values["Registered Year"]]
        
        for way, values in ways_to_raise_data.items():
            data[way] = [values["Amount"]]
            data[f"{way} Registered Month"] = [values["Registered Month"]]
            data[f"{way} Registered Year"] = [values["Registered Year"]]
            data[f"{way} Underwriter"] = [values["Underwriter"]]
        
        df = pd.DataFrame(data)
        
        st.title("Results")
        st.dataframe(df)

        df_main_fundamentals = df_main_fundamentals.append(df, ignore_index=True)
        st.dataframe(df_main_fundamentals)
        df_main_fundamentals.to_feather("df_main_fundamentals.feather")
        save_file_vps()



        
def createChart(df):
    fig = ms.make_subplots(rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(x = df['date_time'],
    low = df['l'],
    high = df['h'],
    close = df['c'],
    open = df['o'],
    increasing_line_color = 'green',
    decreasing_line_color = 'red'),
    row=1,
    col=1)

    colors = ['green' if row['o'] - row['c'] <= 0 
                else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['date_time'], 
                            y=df['v'],
                            marker_color=colors
                        ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date_time'],
        y=df['vwap'],
        mode='lines',
        name='vwap', 
        line=dict(color='lime',width=2)
    ))
    
    fig.add_trace(go.Scatter(x=df['date_time'], 
                    y=df['ema9'], 
                    opacity=0.7, 
                    line=dict(color='orange', width=2), 
                    name='ema9'))

    fig.add_trace(go.Scatter(x=df['date_time'], 
                    y=df['ema20'], 
                    opacity=0.7, 
                    line=dict(color='blue', width=2), 
                    name='ema20'))

    fig.add_trace(go.Scatter(x=df['date_time'], 
                    y=df['ema50'], 
                    opacity=0.7, 
                    line=dict(color='aqua', width=2), 
                    name='ema50'))
    
    fig.add_trace(go.Scatter(x=df['date_time'], 
                    y=df['ema200'], 
                    opacity=0.7, 
                    line=dict(color='black', width=2), 
                    name='ema200'))     

        #Update Price Figure layout
    fig.update_layout(title = "Chart", #ticker + '  -  ' + str(date),
    yaxis1_title = 'Stock Price ($)',
    yaxis2_title = 'Volume',
    xaxis2_title = 'Date Time',
    xaxis1_rangeslider_visible = False,
    xaxis2_rangeslider_visible = False)

    extDateList=pd.unique((df['date']))

    for i in range(1,len(extDateList)):
        xx0=str(extDateList[i-1])+" 16:00:00"
        xx1=str(extDateList[i])+" 09:30:00"
        
        fig.add_vrect(x0=xx0,x1=xx1,fillcolor="blue", opacity=0.1)

    xx0=str(df['date_time'][0])
    xx1=str(extDateList[0])+" 09:30:00"
    
    fig.add_vrect(x0=xx0,x1=xx1,fillcolor="blue", opacity=0.1)      

    xx0=str(extDateList[-1])+" 16:00:00"
    xx1=df['date_time'][(len(df)-1)]
    fig.add_vrect(x0=xx0,x1=xx1,fillcolor="blue", opacity=0.1)

    fig.update_xaxes(
    #rangeslider_visible=True,
    rangebreaks=[
        # NOTE: Below values are bound (not single values), ie. hide x to y
        dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
        dict(bounds=[20, 4], pattern="hour"),  # hide hours outside of 9.30am-4pm
        #dict(values=["2022-06-20"])  # hide holidays (Christmas and New Year's, etc)
        ]
    )
    
    
    fig.update_layout(autosize=False, width=1500, height=900)

    # fig.show()
    st.plotly_chart(fig)
def showDailyChart(df):
    fig = ms.make_subplots(rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(x = df['date'],
    low = df['l'],
    high = df['h'],
    close = df['c'],
    open = df['o'],
    increasing_line_color = 'green',
    decreasing_line_color = 'red'),
    row=1,
    col=1)

    colors = ['green' if row['o'] - row['c'] <= 0 
              else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['date'], 
                         y=df['v'],
                         marker_color=colors
                        ), row=2, col=1)
   


    
    #Update Price Figure layout
    fig.update_layout(title = 'Interactive CandleStick & Volume Chart ',
    yaxis1_title = 'Stock Price ($)',
    yaxis2_title = 'Volume',
    xaxis2_title = 'Date',
    xaxis1_rangeslider_visible = False,
    xaxis2_rangeslider_visible = False)

    fig.update_xaxes(
        #rangeslider_visible=True,
        rangebreaks=[
            # NOTE: Below values are bound (not single values), ie. hide x to y
            dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
            #dict(bounds=[20, 4], pattern="hour"),  # hide hours outside of 9.30am-4pm
            #dict(values=["2022-06-20"])  # hide holidays (Christmas and New Year's, etc)
        ]
    )


    # fig.show()
    st.plotly_chart(fig)

def chart_page():
    st.header('Charts')

    with st.sidebar:
        st.header('Chart Inputs')
        ticker = st.text_input('Ticker')
        start_date = st.date_input('Start Date')
        end_date = st.date_input('End Date')
        # candle_type = st.date_input('Candle Type')
        # timeframe = st.date_input('Timeframe')
        # adjusted = st.date_input('Adjusted')

        candle_type = st.selectbox('Candle Type', ['minute', 'hour', 'day'])
        timeframe = st.number_input('Timeframe', min_value=1)  # Set a minimum value if needed
        adjusted = st.selectbox('Adjusted', [True, False])

    df = get_stock_data(ticker, start_date, end_date, candle_type, timeframe, adjusted)
    if len(df) > 0:# "empty":
        # plot_candlestick(df)
        # st.write(df)
        if candle_type == "day":
            showDailyChart(df)
        else:
            createChart(df)


pages = {
    "Stats": stats_page,
    "Input Fundamentals": input_fundies_page,
    "Charts": chart_page
}


def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page = pages[selection]

    page()


if __name__ == "__main__":
    main()
