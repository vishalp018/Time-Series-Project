import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Load the dataset
df = pd.read_csv('all_currencies.csv', parse_dates=['Date'], index_col='Date')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Home Page
def home_page():
    st.title('Time Series Analysis Project')
    st.write("""
    ## About the Project
    This project analyzes cryptocurrency data using various models to predict future trends.
    
    ## About the Team
    Our team 
    Vishal pal
    Lakshita gulati
    Pulkit singh
    
    ## Models Used
    1. ARIMA
    2. SARIMA
    3. Prophet
    4. LSTM
    """)

# Data Preprocessing
def data_overview():
    st.subheader('Data Overview')
    st.write(df.head())
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

# Bitcoin Data
def bitcoin_data():
    st.subheader('Bitcoin Data')
    btc = df[df['Symbol'] == 'BTC']
    btc.drop(['Volume', 'Market Cap'], axis=1, inplace=True)
    btc = btc.apply(pd.to_numeric, errors='coerce')
    btc_month = btc.resample('M').mean()
    st.write(btc_month.head())
    st.line_chart(btc_month['Close'])
    result = adfuller(btc_month['Close'].dropna())
    st.write(f"Dickey-Fuller Test p-value: {result[1]}")
    seasonal_decompose(btc_month['Close'], model='additive').plot()
    st.pyplot(plt)

# ARIMA Model
def arima_model():
    st.title('ARIMA Model')
    btc_month = df[df['Symbol'] == 'BTC'].resample('M').mean()
    model = ARIMA(btc_month['Close'].dropna(), order=(5, 1, 0))
    results = model.fit()
    st.write(results.summary())
    btc_month['ARIMA_Pred'] = results.predict(start=0, end=len(btc_month)-1)
    st.line_chart(btc_month[['Close', 'ARIMA_Pred']])

# SARIMA Model
def sarima_model():
    st.title('SARIMA Model')
    btc_month = df[df['Symbol'] == 'BTC'].resample('M').mean()
    model = SARIMAX(btc_month['Close'].dropna(), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    st.write(results.summary())
    btc_month['SARIMA_Pred'] = results.predict(start=0, end=len(btc_month)-1)
    st.line_chart(btc_month[['Close', 'SARIMA_Pred']])

# Sidebar Navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Choose a model:', ['Home', 'Data Overview', 'Bitcoin Data', 'ARIMA', 'SARIMA'])

if page == 'Home':
    home_page()
elif page == 'Data Overview':
    data_overview()
elif page == 'Bitcoin Data':
    bitcoin_data()
elif page == 'ARIMA':
    arima_model()
elif page == 'SARIMA':
    sarima_model()
