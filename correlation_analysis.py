import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# File paths
bitcoin_data_path = r"C:/Users/zhuqi/Documents/dataset/bitcoin_2017_to_2023.csv"
gold_data_path = r"C:/Users/zhuqi/Documents/data/goldstock.csv"

# Load the data
bitcoin_df = pd.read_csv(bitcoin_data_path)
gold_df = pd.read_csv(gold_data_path)

# Ensure the date columns are in datetime format
bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['timestamp'])
gold_df['Date'] = pd.to_datetime(gold_df['Date'])

# Set the date column as the index
bitcoin_df.set_index('Date', inplace=True)
gold_df.set_index('Date', inplace=True)

# Sort the dataframes by date
bitcoin_df = bitcoin_df.sort_index()
gold_df = gold_df.sort_index()

# Filter data to the range from January 2018 to June 2023
start_date = '2018-01-01'
end_date = '2023-06-30'
bitcoin_df = bitcoin_df.loc[start_date:end_date]
gold_df = gold_df.loc[start_date:end_date]

# Standardize the 'Close' columns (prices)
bitcoin_df['close'] = (bitcoin_df['close'] - bitcoin_df['close'].mean()) / bitcoin_df['close'].std()
gold_df['Close'] = (gold_df['Close'] - gold_df['Close'].mean()) / gold_df['Close'].std()

# Resample Bitcoin data to daily frequency
bitcoin_daily = bitcoin_df['close'].resample('D').mean().fillna(method='ffill')
gold_daily = gold_df['Close'].resample('D').mean().fillna(method='ffill')

# Align the time frames and sort by time
common_dates = bitcoin_daily.index.intersection(gold_daily.index)
bitcoin_daily = bitcoin_daily.loc[common_dates].sort_index()
gold_daily = gold_daily.loc[common_dates].sort_index()

# Initialize an empty list to store correlation strengths
correlation_strengths = []

# Calculate the correlation strength for each day using a 30-day window
window_size = 360
for i in range(len(common_dates) - window_size + 1):
    window_dates = common_dates[i:i + window_size]
    x = bitcoin_daily.loc[window_dates].values.reshape(-1, 1)
    y = gold_daily.loc[window_dates].values.reshape(-1, 1)
    if len(x) > 1 and len(y) > 1:
        reg = LinearRegression().fit(x, y)
        correlation_strengths.append((window_dates[-1], reg.coef_[0][0]))

# Create a DataFrame to store the correlation strengths
correlation_df = pd.DataFrame(correlation_strengths, columns=['Date', 'Correlation'])

# Plot the correlation strengths using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=correlation_df['Date'], y=correlation_df['Correlation'], mode='lines', name='Correlation Strength'))
fig.update_layout(
    title='Daily Correlation Strength between Bitcoin and Gold Prices',
    xaxis_title='Date',
    yaxis_title='Correlation Strength',
    template='plotly_dark'
)
fig.show()





