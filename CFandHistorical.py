import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt


def get_data(tickers, start, end):
    full = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False)[['Adj Close']]
        data.rename(columns={"Adj Close": "Adjusted_Close"}, inplace=True)
        full[ticker] = data['Adjusted_Close']
    log_returns = np.log(full / full.shift(1))
    log_returns = log_returns.dropna()
    weights = [0.4, 0.3, 0.2, 0.1]
    historical_returns = (log_returns * weights).sum(axis=1)
    return historical_returns.to_frame("Log_Return")

def calculate_rolling_returns(data, window):
    range_returns = data.rolling(window=window).sum()
    range_returns = range_returns.dropna()
    range_returns.rename(columns={"Log_Return": "RollingReturn"}, inplace=True)
    return range_returns

def backtest_cornish_fischer(range_df, training_window, confidence):
    z = norm.ppf(1 - confidence)
    def cornish_fischer_adjustment(series):
        mean = series.mean()
        std = series.std()
        skewness = series.skew()
        kurtosis = series.kurt() - 3  # Excess kurtosis
        cf_var = mean + std * (
            z 
            + ((z ** 2 - 1) * skewness) / 6 
            + ((z ** 3 - 3 * z) * kurtosis) / 24
            + (2 * z ** 3 - 5 * z) * (kurtosis ** 2) / 36
        )
        return cf_var
    
    range_df['HistoricalVaR'] = (
        range_df['RollingReturn']
        .shift(1)
        .rolling(window=training_window)
        .quantile(1 - confidence)
    )
    range_df['CornishFischerVaR'] = (
        range_df['RollingReturn']
        .shift(1)
        .rolling(window=training_window)
        .apply(cornish_fischer_adjustment, raw=False)
    )
    return range_df.dropna()

# Parameters
tickers = ['AAPL', 'SPY', 'XOM', 'MSFT']
start_date = "2010-01-01"
end_date = "2024-12-01"
confidence_levels = [0.90, 0.95, 0.97, 0.99, 0.995]
windows = [1, 5, 10, 30, 50]
training_window = 1000

# Download data
data = get_data(tickers, start=start_date, end=end_date)

# Store results
results = {}
for confidence in confidence_levels:
    for window in windows:
        rolling_data = calculate_rolling_returns(data, window)
        results[(confidence, window)] = backtest_cornish_fischer(rolling_data, training_window, confidence)

# Prepare Cornish-Fischer and Historical Error Tables
cornish_fischer_errors = np.zeros((len(windows), len(confidence_levels)))
historical_errors = np.zeros((len(windows), len(confidence_levels)))

for i, window in enumerate(windows):
    for j, confidence in enumerate(confidence_levels):
        result_df = results[(confidence, window)]
        total_values = len(result_df)
        cf_exceptions = (result_df['RollingReturn'] < result_df['CornishFischerVaR']).sum()
        hist_exceptions = (result_df['RollingReturn'] < result_df['HistoricalVaR']).sum()

        # Calculate accuracy using the given formula
        cf_accuracy = abs((1 - confidence) - (cf_exceptions / total_values)) * 100  # Format as percent
        hist_accuracy = abs((1 - confidence) - (hist_exceptions / total_values)) * 100  # Format as percent

        # Store the accuracy values in respective tables
        cornish_fischer_errors[i, j] = cf_accuracy
        historical_errors[i, j] = hist_accuracy
        

# Round to two decimal places and add a % sign
formatted_cornish_fischer_errors = [[f"{round(value, 2)}%" for value in row] for row in cornish_fischer_errors]
formatted_historical_errors = [[f"{round(value, 2)}%" for value in row] for row in historical_errors]

# Create Cornish-Fischer Error Table
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.axis('tight')
ax1.axis('off')

cf_table = ax1.table(
    cellText=formatted_cornish_fischer_errors,
    rowLabels=windows,
    colLabels=confidence_levels,
    loc='center',
    cellLoc='center'
)
cf_table.auto_set_font_size(False)
cf_table.set_fontsize(10)
cf_table.scale(1.2, 1.2)

plt.title("Cornish-Fischer Errors (5x5 Table)", pad=20)
plt.savefig("cornish_fischer_table_5x5.png", bbox_inches="tight")
plt.close()

# Create Historical Error Table
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.axis('tight')
ax2.axis('off')

hist_table = ax2.table(
    cellText=formatted_historical_errors,
    rowLabels=windows,
    colLabels=confidence_levels,
    loc='center',
    cellLoc='center'
)
hist_table.auto_set_font_size(False)
hist_table.set_fontsize(10)
hist_table.scale(1.2, 1.2)

plt.title("Historical Errors (5x5 Table)", pad=20)
plt.savefig("historical_error_table_5x5.png", bbox_inches="tight")
plt.close()

