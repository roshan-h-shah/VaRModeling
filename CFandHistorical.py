import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as stats
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
    hh = historical_returns.to_frame("Log_Return")
    return hh

# Fetch data once to avoid redundant downloads
spy_data = get_data(['AAPL', 'SPY', 'XOM', 'MSFT'], start="2010-01-01", end="2024-12-01")

def process_data(lambd, window, confidence_level, n_days, spy_data):
    # EWMA Volatility calculation
    log_returns = spy_data['Log_Return'].values
    ewma_volatility = []
    for i in range(len(log_returns)):
        if i < window:
            ewma_volatility.append(np.nan)
        else:
            returns_window = log_returns[i-window:i]
            weights = np.array([lambd ** (window - j - 1) for j in range(window)])
            weights /= weights.sum()
            volatility = np.sqrt(np.sum(weights * returns_window ** 2))
            ewma_volatility.append(volatility)
    spy_data[f'EWMA_Volatility_{n_days}_{confidence_level}'] = ewma_volatility

    # Calculate n-day VaR
    critical_value = stats.norm.ppf(1 - confidence_level)
    spy_data[f'{n_days}d_VaR_{confidence_level}'] = critical_value * spy_data[f'EWMA_Volatility_{n_days}_{confidence_level}'] * np.sqrt(n_days)

    # Calculate summed returns for backtesting
    spy_data[f'{n_days}d_Summed_Return'] = spy_data['Log_Return'].rolling(window=n_days).sum().shift(-n_days + 1)

    # Calculate diff column and exceptions
    spy_data[f'diff_{n_days}_{confidence_level}'] = (spy_data[f'{n_days}d_Summed_Return'] <= spy_data[f'{n_days}d_VaR_{confidence_level}']).astype(int)

    # Calculate accuracy metrics
    valid_data = spy_data.dropna(subset=[f'{n_days}d_VaR_{confidence_level}', f'{n_days}d_Summed_Return'])
    total_observations = len(valid_data)
    expected_exceptions = (1 - confidence_level) * total_observations
    actual_exceptions = valid_data[f'diff_{n_days}_{confidence_level}'].sum()
    if total_observations > 0:
        error = abs((1 - confidence_level) - (actual_exceptions / total_observations))
    else:
        error = np.nan  # Avoid division by zero
    return {
        'error': error,
        'actual_exceptions': actual_exceptions,
        'expected_exceptions': expected_exceptions,
        'total_observations': total_observations
    }

if __name__ == "__main__":
    n_days_list = [1, 5, 10, 30, 50]
    confidence_levels = [0.90,0.95,0.97, 0.99, 0.995]
    lambd = 0.93
    window = 30

    # Initialize a DataFrame to store errors
    errors_table = pd.DataFrame(index=n_days_list, columns=confidence_levels)

    for n_days in n_days_list:
        for confidence_level in confidence_levels:
            result = process_data(lambd, window, confidence_level, n_days, spy_data.copy())
            error = result['error']
            errors_table.loc[n_days, confidence_level] = error

    # Convert errors to percentage format
    errors_table = errors_table.applymap(lambda x: f"{x:.2%}" if pd.notnull(x) else x)

    print("Errors of VaR Estimates:")
    print(errors_table)

    # Export the table as an image
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=errors_table.values,
                     rowLabels=errors_table.index,
                     colLabels=errors_table.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)  # Adjust as needed
    plt.title('EWMA Errors (5x5 Table)', fontsize=16)
    plt.savefig('errors_table.png', bbox_inches='tight')
    plt.show()
