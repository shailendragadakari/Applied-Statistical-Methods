# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Load the historical dataset
from google.colab import drive
drive.mount('/content/drive')
data_path = '/content/drive/My Drive/ASM Assignments/Time Series Analysis/AAPL.csv'
data = pd.read_csv(data_path)

# Exploring the dataset
print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Parse dates and set the index if necessary
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

# Select the 'Close' column for analysis if available
if 'Close' not in data.columns:
    raise ValueError("The dataset does not contain a 'Close' column for stock prices.")

close_prices = data['Close']

# Plot the historical data
plt.figure(figsize=(10, 6))
plt.plot(close_prices, label='Closing Prices')
plt.title('Apple Historical Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Decompose the time series to identify trends and seasonality
decompose_result = seasonal_decompose(close_prices, model='additive', period=30)

# Plot decomposition results
fig = decompose_result.plot()
fig.set_size_inches(12, 8)
plt.show()

# Train-test split
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[:train_size], close_prices[train_size:]

# Function to evaluate model performance
def evaluate_model(test, predictions):
    mae = mean_absolute_error(test, predictions)
    rmse = sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    print(f"Model Performance:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%")

# ARIMA modeling
print("\nFitting ARIMA model...")
arima_model = ARIMA(train, order=(5, 1, 2))  # Default parameters, can be tuned
arima_fit = arima_model.fit()
arima_predictions = arima_fit.forecast(steps=len(test))

# Evaluate ARIMA model
evaluate_model(test, arima_predictions)

# SARIMA modeling (if seasonality is detected)
print("\nFitting SARIMA model...")
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))  # Default parameters, can be tuned
sarima_fit = sarima_model.fit()
sarima_predictions = sarima_fit.forecast(steps=len(test))

# Evaluate SARIMA model
evaluate_model(test, sarima_predictions)

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, arima_predictions, label='ARIMA Predictions', color='orange')
plt.plot(test.index, sarima_predictions, label='SARIMA Predictions', color='green')
plt.title('Actual vs. Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Forecast the next month (30 days) using SARIMA
sarima_forecast = sarima_fit.get_forecast(steps=30)
sarima_forecast_values = sarima_forecast.predicted_mean
sarima_conf_int = sarima_forecast.conf_int()

# Plot the forecast with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(close_prices.index, close_prices, label='Historical Data', color='blue')
plt.plot(pd.date_range(start=close_prices.index[-1], periods=31, freq='D')[1:],
         sarima_forecast_values, label='Forecast', color='orange')
plt.fill_between(pd.date_range(start=close_prices.index[-1], periods=31, freq='D')[1:],
                 sarima_conf_int.iloc[:, 0], sarima_conf_int.iloc[:, 1], color='gray', alpha=0.2,
                 label='Confidence Interval')
plt.title('SARIMA Forecast for Next Month')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

