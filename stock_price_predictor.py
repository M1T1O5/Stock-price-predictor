import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf

ticker = "AAPL" 
data = yf.download(ticker, start="2023-01-01", end="2025-01-01")
print("Downloaded Data:\n", data.head())

data = data[['Close']]

forecast_days = 30
data['Prediction'] = data[['Close']].shift(-forecast_days)

scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(data.drop(['Prediction'], axis=1))[:-forecast_days])
y = np.array(data['Prediction'])[:-forecast_days]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

future_data = np.array(data.drop(['Prediction'], axis=1))[-forecast_days:]
future_data_scaled = scaler.transform(future_data)
future_predictions = model.predict(future_data_scaled)

print(f"\nNext {forecast_days} Days Predicted Prices:")
for i, price in enumerate(future_predictions, 1):
    print(f"Day {i}: ${price:.2f}")