import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os
apple = yf.download("AAPL", start="2019-01-01", end="2024-12-31")
nvidia = yf.download("NVDA", start="2019-01-01", end="2024-12-31")
sp500 = yf.download("^GSPC", start="2019-01-01", end="2024-12-31")
amd = yf.download("AMD", start="2019-01-01", end="2024-12-31")
slb = yf.download("SLB", start="2019-01-01", end="2024-12-31")

stocks = {
    "Apple": apple,
    "Nvidia": nvidia,
    "S&P 500": sp500,
    "AMD": amd,
    "SLB": slb
}


for name, stock in stocks.items():
    stock = stock[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock['MA_10'] = stock['Close'].rolling(window=10).mean()
    stock['MA_50'] = stock['Close'].rolling(window=50).mean()
    stock = stock.dropna()
    stocks[name] = stock

for name, stock in stocks.items():
    print(f"\n--- {name} ---")
    print(stock.head())


models = {}

for name, stock in stocks.items():
    X = stock[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50']]
    y = stock['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    models[name] = model


    os.makedirs("saved_models", exist_ok=True)
    file_name = f"{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, f"saved_models/{file_name}")
    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

