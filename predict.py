import joblib
import yfinance as yf
import numpy as np

tickers = {
    "Apple": ("AAPL", "apple"),
    "Nvidia": ("NVDA", "nvidia"),
    "S&P 500": ("^GSPC", "s&p_500"),
    "AMD": ("AMD", "amd"),
    "SLB": ("SLB", "slb")
}

print("Predictions")

for display_name, (ticker, file_name) in tickers.items():
    model = joblib.load(f"saved_models/{file_name}_model.pkl")

    stock = yf.download(ticker, start="2024-01-01", end="2024-12-31")
    stock = stock[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock['MA_10'] = stock['Close'].rolling(window=10).mean()
    stock['MA_50'] = stock['Close'].rolling(window=50).mean()
    stock = stock.dropna()

    X_new = stock[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50']]
    predictions = model.predict(X_new)
    print(f"\n--- {display_name} ---")
    print(predictions)