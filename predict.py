import joblib
import yfinance as yf
import numpy as np

tickers = {
    "Apple":   ("AAPL",  "apple"),
    "Nvidia":  ("NVDA",  "nvidia"),
    "S&P 500": ("^GSPC", "s&p_500"),
    "AMD":     ("AMD",   "amd"),
    "SLB":     ("SLB",   "slb")
}

print("Predictions")

for display_name, (ticker, file_name) in tickers.items():
    model = joblib.load(f"saved_models/{file_name}_model.pkl")

    stock = yf.download(ticker, period="60d", auto_adjust=True)
    stock.columns = stock.columns.get_level_values(0)  # flatten multi-level columns
    stock['MA_10'] = stock['Close'].rolling(window=10).mean()
    stock['MA_50'] = stock['Close'].rolling(window=50).mean()
    stock = stock.dropna()

    latest = stock[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_50']].iloc[-1:]
    predicted_tomorrow = float(model.predict(latest)[0])
    actual_today = float(stock['Close'].iloc[-1])

    diff = predicted_tomorrow - actual_today
    direction = "▲" if diff > 0 else "▼"

    print(f"\n--- {display_name} ---")
    print(f"  Today's close:      ${actual_today:.2f}")
    print(f"  Predicted tomorrow: ${predicted_tomorrow:.2f}  {direction} ${abs(diff):.2f}")