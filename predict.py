import torch
import torch.nn as nn
import joblib
import yfinance as yf
import numpy as np

SEQUENCE_LENGTH = 30
features = [ 'Open', 'High', 'Low', 'Close', 'Volume',
    'MA_10', 'MA_50',
    'RSI',
    'MACD', 'MACD_signal', 'MACD_hist',
    'BB_upper', 'BB_lower', 'BB_mid',
    'Volume_change', 
    'Return_1d', 'Return_5d', 'Return_10d',
    'Lag_1', 'Lag_2', 'Lag_3'
    ]

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_bollinger(series, period=20):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, lower, mid


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc2(self.relu(self.fc1(out[:, -1, :])))
        return out

tickers = {
    "Apple":   ("AAPL",  "apple"),
    "Nvidia":  ("NVDA",  "nvidia"),
    "S&P 500": ("^GSPC", "s&p_500"),
    "AMD":     ("AMD",   "amd"),
    "SLB":     ("SLB",   "slb"),
    "Tesla":   ("TSLA",  "tesla"),
    "Intel":   ("INTC",  "intel"),
    "Roblox":  ("RBLX",  "roblox")
}

print("========== TOMORROW'S PREDICTIONS (LSTM) ==========")

for display_name, (ticker, file_name) in tickers.items():
    model = LSTMModel(input_size=len(features))
    model.load_state_dict(torch.load(f"saved_models/{file_name}_lstm.pth", weights_only=True))
    model.eval()

    scaler_X = joblib.load(f"saved_models/{file_name}_scaler_X.pkl")
    scaler_y = joblib.load(f"saved_models/{file_name}_scaler_y.pkl")


    stock = yf.download(ticker, period="100d", auto_adjust=True)
    stock.columns = stock.columns.get_level_values(0)

    # Compute all features — must match setup.py exactly
    stock['MA_10'] = stock['Close'].rolling(window=10).mean()
    stock['MA_50'] = stock['Close'].rolling(window=50).mean()
    stock['RSI']   = compute_RSI(stock['Close'])
    stock['MACD'], stock['MACD_signal'], stock['MACD_hist'] = compute_MACD(stock['Close'])
    stock['BB_upper'], stock['BB_lower'], stock['BB_mid'] = compute_bollinger(stock['Close'])
    stock['Volume_change'] = stock['Volume'].pct_change()
    stock['Volume_change'] = stock['Volume_change'].replace([np.inf, -np.inf], np.nan)
    stock['Return_1d']  = stock['Close'].pct_change(1)
    stock['Return_5d']  = stock['Close'].pct_change(5)
    stock['Return_10d'] = stock['Close'].pct_change(10)
    stock['Lag_1'] = stock['Close'].shift(1)
    stock['Lag_2'] = stock['Close'].shift(2)
    stock['Lag_3'] = stock['Close'].shift(3)
    stock = stock.dropna()

    X_scaled = scaler_X.transform(stock[features])
    sequence_yesterday = torch.tensor(X_scaled[-SEQUENCE_LENGTH-1:-1], dtype=torch.float32).unsqueeze(0)
    sequence_today = torch.tensor(X_scaled[-SEQUENCE_LENGTH: ], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_today    = float(scaler_y.inverse_transform(model(sequence_yesterday).numpy())[0][0])
        predicted_tomorrow = float(scaler_y.inverse_transform(model(sequence_today).numpy())[0][0])

    
    actual_today = float(stock['Close'].iloc[-1])
    diff = predicted_tomorrow - actual_today
    direction = "▲" if diff > 0 else "▼"
    print(f"\n--- {display_name} ---")
    print(f"  Predicted today:    ${predicted_today:.2f}  (actual: ${actual_today:.2f}  off by: ${abs(predicted_today - actual_today):.2f})")
    print(f"  Predicted tomorrow: ${predicted_tomorrow:.2f}  {direction} ${abs(diff):.2f}")
