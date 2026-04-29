import torch
import torch.nn as nn
import joblib
import yfinance as yf
import numpy as np

SEQUENCE_LENGTH = 60
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_50']

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
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
    "Intel":   ("INTC",  "intel")
}

print("========== TOMORROW'S PREDICTIONS (LSTM) ==========")

for display_name, (ticker, file_name) in tickers.items():
    # --- Load model and scalers ---
    model = LSTMModel(input_size=len(features))
    model.load_state_dict(torch.load(f"saved_models/{file_name}_lstm.pth", weights_only=True))
    model.eval()

    scaler_X = joblib.load(f"saved_models/{file_name}_scaler_X.pkl")
    scaler_y = joblib.load(f"saved_models/{file_name}_scaler_y.pkl")


    stock = yf.download(ticker, period="120d", auto_adjust=True)
    stock.columns = stock.columns.get_level_values(0)
    stock['MA_10'] = stock['Close'].rolling(window=10).mean()
    stock['MA_50'] = stock['Close'].rolling(window=50).mean()
    stock = stock.dropna()

    X_scaled = scaler_X.transform(stock[features])
    sequence = torch.tensor(X_scaled[-SEQUENCE_LENGTH:], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_scaled = model(sequence).numpy()

    predicted_tomorrow = float(scaler_y.inverse_transform(predicted_scaled)[0][0])
    actual_today = float(stock['Close'].iloc[-1])

    diff = predicted_tomorrow - actual_today
    direction = "▲" if diff > 0 else "▼"

    print(f"\n--- {display_name} ---")
    print(f"  Today's close:      ${actual_today:.2f}")
    print(f"  Predicted tomorrow: ${predicted_tomorrow:.2f}  {direction} ${abs(diff):.2f}")