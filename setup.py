import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import matplotlib.pyplot as plt

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


apple = yf.download("AAPL", start="2023-01-01", end="2026-6-16")
nvidia = yf.download("NVDA", start="2023-01-01", end="2026-6-16")
sp500 = yf.download("^GSPC", start="2023-01-01", end="2026-6-16")
amd = yf.download("AMD", start="2023-01-01", end="2026-6-16")
slb = yf.download("SLB", start="2023-01-01", end="2026-6-16")
tesla = yf.download("TSLA", start="2023-01-01", end="2026-6-16")
intel = yf.download("INTC", start="2023-01-01", end="2026-6-16")
google = yf.download("GOOGL", start="2023-01-01", end="2026-6-16")
meta = yf.download("META", start="2023-01-01", end="2026-6-16")
nasdaq = yf.download("^IXIC", start="2023-01-01", end="2026-6-16")
roblox = yf.download("RBLX", start="2023-01-01", end="2026-6-16")
spacex = yf.download("SPCX", start="2026-06-12", end="2026-06-16")

stocks = {
    "Apple":   ("AAPL",  apple),
    "Nvidia":  ("NVDA",  nvidia),
    "Tesla":  ("TSLA",  tesla),
    "S&P 500": ("^GSPC", sp500),
    "AMD":     ("AMD",   amd),
    "SLB":     ("SLB",   slb),
    "Intel":   ("INTC",  intel),
    "Google":   ("GOOGL", google),
    "Meta":   ("META", meta),
    "Nasdaq":   ("^IXIC", nasdaq),
    "Roblox":  ("RBLX",  roblox),
    # "SpaceX": ("SPCX", spacex)
}

os.makedirs("saved_models", exist_ok=True)

SEQUENCE_LENGTH = 2
EPOCHS = 1000
BATCH_SIZE = 32


best_loss = float('inf')
epochs_without_improvement = 0
patience = 50


features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA_10', 'MA_50',
    'RSI',
    'MACD', 'MACD_signal', 'MACD_hist',
    'BB_upper', 'BB_lower', 'BB_mid',
    'Volume_change', 
    'Return_1d', 'Return_5d', 'Return_10d',
    'Lag_1', 'Lag_2', 'Lag_3'
]

for name, (ticker, data) in stocks.items():
    print(f"\n========== Training {name} ==========")

    data.columns = data.columns.get_level_values(0)
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI']   = compute_RSI(df['Close'])
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_MACD(df['Close'])
    df['BB_upper'], df['BB_lower'], df['BB_mid'] = compute_bollinger(df['Close'])
    df['Volume_change'] = df['Volume'].pct_change()
    df['Volume_change'] = df['Volume_change'].replace([np.inf, -np.inf], np.nan)  # ← fixed
    df['Return_1d']  = df['Close'].pct_change(1)
    df['Return_5d']  = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Lag_1'] = df['Close'].shift(1)  
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[['Target']])

    X_seq, y_seq = [], []
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        X_seq.append(X_scaled[i - SEQUENCE_LENGTH:i])
        y_seq.append(y_scaled[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(input_size=len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    file_name = name.lower().replace(" ", "_")

    best_loss = float('inf')
    epochs_without_improvement = 0
    loss_history = []

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"saved_models/{file_name}_lstm.pth")  # save best here
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0:  # print every 10 epochs only
            print(f"  Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.6f} — Best: {best_loss:.6f}")

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

   
    model.load_state_dict(torch.load(f"saved_models/{file_name}_lstm.pth", weights_only=True))

  
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(torch.tensor(X_test)).numpy()

    predictions = scaler_y.inverse_transform(predictions_scaled)
    actual = scaler_y.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)
    print(f"\n  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")

  
    
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual Price", color="blue")
    plt.plot(predictions, label="Predicted Price", color="orange", linestyle="--")
    plt.title(f"{name} — LSTM Predictions vs Actual")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"saved_models/{file_name}_graph.png")
    plt.show()

    
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, color="red")
    plt.title(f"{name} — Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"saved_models/{file_name}_loss.png")
    plt.show()

    
    
    joblib.dump(scaler_X, f"saved_models/{file_name}_scaler_X.pkl")
    joblib.dump(scaler_y, f"saved_models/{file_name}_scaler_y.pkl")