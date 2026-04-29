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

# --- LSTM Model Definition ---
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

# --- Data Collection ---
apple = yf.download("AAPL", start="2020-01-01", end="2026-4-28")
nvidia = yf.download("NVDA", start="2020-01-01", end="2026-4-28")
sp500 = yf.download("^GSPC", start="2020-01-01", end="2026-4-28")
amd = yf.download("AMD", start="2020-01-01", end="2026-4-28")
slb = yf.download("SLB", start="2020-01-01", end="2026-4-28")
tesla = yf.download("TSLA", start="2020-01-01", end="2026-4-28")
intel = yf.download("INTC", start="2020-01-01", end="2026-4-28")

stocks = {
    # "Apple":   ("AAPL",  apple),
    "Nvidia":  ("NVDA",  nvidia),
    # "Tesla":  ("TSLA",  tesla),
    # "S&P 500": ("^GSPC", sp500),
    # "AMD":     ("AMD",   amd),
    # "SLB":     ("SLB",   slb),
    # "Intel":   ("INTC",  intel)
}

os.makedirs("saved_models", exist_ok=True)

SEQUENCE_LENGTH = 20
EPOCHS = 1200
BATCH_SIZE = 1100
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_50']

for name, (ticker, data) in stocks.items():
    print(f"\n========== Training {name} ==========")

    data.columns = data.columns.get_level_values(0)
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.6f}")


    # --- Evaluate ---
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
    file_name = name.lower().replace(" ", "_")
    plt.savefig(f"saved_models/{file_name}_graph.png")
    plt.show()   

    # --- Save ---
    file_name = name.lower().replace(" ", "_")
    torch.save(model.state_dict(), f"saved_models/{file_name}_lstm.pth")
    joblib.dump(scaler_X, f"saved_models/{file_name}_scaler_X.pkl")
    joblib.dump(scaler_y, f"saved_models/{file_name}_scaler_y.pkl")
   