# Stock Price Predictor

An LSTM-based neural network that predicts future stock prices using historical 
data pulled from Yahoo Finance.

## How it Works
- Fetches historical OHLCV data via yfinance
- Trains a PyTorch LSTM on configurable sequence windows
- Evaluates predictions using MAE

## How to Run
1. `pip install -r Requirements.txt`
2. Train: `python setup.py`
3. Predict: `python predict.py`

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| epochs | 1000 | Training iterations |
| batch_size | 32 | Samples per batch |
| sequence_length | 30 | Lookback window |
| patience | 50 | Tolerance of loss |
