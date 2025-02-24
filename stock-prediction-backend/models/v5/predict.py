import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from lstm_model import LSTMModel

MODEL_PATH = "final_lstm_stock_prediction_model.pth"
SEQUENCE_LENGTH = 250
INPUT_SIZE = 19

def load_meta_data():
    meta_file_path = "dataset/symbols_valid_meta.csv"  
    meta_data = pd.read_csv(meta_file_path)
    meta_data = meta_data[['Symbol', 'Market Category', 'Listing Exchange', 'ETF']]
    meta_data.fillna("Unknown", inplace=True)
    return meta_data

def get_meta_features(ticker, ohe, meta_data):
    ticker_meta = meta_data[meta_data['Symbol'] == ticker]
    if ticker_meta.empty:
        raise ValueError(f"Ticker {ticker} not found in metadata.")
    return ohe.transform(ticker_meta[['Market Category', 'Listing Exchange', 'ETF']]).flatten()

def load_model(input_size, hidden_size=200, num_layers=5, dropout=0.2):
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def fetch_data(ticker, sequence_length=SEQUENCE_LENGTH, meta_data=None, ohe=None):
    periods = ["1y", "2y", "3y", "5y"]
    data = None
    for period in periods:
        data = yf.download(ticker, period=period, interval="1d")
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'], data['Signal Line'] = calculate_macd(data['Close'])
        data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data['Close'])
        data.dropna(inplace=True)

        if data.shape[0] >= sequence_length:
            print(f"Data fetched for {ticker} with period: {period}")
            break
        else:
            print(f"Not enough data with period {period}. Trying a longer period...")

    if data is None or data.shape[0] < sequence_length:
        raise ValueError(f"Not enough data available for {ticker}. Try a longer period or a different stock.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'MA100', 'MA200', 'RSI', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band']])
    x_input = scaled_data[-sequence_length:]

    meta_features = get_meta_features(ticker, ohe, meta_data)
    meta_features = np.tile(meta_features, (sequence_length, 1))
    x_input = np.concatenate((x_input, meta_features), axis=1)

    x_input = np.expand_dims(x_input, axis=0)
    x_input = torch.tensor(x_input, dtype=torch.float32)
    last_close_price = float(data['Close'].values[-1])

    return x_input, scaler, last_close_price

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    exp1 = prices.ewm(span=fast_period, adjust=False).mean()
    exp2 = prices.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def predict_trend(model, x_input, scaler, last_close_price):
    with torch.no_grad():
        predicted_price_scaled = model(x_input).item()
    predicted_price = scaler.inverse_transform(np.array([[predicted_price_scaled, 0, 0, 0, 0, 0, 0, 0]]))[0, 0]
    trend = "Bullish" if predicted_price > last_close_price else "Bearish"
    return predicted_price, trend

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <STOCK_TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    try:
        meta_data = load_meta_data()
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(meta_data[['Market Category', 'Listing Exchange', 'ETF']])
        
        model = load_model(input_size=INPUT_SIZE)
        x_input, scaler, last_close_price = fetch_data(ticker, meta_data=meta_data, ohe=ohe)
        predicted_price, trend = predict_trend(model, x_input, scaler, last_close_price)
        
        print(f"Stock: {ticker}")
        print(f"Last Close Price: {last_close_price:.2f}")
        print(f"Predicted Price: {predicted_price:.2f}")
        print(f"Market Trend: {trend}")

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
