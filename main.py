import os
from dotenv import load_dotenv
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from web_scraper.fetch_stock_data import fetch_stock_data
from web_scraper.collect_live_data import collect_live_data


# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), 'env/.env')
load_dotenv(env_path)

# Paths to models and metadata
LSTM_MODEL_PATH = os.path.join('models', 'v5', 'final_lstm_stock_prediction_model.pth')
BERT_MODEL_PATH = os.path.join('sentiment_analysis', 'sentiment_model', 'fine_tuned_financial_bert_combined')
META_DATA_PATH = os.path.join('dataset', 'symbols_valid_meta.csv')

# Constants
SEQUENCE_LENGTH = 250
INPUT_SIZE = 19

# Load Metadata and OneHotEncoder
meta_data = pd.read_csv(META_DATA_PATH)[['Symbol', 'Market Category', 'Listing Exchange', 'ETF']].fillna("Unknown")
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(meta_data[['Market Category', 'Listing Exchange', 'ETF']])


# Define LSTM Model
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=200, num_layers=5, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Output of the last time step
        return out


# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMModel(input_size=INPUT_SIZE).to(device)
lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
lstm_model.eval()

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH).to(device)


# Helper Function: Extract Metadata Features
def get_meta_features(ticker, ohe, meta_data):
    ticker_meta = meta_data[meta_data['Symbol'] == ticker]
    if ticker_meta.empty:
        raise ValueError(f"Ticker {ticker} not found in metadata.")
    encoded_meta = ohe.transform(ticker_meta[['Market Category', 'Listing Exchange', 'ETF']])
    return encoded_meta.flatten()


# Sentiment Analysis
def analyze_sentiment(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    sentiment_mapping = {0: 0.0, 1: 1.0, 2: 0.5}
    return [sentiment_mapping[pred.item()] for pred in predictions]


# Preprocessing Functions
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


def preprocess_stock_data(stock_data, meta_data, ohe, ticker):
    stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    stock_data['MA100'] = stock_data['Close'].rolling(window=100).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
    stock_data['MACD'], stock_data['Signal Line'] = calculate_macd(stock_data['Close'])
    stock_data['Upper Band'], stock_data['Lower Band'] = calculate_bollinger_bands(stock_data['Close'])
    stock_data.fillna(0, inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close', 'MA100', 'MA200', 'RSI', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band']])

    if len(scaled_data) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough data to create a sequence of length {SEQUENCE_LENGTH}. Available: {len(scaled_data)} rows.")

    meta_features = get_meta_features(ticker, ohe, meta_data)
    meta_features = np.tile(meta_features, (SEQUENCE_LENGTH, 1))
    combined_data = np.concatenate((scaled_data[-SEQUENCE_LENGTH:], meta_features), axis=1)

    return torch.tensor(np.expand_dims(combined_data, axis=0), dtype=torch.float32), scaler, float(stock_data['Close'].values[-1])


# Main Pipeline
# Main Pipeline
def run_pipeline():
    print("Collecting live data...")

    # Collect live data, focusing on web scraping
    live_data = collect_live_data()

    # Prioritize web scraping results
    if 'headlines' in live_data and live_data['headlines']:
        print("Using web scraping results for sentiment analysis.")
        headlines = live_data['headlines']
    else:
        print("No web scraping results available. Using fallback data for sentiment analysis.")
        # Define fallback data or gracefully exit
        headlines = []

    # Analyze sentiment only if headlines are available
    if headlines:
        print("Analyzing sentiment of headlines...")
        avg_sentiment_score = np.mean(analyze_sentiment(headlines))
    else:
        print("No headlines available for sentiment analysis. Defaulting sentiment score to neutral (0.5).")
        avg_sentiment_score = 0.5

    # Handle stock data
    if 'stock_data' in live_data and live_data['stock_data'] is not None:
        print("Preparing stock data for LSTM...")
        stock_data = live_data['stock_data']
        ticker = os.getenv("DEFAULT_STOCK_TICKER", "AAPL")
        try:
            x_input, scaler, last_close_price = preprocess_stock_data(
                stock_data=stock_data,
                meta_data=meta_data,
                ohe=ohe,
                ticker=ticker
            )
        except ValueError as e:
            print(f"Error preprocessing stock data: {e}")
            return
    else:
        print("No stock data retrieved. Exiting pipeline.")
        return

    # Move input tensor to the same device as the model
    x_input = x_input.to(device)

    print("Predicting stock trend using LSTM...")
    lstm_prediction = lstm_model(x_input)

    # Debug: Print raw LSTM outputs
    print("Raw LSTM Prediction:", lstm_prediction)

    # Interpret regression output as stock trend
    stock_trend = 1 if lstm_prediction.item() > 0 else 0  # Positive trend if > 0
    print("Stock Trend:", stock_trend)


    # Decision score calculation
    decision_score = (avg_sentiment_score * 0.7) + (stock_trend * 0.3)

    if decision_score > 0.7:
        decision = "Strong Buy"
    elif 0.55 < decision_score <= 0.7:
        decision = "Moderate Buy"
    elif 0.4 <= decision_score <= 0.55:
        decision = "Hold"
    elif 0.25 < decision_score < 0.4:
        decision = "Moderate Sell"
    else:
        decision = "Strong Sell"
    print("\nResults:")
    print(f"Average Sentiment Score: {avg_sentiment_score:.2f}")
    print(f"Stock Trend Prediction: {'Positive' if stock_trend > 0 else 'Negative'}")
    print(f"Investment Decision: {decision}")





if __name__ == "__main__":
    run_pipeline()
