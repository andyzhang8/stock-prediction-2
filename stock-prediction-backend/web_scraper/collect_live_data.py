from .scrape_headlines import scrape_headlines
from .fetch_stock_data import fetch_stock_data
from dotenv import load_dotenv
import os
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), '../env/.env')
load_dotenv(env_path)


def preprocess_texts(texts):
    return [text.strip() for text in texts if isinstance(text, str) and text.strip()]


def prepare_stock_data(stock_data):
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)
    return stock_data.values[-60:]  # Last 60 days


def collect_live_data(ticker=None):
    # Default to provided ticker or environment variable
    stock_ticker = ticker or os.getenv("DEFAULT_STOCK_TICKER", "AAPL")

    print(f"Fetching headlines for stock ticker: {stock_ticker}...")
    headlines = preprocess_texts(scrape_headlines(stock_ticker))

    print(f"Fetching stock data for {stock_ticker}...")
    stock_data = fetch_stock_data(ticker=stock_ticker)
    
    # Debug: Print the stock data after fetching
    print(f"Fetched stock data columns for {stock_ticker}: {stock_data.columns}")
    
    return {
        "headlines": headlines,
        "stock_data": stock_data
    }


if __name__ == "__main__":
    live_data = collect_live_data()
    print("\nLive Data:")
    print("Headlines:")
    for headline in live_data["headlines"]:
        print(f"- {headline}")
    print("\nStock Data:")
    print(live_data["stock_data"])
