import yfinance as yf
from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(__file__), '../env/.env')
load_dotenv(env_path)
SEQUENCE_LENGTH = 250
def fetch_stock_data(ticker=None, period="2y", interval="1d"):
    """
    Fetch historical stock data using Yahoo Finance API, ensuring sufficient rows for the sequence length.
    """
    periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    for p in periods:
        print(f"Fetching data for {ticker} with period '{p}' and interval '{interval}'...")
        stock = yf.Ticker(ticker)
        hist = stock.history(period=p, interval=interval)
        if len(hist) >= SEQUENCE_LENGTH:
            print(f"Data fetched for {ticker} with period: {p}")
            return hist
        print(f"Not enough data for period {p}. Trying a longer period...")
    raise ValueError(f"Unable to fetch sufficient data for ticker {ticker}.")


if __name__ == "__main__":
    stock_data = fetch_stock_data()
    print("Live Stock Data:")
    print(stock_data.head())

