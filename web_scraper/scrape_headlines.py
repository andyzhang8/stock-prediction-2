import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '../env/.env')
load_dotenv(env_path)

def scrape_headlines_from_finviz(stock_ticker):
    """
    Scrape stock news headlines from Finviz for a given ticker.
    Args:
        stock_ticker (str): Stock ticker symbol.
    Returns:
        list: List of scraped headlines.
    """
    try:
        # Finviz URL for the given stock ticker
        url = f'https://finviz.com/quote.ashx?t={stock_ticker}&p=d'
        request = Request(url=url, headers={'user-agent': 'news_scraper'})
        response = urlopen(request)
        
        # Parse the page
        html = BeautifulSoup(response, features='html.parser')
        news_table = html.find(id='news-table')

        # Extract headlines
        headlines = []
        for row in news_table.findAll('tr'):
            try:
                headline = row.a.getText()
                headlines.append(headline)
            except AttributeError:
                continue
        
        return headlines
    except Exception as e:
        print(f"Error scraping Finviz for {stock_ticker}: {e}")
        return []

def scrape_headlines():
    """
    Aggregate headlines from multiple sources, including Finviz.
    Returns:
        list: Aggregated list of headlines.
    """
    stock_ticker = os.getenv("DEFAULT_STOCK_TICKER", "AAPL")  # Default stock ticker if not set in .env
    print(f"Scraping headlines for stock ticker: {stock_ticker}")
    
    headlines = scrape_headlines_from_finviz(stock_ticker)
    
    # If more sources are added, append their headlines here
    # Example: headlines += scrape_other_source(stock_ticker)
    
    return headlines

if __name__ == "__main__":
    headlines = scrape_headlines()
    print("Scraped Headlines:")
    print(headlines)

    # Save to CSV if needed
    df = pd.DataFrame(headlines, columns=["Headline"])
    df.to_csv("scraped_headlines.csv", index=False)
    print("Headlines saved to scraped_headlines.csv")
