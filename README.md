Link to sentiment analysis model download: https://drive.google.com/file/d/1nUoK46fHinRQqc4jI5-qa23WJkoB4iOs/view?usp=sharing

Unzip in the sentiment_analysis/ directory

# Core Functionality
This platform integrates two key predictive models to deliver comprehensive insights:

## Custom LSTM Architecture: 
Built specifically for time-series forecasting, the LSTM model predicts stock trends based on enriched historical data, including technical indicators like moving averages, Bollinger Bands, and Relative Strength Index (RSI). The model is optimized for accuracy and efficiency, making it capable of capturing complex market dynamics.
## Fine-Tuned BERT Sentiment Analysis: 
By analyzing financial news headlines and social media sentiment, the platform classifies market sentiment as bullish, bearish, or neutral. This sentiment score is combined with LSTM predictions to produce a well-rounded investment recommendation.

The backend handles all data processing, model inference, and API communications. It also incorporates risk evaluation metrics like Value at Risk (VaR) to quantify investment risks, further enhancing decision-making capabilities.
The frontend delivers a dynamic interface, enabling users to analyze individual stocks, manage their portfolios, and visualize data through visuals.

T Technical Details
# 
## Backend
### Machine Learning: 
A custom-built LSTM model predicts stock trends, while a fine-tuned BERT model performs sentiment analysis. TensorFlow and PyTorch were used to implement and train these models.
### Data Processing: 
Time-series data is enriched with advanced technical indicators using pandas and NumPy. VaR calculations quantify risk exposure.
### Web Scraping: 
Real-time stock data and financial headlines are fetched from sources like Yahoo Finance using Python's BeautifulSoup and requests libraries.
### Flask API: 
The backend is powered by Flask, exposing robust endpoints for stock analysis, portfolio management, and sentiment scoring.

## Frontend
### React: 
A modern, responsive UI provides an interactive experience for users, with seamless communication to the backend via Axios.
### Material-UI: 
Used for professional and accessible design, ensuring the dashboard is visually appealing and intuitive.
### Chart.js: 
Implements pie charts for portfolio allocation and line charts for visualizing historical stock trends and sentiment over time.
### State Management: 
Portfolio data is persistently stored using localStorage, offering flexibility for users without requiring account-based systems.

# Key Features
Comprehensive Stock Analysis: Predicts stock trends, evaluates risks, and provides investment recommendations (e.g., Strong Buy, Hold, Moderate Sell).
Portfolio Insights: Aggregates risk (VaR), sentiment scores, and portfolio composition metrics to give users a holistic view of their investments.
Interactive Visualizations: Pie charts for portfolio allocations and line charts for stock performance trends provide an engaging way to explore data.

