Link to sentiment analysis model download: https://drive.google.com/file/d/1nUoK46fHinRQqc4jI5-qa23WJkoB4iOs/view?usp=sharing

Unzip in the stock-prediction-backend/sentiment_analysis/ directory.

# Core Functionality
This platform integrates two key predictive models to deliver comprehensive insights:

## Custom LSTM Architecture: 
Built specifically for time-series forecasting, the LSTM model predicts stock trends based on enriched historical data, including technical indicators like moving averages, Bollinger Bands, and Relative Strength Index (RSI). The model is optimized for accuracy and efficiency, making it capable of capturing complex market dynamics.


- Implements a multi-layer PyTorch LSTM network optimized for time-series forecasting.

- Predicts stock movement using a combination of short-term and long-term trends based on SMA, RSI, MACD, and Bollinger Bands.

- Incorporates 120 historical time steps, enabling deep insight into trend reversals and momentum shifts.

- Leverages attention mechanisms for enhanced interpretability and feature importance analysis.

- Trained on a large-scale dataset with real-time updates to ensure robustness against market fluctuations.

- Optimized for GPU acceleration, significantly improving training and inference speed.



## Fine-Tuned BERT Sentiment Analysis: 
By analyzing financial news headlines and social media sentiment, the platform classifies market sentiment as bullish, bearish, or neutral. This sentiment score is combined with LSTM predictions to produce a well-rounded investment recommendation.

The backend handles all data processing, model inference, and API communications. It also incorporates risk evaluation metrics like Value at Risk (VaR) to quantify investment risks, further enhancing decision-making capabilities.
The frontend delivers a dynamic interface, enabling users to analyze individual stocks, manage their portfolios, and visualize data through visuals.


# Key Features
### Comprehensive Stock Analysis: 
Predicts stock trends, evaluates risks, and provides investment recommendations (e.g., Strong Buy, Hold, Moderate Sell).
### Portfolio Insights: 
Aggregates risk (VaR), sentiment scores, and portfolio composition metrics to give users a holistic view of their investments.
### Interactive Visualizations: 
Pie charts for portfolio allocations and line charts for stock performance trends provide an engaging way to explore data.

