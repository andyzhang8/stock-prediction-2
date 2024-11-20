 
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import argparse
from main import run_pipeline, analyze_sentiment
from web_scraper.fetch_stock_data import fetch_stock_data
from web_scraper.collect_live_data import collect_live_data

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL')  # Default to AAPL if no ticker is provided
        print(f"Running prediction for ticker: {ticker}")

        # Pass the ticker to run_pipeline
        results = run_pipeline(ticker=ticker)

        return jsonify({
            "success": True,
            "results": results
        }), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/analyze_sentiment', methods=['POST'])
def sentiment_analysis():
    # Endpoint to analyze sentiment of headlines.
    # Expects JSON input with a list of headlines.
    try:
        data = request.json
        headlines = data.get('headlines', [])

        if not headlines:
            raise ValueError("No headlines provided for sentiment analysis.")

        sentiment_scores = analyze_sentiment(headlines)

        return jsonify({
            "success": True,
            "sentiment_scores": sentiment_scores
        }), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/fetch_dashboard_data', methods=['GET'])
def fetch_dashboard_data():
    """
    Endpoint to fetch stock data for a dashboard.
    Expects a stock ticker as a query parameter.
    """
    try:
        ticker = request.args.get('ticker', 'AAPL') 

        # Fetch stock data
        stock_data = fetch_stock_data(ticker=ticker)

        if stock_data is None:
            raise ValueError("No stock data available for the given ticker.")

        return jsonify({
            "success": True,
            "stock_data": stock_data.to_dict()
        }), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/portfolio', methods=['POST', 'GET'])
def portfolio():
    """
    Handle portfolio persistence in the backend.
    """
    if request.method == 'POST':
        try:
            portfolio_data = request.json.get('portfolio', [])
            with open('portfolio.json', 'w') as f:
                json.dump(portfolio_data, f)
            return jsonify({"success": True, "message": "Portfolio saved successfully."}), 200
        except Exception as e:
            print(traceback.format_exc())
            return jsonify({"success": False, "error": str(e)}), 500
    elif request.method == 'GET':
        try:
            if os.path.exists('portfolio.json'):
                with open('portfolio.json', 'r') as f:
                    portfolio_data = json.load(f)
                return jsonify({"success": True, "portfolio": portfolio_data}), 200
            else:
                return jsonify({"success": True, "portfolio": []}), 200
        except Exception as e:
            print(traceback.format_exc())
            return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
