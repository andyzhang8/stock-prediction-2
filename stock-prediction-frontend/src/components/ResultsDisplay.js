import React from "react";

const ResultsDisplay = ({ results }) => {
  if (!results) return null;

  const [varValue, sentimentScore, stockTrend, investmentDecision] = results;

  return (
    <div style={{ marginTop: "20px" }}>
      <h3>Analysis Results:</h3>
      <p><strong>Risk at Value (VaR at 5%):</strong> {varValue}%</p>
      <p><strong>Average Sentiment Score:</strong> {sentimentScore}</p>
      <p><strong>Stock Trend Prediction:</strong> {stockTrend > 0 ? "Positive" : "Negative"}</p>
      <p><strong>Investment Decision:</strong> {investmentDecision}</p>
    </div>
  );
};

export default ResultsDisplay;
