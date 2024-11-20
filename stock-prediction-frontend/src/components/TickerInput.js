import React, { useState } from "react";

const TickerInput = ({ onSubmit }) => {
  const [ticker, setTicker] = useState("");
  const [amount, setAmount] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!ticker.trim()) {
      setError("Please enter a valid stock ticker.");
      return;
    }

    if (!amount || isNaN(amount) || Number(amount) <= 0) {
      setError("Please enter a valid amount of stock.");
      return;
    }

    setError(""); // Clear any previous error
    onSubmit(ticker.trim().toUpperCase(), Number(amount)); // Pass both ticker and amount to the parent component
    setTicker(""); // Reset the ticker input field
    setAmount(""); // Reset the amount input field
  };

  return (
    <div style={{ margin: "20px 0", textAlign: "center" }}>
      <form onSubmit={handleSubmit} style={{ display: "inline-block" }}>
        <input
          type="text"
          placeholder="Enter stock ticker (e.g., AAPL)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          style={{
            padding: "10px",
            width: "200px",
            marginRight: "10px",
            border: "1px solid #ccc",
            borderRadius: "4px",
          }}
        />
        <input
          type="number"
          placeholder="Enter amount of stock"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
          style={{
            padding: "10px",
            width: "200px",
            marginRight: "10px",
            border: "1px solid #ccc",
            borderRadius: "4px",
          }}
        />
        <button
          type="submit"
          style={{
            padding: "10px 20px",
            backgroundColor: "#007bff",
            color: "#fff",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Analyze
        </button>
      </form>
      {error && (
        <div
          style={{
            marginTop: "10px",
            color: "red",
            fontSize: "14px",
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
};

export default TickerInput;
