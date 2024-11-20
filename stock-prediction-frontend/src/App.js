import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
} from "chart.js";
import { Pie, Line } from "react-chartjs-2";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import {
  Button,
  TextField,
  Typography,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Box,
  Paper,
} from "@mui/material";

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, LineElement, PointElement);

const theme = createTheme({
  palette: {
    primary: {
      main: "#00796b",
    },
    secondary: {
      main: "#8e24aa",
    },
  },
  typography: {
    fontFamily: "Roboto, Arial, sans-serif",
  },
});

const App = () => {
  const [ticker, setTicker] = useState("");
  const [results, setResults] = useState(null);
  const [portfolio, setPortfolio] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [historicalPrices, setHistoricalPrices] = useState([]);
  const [sentimentTrends, setSentimentTrends] = useState([]);
  const [stockAmount, setStockAmount] = useState(0);

  useEffect(() => {
    const savedPortfolio = JSON.parse(localStorage.getItem("portfolio")) || [];
    setPortfolio(savedPortfolio);
  }, []);

  const savePortfolio = (newPortfolio) => {
    localStorage.setItem("portfolio", JSON.stringify(newPortfolio));
    setPortfolio(newPortfolio);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await axios.post("http://localhost:5000/api/predict", { ticker });
      if (response.data.success) {
        setResults(response.data.results);
        setHistoricalPrices(response.data.historicalPrices || []);
        setSentimentTrends(response.data.sentimentTrends || []);
      } else {
        setError(response.data.error || "An error occurred.");
      }
    } catch (err) {
      setError("Failed to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveToPortfolio = () => {
    if (!results) return;
    const updatedPortfolio = [...portfolio, { ticker, results, stockAmount }];
    savePortfolio(updatedPortfolio);
  };

  const handleRemoveFromPortfolio = (index) => {
    const updatedPortfolio = portfolio.filter((_, i) => i !== index);
    savePortfolio(updatedPortfolio);
  };

  const calculatePortfolioInsights = () => {
    if (portfolio.length === 0) return {};
    const totalVaR = portfolio.reduce((acc, stock) => acc + stock.results[0] * stock.stockAmount, 0);
    const totalAmount = portfolio.reduce((acc, stock) => acc + stock.stockAmount, 0);
    const avgSentiment =
      portfolio.reduce((acc, stock) => acc + stock.results[1] * stock.stockAmount, 0) / totalAmount;

    return {
      totalVaR: totalVaR / totalAmount,
      avgSentiment,
    };
  };

  const portfolioInsights = calculatePortfolioInsights();

  const portfolioData = {
    labels: portfolio.map((stock) => stock.ticker),
    datasets: [
      {
        label: "Portfolio Allocation",
        data: portfolio.map((stock) => stock.stockAmount),
        backgroundColor: ["#ff6384", "#36a2eb", "#ffcd56", "#4bc0c0"],
        borderWidth: 1,
      },
    ],
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ padding: 3, maxWidth: "800px", margin: "0 auto" }}>
        <Typography variant="h4" gutterBottom align="center" color="primary">
          Stock Analysis Dashboard
        </Typography>
        <Paper elevation={3} sx={{ padding: 3, marginBottom: 3 }}>
          <Typography variant="h6" gutterBottom>
            Enter Stock Ticker and Amount
          </Typography>
          <Box display="flex" gap={2}>
            <TextField
              label="Ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              variant="outlined"
              fullWidth
            />
            <TextField
              label="Amount"
              type="number"
              value={stockAmount}
              onChange={(e) => setStockAmount(Number(e.target.value))}
              variant="outlined"
              fullWidth
            />
            <Button onClick={handlePredict} disabled={loading} variant="contained" color="primary">
              Analyze
            </Button>
          </Box>
        </Paper>
        {loading && <Typography>Loading...</Typography>}
        {error && <Typography color="error">{error}</Typography>}
        {results && (
          <Card sx={{ marginBottom: 3 }}>
            <CardContent>
              <Typography variant="h5">Analysis Results</Typography>
              <Typography>Risk at Value (VaR at 5%): {results[0].toFixed(2)}%</Typography>
              <Typography>Average Sentiment Score: {results[1]}</Typography>
              <Typography>Stock Trend Prediction: {results[2] ? "Positive" : "Negative"}</Typography>
              <Typography>Investment Decision: {results[3]}</Typography>
              <Button onClick={handleSaveToPortfolio} variant="contained" color="secondary" sx={{ marginTop: 2 }}>
                Save to Portfolio
              </Button>
            </CardContent>
          </Card>
        )}
        {portfolio.length > 0 && (
          <div className="portfolio-section">
            <Typography variant="h5" gutterBottom>
              Your Portfolio
            </Typography>
            <List>
              {portfolio.map((stock, index) => (
                <ListItem key={index} sx={{ display: "flex", justifyContent: "space-between" }}>
                  <ListItemText
                    primary={`${stock.ticker} (${stock.stockAmount} shares)`}
                    secondary={`VaR: ${stock.results[0].toFixed(2)}%, Decision: ${stock.results[3]}`}
                  />
                  <Button
                    onClick={() => handleRemoveFromPortfolio(index)}
                    color="secondary"
                    variant="outlined"
                  >
                    Remove
                  </Button>
                </ListItem>
              ))}
            </List>
          </div>
        )}
        {portfolio.length > 0 && (
          <div className="chart-section" style={{ marginTop: 20 }}>
            <Typography variant="h6">Portfolio Insights</Typography>
            <Typography>Average Portfolio VaR: {portfolioInsights.totalVaR.toFixed(2)}%</Typography>
            <Typography>Average Sentiment Score: {portfolioInsights.avgSentiment.toFixed(2)}</Typography>
            <div style={{ width: "400px", height: "400px", margin: "20px auto" }}>
              <Pie
                data={portfolioData}
                options={{
                  maintainAspectRatio: false,
                }}
              />
            </div>
          </div>
        )}
      </Box>
    </ThemeProvider>
  );
};

export default App;
