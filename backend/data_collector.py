"""
Data collection and preparation for Sentiment-Based Market Movement Prediction.

This module handles:
1. Market OHLCV data ingestion from yfinance
2. News headline fetching from NewsAPI
3. Data alignment and preprocessing
4. Training dataset creation

Reference: Phase 2 (Week 1-2) - Data Ingestion and Alignment
"""

import logging
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
from pathlib import Path

from config import (
    STOCK_UNIVERSE,
    DATA_SOURCES,
    LABEL_WINDOW_DAYS,
    DIRECTION_THRESHOLD,
    ARTIFACT_PATHS,
)

# ============================================================================
# Setup
# ============================================================================

logger = logging.getLogger(__name__)

# Create data directories
Path(ARTIFACT_PATHS["datasets_dir"]).mkdir(parents=True, exist_ok=True)
Path(ARTIFACT_PATHS["logs_dir"]).mkdir(parents=True, exist_ok=True)
Path(ARTIFACT_PATHS["cache_dir"]).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Market Data Ingestion (yfinance)
# ============================================================================

def fetch_ohlcv(symbol, start_date=None, end_date=None):
    """
    Fetch OHLCV data from Yahoo Finance.
    
    Args:
        symbol (str): Stock ticker (e.g., 'AAPL')
        start_date (str): YYYY-MM-DD format, default: 3 years ago
        end_date (str): YYYY-MM-DD format, default: today
    
    Returns:
        pd.DataFrame: Columns [Open, High, Low, Close, Volume] with DatetimeIndex
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=252*3)).strftime("%Y-%m-%d")
    
    try:
        logger.info(f"Fetching OHLCV for {symbol} ({start_date} to {end_date})")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        # Ensure DatetimeIndex and rename columns to lowercase
        data.index.name = "date"
        data.columns = [col.lower() for col in data.columns]
        
        logger.info(f"Fetched {len(data)} rows for {symbol}")
        return data
    
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {str(e)}")
        return None


def create_market_labels(ohlcv_df, label_window=LABEL_WINDOW_DAYS, threshold=DIRECTION_THRESHOLD):
    """
    Create binary labels: 1 if price goes UP, 0 if DOWN.
    
    Args:
        ohlcv_df (pd.DataFrame): OHLCV data with 'Close' column
        label_window (int): Number of days ahead to predict
        threshold (float): Percentage change threshold (0.0 = any positive change is UP)
    
    Returns:
        pd.Series: Binary labels (1=UP, 0=DOWN)
    """
    close_prices = ohlcv_df["close"].copy()
    
    # Shift to get future price at t+label_window
    future_price = close_prices.shift(-label_window)
    
    # Calculate % change
    pct_change = (future_price - close_prices) / close_prices * 100
    
    # Create label: 1 if up, 0 if down
    labels = (pct_change > threshold).astype(int)
    
    logger.info(f"Created labels: {labels.sum()} UP, {(1-labels).sum()} DOWN")
    
    return labels


def fetch_news_headlines(symbol, api_key=None, lookback_days=1):
    """
    Fetch news headlines from NewsAPI.
    
    Phase 2 implementation note: Requires newsapi.org API key in environment or config.
    For MVP, can mock or skip if rate-limited.
    
    Args:
        symbol (str): Stock ticker
        api_key (str): NewsAPI key, default: from environment NEWSAPI_KEY
        lookback_days (int): Include news from past N days
    
    Returns:
        pd.DataFrame: Columns [date, headline, source, url, sentiment_text]
                     Placeholder until Phase 2 full implementation
    """
    api_key = api_key or os.getenv("NEWSAPI_KEY", None)
    
    if api_key is None:
        logger.warning("NEWSAPI_KEY not found; returning mock headlines")
        # Phase 2: Mock data for testing
        mock_headlines = [
            f"{symbol} stock rallies on strong earnings",
            f"{symbol} faces regulatory scrutiny",
            f"Analysts rate {symbol} as top buy",
        ]
        return pd.DataFrame({
            "date": [datetime.now()] * len(mock_headlines),
            "headline": mock_headlines,
            "source": ["NewsAPI"] * len(mock_headlines),
        })
    
    try:
        import requests
        from newsapi import NewsApiClient
        
        logger.info(f"Fetching news for {symbol} (past {lookback_days} days)")
        
        # Phase 2: Implement actual API call here
        # newsapi = NewsApiClient(api_key=api_key)
        # articles = newsapi.get_everything(q=symbol, ...)
        
        logger.warning("NewsAPI integration not yet implemented (Phase 2)")
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# Data Alignment and Dataset Creation
# ============================================================================

def create_training_dataset(symbols=None, start_date=None, end_date=None):
    """
    Create a unified training dataset with market data + labels.
    
    Phase 2 deliverable: This dataset is the input for Phase 3 (sentiment features).
    
    Args:
        symbols (list): Stock symbols, default: STOCK_UNIVERSE
        start_date (str): YYYY-MM-DD, default: 3 years ago
        end_date (str): YYYY-MM-DD, default: today
    
    Returns:
        pd.DataFrame: Columns [symbol, date, open, high, low, close, volume, label]
    """
    if symbols is None:
        symbols = STOCK_UNIVERSE
    
    all_data = []
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        # Fetch OHLCV
        ohlcv = fetch_ohlcv(symbol, start_date, end_date)
        if ohlcv is None:
            logger.warning(f"Skipping {symbol} due to fetch error")
            continue
        
        # Create labels
        labels = create_market_labels(ohlcv)
        
        # Combine
        ohlcv["label"] = labels
        ohlcv["symbol"] = symbol
        ohlcv = ohlcv.reset_index()
        
        all_data.append(ohlcv)
    
    if not all_data:
        logger.error("No data collected!")
        return None
    
    # Concatenate all symbols
    dataset = pd.concat(all_data, ignore_index=True)
    
    # Remove rows with NaN labels (future window at dataset end)
    dataset = dataset.dropna(subset=["label"])
    
    # Sort by symbol and date
    dataset = dataset.sort_values(["symbol", "date"]).reset_index(drop=True)
    
    logger.info(f"Created dataset: {len(dataset)} rows, {dataset['symbol'].nunique()} symbols")
    
    return dataset


def save_dataset(dataset, filename="training_dataset.csv"):
    """Save dataset to disk."""
    filepath = Path(ARTIFACT_PATHS["datasets_dir"]) / filename
    dataset.to_csv(filepath, index=False)
    logger.info(f"Dataset saved to {filepath}")
    return filepath


def load_dataset(filename="training_dataset.csv"):
    """Load dataset from disk."""
    filepath = Path(ARTIFACT_PATHS["datasets_dir"]) / filename
    if not filepath.exists():
        logger.error(f"Dataset not found: {filepath}")
        return None
    dataset = pd.read_csv(filepath)
    logger.info(f"Loaded {len(dataset)} rows from {filepath}")
    return dataset


# ============================================================================
# Data Quality Checks
# ============================================================================

def validate_dataset(dataset):
    """
    Run validation checks on the dataset.
    
    Returns:
        dict: Validation results {check_name: passed (bool)}
    """
    checks = {}
    
    # Schema
    required_cols = {"symbol", "date", "close", "label"}
    checks["schema"] = required_cols.issubset(set(dataset.columns))
    logger.info(f"Schema check: {checks['schema']}")
    
    # Nulls
    checks["no_nulls"] = dataset[["close", "label"]].isnull().sum().sum() == 0
    logger.info(f"Nulls check: {checks['no_nulls']}")
    
    # Label balance
    label_counts = dataset["label"].value_counts()
    balance = min(label_counts) / max(label_counts)
    checks["balanced"] = balance > 0.3  # At least 30/70 split
    logger.info(f"Label balance: {label_counts.to_dict()} (ratio: {balance:.2f})")
    
    # No duplicates
    checks["no_duplicates"] = dataset.drop_duplicates(subset=["symbol", "date"]).shape[0] == dataset.shape[0]
    logger.info(f"Duplicates check: {checks['no_duplicates']}")
    
    return checks


# ============================================================================
# Main: Build Dataset
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("=" * 80)
    logger.info("PHASE 2: DATA COLLECTION & ALIGNMENT")
    logger.info("=" * 80)
    
    # Build dataset
    dataset = create_training_dataset()
    
    if dataset is not None:
        # Validate
        validation_results = validate_dataset(dataset)
        logger.info(f"Validation results: {validation_results}")
        
        # Save
        save_dataset(dataset)
        
        logger.info("=" * 80)
        logger.info("Phase 2 complete. Dataset ready for Phase 3 (sentiment features).")
        logger.info("=" * 80)
    else:
        logger.error("Dataset creation failed")
        sys.exit(1)
