"""
Sentiment text aggregation for Sentiment-Based Market Movement Prediction.

This module handles:
1. Fetching news headlines via NewsAPI
2. Aggregating sentiment text by stock and date
3. Storing/retrieving sentiment history

Reference: Phase 2 (Week 1-2) - Data Ingestion and Alignment
"""

import logging
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path

from config import STOCK_UNIVERSE, ARTIFACT_PATHS

logger = logging.getLogger(__name__)

# Create directories
Path(ARTIFACT_PATHS["datasets_dir"]).mkdir(parents=True, exist_ok=True)


def get_mock_headlines(symbol, lookback_days=1):
    """
    Generate mock news headlines for MVP development.
    
    In production (Phase 7), replace with real NewsAPI integration.
    
    Args:
        symbol (str): Stock ticker
        lookback_days (int): Days of history to mock
    
    Returns:
        pd.DataFrame: Columns [date, symbol, headline, source]
    """
    mock_sentiments = {
        "positive": [
            f"{symbol} stock rallies on strong earnings report",
            f"{symbol} announces record quarterly revenue",
            f"Analysts upgrade {symbol} to buy rating",
            f"{symbol} wins major new contract",
            f"Investors bullish on {symbol} expansion plans",
        ],
        "neutral": [
            f"{symbol} to hold analyst meeting next week",
            f"{symbol} stock trading near 52-week average",
            f"Market watch: {symbol} in focus today",
        ],
        "negative": [
            f"{symbol} faces regulatory investigation",
            f"{symbol} stock falls on weaker-than-expected guidance",
            f"Analysts cut {symbol} price target",
            f"{symbol} hit by supply chain disruptions",
            f"Concerns grow over {symbol} market position",
        ],
    }
    
    headlines = []
    base_date = datetime.now()
    
    for day_offset in range(lookback_days):
        date = base_date - timedelta(days=day_offset)
        
        # Randomly select from sentiments for variety
        import random
        sentiment_type = random.choice(list(mock_sentiments.keys()))
        headline = random.choice(mock_sentiments[sentiment_type])
        
        headlines.append({
            "date": date.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "headline": headline,
            "source": "MockNews",
        })
    
    return pd.DataFrame(headlines)


def fetch_newsapi_headlines(symbol, api_key=None, lookback_days=1):
    """
    Fetch real news headlines from NewsAPI.
    
    Phase 2 MVP: Uses mock data by default. Uncomment real API for production.
    
    Args:
        symbol (str): Stock ticker
        api_key (str): NewsAPI key (from env or param)
        lookback_days (int): Days to fetch
    
    Returns:
        pd.DataFrame: Columns [date, symbol, headline, source, url]
    """
    api_key = api_key or os.getenv("NEWSAPI_KEY", None)
    
    # Phase 2: For MVP, use mock data to avoid rate limits
    logger.warning(f"Using mock headlines for {symbol} (set NEWSAPI_KEY env var for real data)")
    return get_mock_headlines(symbol, lookback_days)
    
    # Phase 7 production version (uncomment when ready):
    # try:
    #     from newsapi import NewsApiClient
    #     newsapi = NewsApiClient(api_key=api_key)
    #     articles = newsapi.get_everything(
    #         q=symbol,
    #         sort_by="publishedAt",
    #         language="en",
    #         from_param=from_date,
    #     )
    #     headlines = []
    #     for article in articles.get("articles", []):
    #         headlines.append({
    #             "date": article["publishedAt"][:10],
    #             "symbol": symbol,
    #             "headline": article["title"],
    #             "source": article["source"]["name"],
    #             "url": article["url"],
    #         })
    #     return pd.DataFrame(headlines)
    # except Exception as e:
    #     logger.error(f"Error fetching news for {symbol}: {str(e)}")
    #     return get_mock_headlines(symbol, lookback_days)


def aggregate_headlines_by_date(headlines_df):
    """
    Aggregate multiple headlines for a symbol into a single text field per date.
    
    Args:
        headlines_df (pd.DataFrame): Columns [date, symbol, headline, source]
    
    Returns:
        pd.DataFrame: Columns [date, symbol, aggregated_text, headline_count]
    """
    aggregated = []
    
    for (date, symbol), group in headlines_df.groupby(["date", "symbol"]):
        headlines = group["headline"].tolist()
        aggregated_text = " | ".join(headlines)
        
        aggregated.append({
            "date": date,
            "symbol": symbol,
            "aggregated_text": aggregated_text,
            "headline_count": len(headlines),
            "sources": ", ".join(group["source"].unique()),
        })
    
    return pd.DataFrame(aggregated)


def collect_sentiment_texts(symbols=None, lookback_days=3):
    """
    Collect sentiment texts (news headlines) for all symbols.
    
    Args:
        symbols (list): Stock symbols, default: STOCK_UNIVERSE
        lookback_days (int): Days of history to collect
    
    Returns:
        pd.DataFrame: Aggregated sentiment texts by symbol and date
    """
    if symbols is None:
        symbols = STOCK_UNIVERSE
    
    all_headlines = []
    
    for symbol in symbols:
        logger.info(f"Collecting headlines for {symbol}...")
        headlines = fetch_newsapi_headlines(symbol, lookback_days=lookback_days)
        all_headlines.append(headlines)
    
    if not all_headlines:
        logger.error("No headlines collected!")
        return None
    
    combined = pd.concat(all_headlines, ignore_index=True)
    aggregated = aggregate_headlines_by_date(combined)
    
    logger.info(f"Collected {len(aggregated)} aggregated sentiment texts")
    
    return aggregated


def save_sentiment_texts(sentiment_df, filename="sentiment_texts.csv"):
    """Save sentiment texts to disk."""
    filepath = Path(ARTIFACT_PATHS["datasets_dir"]) / filename
    sentiment_df.to_csv(filepath, index=False)
    logger.info(f"Sentiment texts saved to {filepath}")
    return filepath


def load_sentiment_texts(filename="sentiment_texts.csv"):
    """Load sentiment texts from disk."""
    filepath = Path(ARTIFACT_PATHS["datasets_dir"]) / filename
    if not filepath.exists():
        logger.warning(f"Sentiment texts not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} sentiment text records")
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("=" * 80)
    logger.info("PHASE 2: SENTIMENT TEXT AGGREGATION")
    logger.info("=" * 80)
    
    # Collect headlines
    sentiment_df = collect_sentiment_texts(lookback_days=3)
    
    if sentiment_df is not None:
        # Save
        save_sentiment_texts(sentiment_df)
        logger.info(sentiment_df.head(10))
        logger.info("=" * 80)
        logger.info("Sentiment text collection complete.")
        logger.info("=" * 80)
    else:
        logger.error("Sentiment text collection failed")
