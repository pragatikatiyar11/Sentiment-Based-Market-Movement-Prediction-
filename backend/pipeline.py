"""
Unified data pipeline for Sentiment-Based Market Movement Prediction.

This module orchestrates the full Phase 2 pipeline:
1. Collect market OHLCV data
2. Collect sentiment texts
3. Join by date and symbol
4. Create labels
5. Validate and export

Reference: Phase 2 (Week 1-2) - Data Ingestion and Alignment
"""

import logging
import pandas as pd
import sys
from pathlib import Path

from config import STOCK_UNIVERSE, ARTIFACT_PATHS
from data_collector import (
    fetch_ohlcv,
    create_market_labels,
    validate_dataset,
    save_dataset,
)
from sentiment_collector import (
    collect_sentiment_texts,
    save_sentiment_texts,
)

logger = logging.getLogger(__name__)


def build_training_pipeline(symbols=None, start_date=None, end_date=None, sentiment_lookback=3):
    """
    Build complete training dataset from market + sentiment data.
    
    Phase 2 deliverable: This is the input for Phase 3 feature engineering.
    
    Args:
        symbols (list): Stock symbols, default: STOCK_UNIVERSE
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
        sentiment_lookback (int): Days of news history to aggregate
    
    Returns:
        pd.DataFrame: Unified training dataset with market, labels, and sentiment texts
    """
    if symbols is None:
        symbols = STOCK_UNIVERSE
    
    logger.info("=" * 80)
    logger.info("PHASE 2 PIPELINE: UNIFIED DATASET CREATION")
    logger.info("=" * 80)
    
    # Step 1: Collect market data
    logger.info("\n[Step 1/4] Collecting market OHLCV data...")
    market_data_list = []
    for symbol in symbols:
        ohlcv = fetch_ohlcv(symbol, start_date, end_date)
        if ohlcv is not None:
            ohlcv["symbol"] = symbol
            ohlcv = ohlcv.reset_index()
            market_data_list.append(ohlcv)
        else:
            logger.warning(f"Skipped {symbol}")
    
    if not market_data_list:
        logger.error("No market data collected!")
        return None
    
    market_df = pd.concat(market_data_list, ignore_index=True)
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.strftime("%Y-%m-%d")
    logger.info(f"Collected {len(market_df)} market data rows")
    
    # Step 2: Create labels
    logger.info("\n[Step 2/4] Creating price labels...")
    labels_list = []
    for symbol in symbols:
        symbol_data = market_df[market_df["symbol"] == symbol].copy()
        if len(symbol_data) > 0:
            labels = create_market_labels(symbol_data.sort_values("date"))
            symbol_data["label"] = labels.values
            labels_list.append(symbol_data)
    
    if not labels_list:
        logger.error("No labels created!")
        return None
    
    market_df = pd.concat(labels_list, ignore_index=True)
    market_df = market_df.dropna(subset=["label"])
    logger.info(f"Created labels for {len(market_df)} rows")
    
    # Step 3: Collect sentiment texts
    logger.info("\n[Step 3/4] Collecting sentiment texts...")
    sentiment_df = collect_sentiment_texts(symbols, lookback_days=sentiment_lookback)
    if sentiment_df is not None:
        logger.info(f"Collected {len(sentiment_df)} sentiment text rows")
    else:
        logger.warning("Sentiment collection returned empty; proceeding with market data only")
        sentiment_df = pd.DataFrame()
    
    # Step 4: Join market + sentiment
    logger.info("\n[Step 4/4] Joining market data and sentiment texts...")
    if not sentiment_df.empty:
        # Merge on symbol and date
        unified_df = market_df.merge(
            sentiment_df,
            on=["date", "symbol"],
            how="left"
        )
        logger.info(f"Joined {len(unified_df)} rows with sentiment texts")
    else:
        # Just use market data
        unified_df = market_df.copy()
        unified_df["aggregated_text"] = ""
        unified_df["headline_count"] = 0
        logger.info("No sentiment texts available; using market data only")
    
    # Reorder and clean columns
    unified_df = unified_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    
    # Keep key columns
    key_cols = [
        "symbol", "date", "open", "high", "low", "close", "volume",
        "label", "aggregated_text", "headline_count"
    ]
    available_cols = [col for col in key_cols if col in unified_df.columns]
    unified_df = unified_df[available_cols]
    
    logger.info(f"\n✓ Final unified dataset: {len(unified_df)} rows")
    logger.info(f"✓ Symbols: {unified_df['symbol'].nunique()}")
    logger.info(f"✓ Date range: {unified_df['date'].min()} to {unified_df['date'].max()}")
    
    return unified_df


def validate_pipeline_output(unified_df):
    """
    Validate the unified dataset output.
    
    Returns:
        dict: Validation results
    """
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 80)
    
    results = {}
    
    # Schema
    required_cols = {"symbol", "date", "close", "label", "aggregated_text"}
    results["schema_valid"] = required_cols.issubset(set(unified_df.columns))
    logger.info(f"✓ Schema valid: {results['schema_valid']}")
    
    # Nulls in critical columns
    critical_nulls = unified_df[["close", "label"]].isnull().sum().sum()
    results["no_critical_nulls"] = critical_nulls == 0
    logger.info(f"✓ No nulls in critical columns: {results['no_critical_nulls']} ({critical_nulls} found)")
    
    # Label balance
    if "label" in unified_df.columns:
        counts = unified_df["label"].value_counts()
        balance = min(counts) / max(counts) if len(counts) == 2 else 0
        results["balanced_labels"] = balance > 0.3
        logger.info(f"✓ Balanced labels: {results['balanced_labels']} (ratio: {balance:.2%})")
        logger.info(f"  - UP (1): {counts.get(1, 0)}, DOWN (0): {counts.get(0, 0)}")
    
    # Duplicates
    dupes = unified_df.duplicated(subset=["symbol", "date"]).sum()
    results["no_duplicates"] = dupes == 0
    logger.info(f"✓ No duplicates: {results['no_duplicates']} ({dupes} found)")
    
    # Symbols
    results["all_symbols_present"] = unified_df["symbol"].nunique() == len(STOCK_UNIVERSE)
    logger.info(f"✓ All symbols present: {results['all_symbols_present']} ({unified_df['symbol'].nunique()}/{len(STOCK_UNIVERSE)})")
    
    # Sentiment coverage
    has_text = (unified_df["aggregated_text"].str.len() > 0).sum()
    coverage = has_text / len(unified_df) if len(unified_df) > 0 else 0
    logger.info(f"✓ Sentiment text coverage: {coverage:.1%} ({has_text}/{len(unified_df)} rows)")
    
    all_passed = all(results.values())
    logger.info(f"\n✓ All checks passed: {all_passed}")
    
    return results


def main():
    """Run the complete Phase 2 pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Build dataset
    unified_df = build_training_pipeline()
    
    if unified_df is None:
        logger.error("Pipeline failed!")
        return False
    
    # Validate
    validation_results = validate_pipeline_output(unified_df)
    
    if not all(validation_results.values()):
        logger.warning("Some validation checks failed, but proceeding...")
    
    # Save
    logger.info("\n" + "=" * 80)
    logger.info("SAVING DATASET")
    logger.info("=" * 80)
    save_dataset(unified_df, "unified_training_dataset.csv")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 COMPLETE - DATASET READY FOR PHASE 3")
    logger.info("=" * 80)
    logger.info(f"Output file: {Path(ARTIFACT_PATHS['datasets_dir']) / 'unified_training_dataset.csv'}")
    logger.info("\nNext steps:")
    logger.info("1. Phase 3: Integrate FinBERT sentiment extraction")
    logger.info("2. Run: python backend/sentiment_extractor.py")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
