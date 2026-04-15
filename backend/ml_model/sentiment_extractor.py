"""
Sentiment feature extraction for Sentiment-Based Market Movement Prediction.

This module handles:
1. Loading a finance-tuned sentiment model (FinBERT or alternative)
2. Extracting sentiment scores from news headlines
3. Creating rolling aggregates
4. Augmenting the dataset with sentiment features

Reference: Phase 3 (Week 2-3) - Finance Sentiment Layer
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SENTIMENT_CONFIG, ARTIFACT_PATHS

logger = logging.getLogger(__name__)


def load_sentiment_model(model_name="FinBERT"):
    """
    Load a finance-tuned sentiment model.
    
    Phase 3 MVP: Uses basic transformers sentiment pipeline.
    Phase 7 production: Upgrade to fine-tuned FinBERT on finance data.
    
    Args:
        model_name (str): Model to load
    
    Returns:
        function: Sentiment analysis function
    """
    try:
        from transformers import pipeline
        
        if model_name == "FinBERT":
            logger.info("Loading FinBERT model (ProsusAI/finBERT)...")
            # Phase 7: Use real FinBERT checkpoint
            # model = pipeline("sentiment-analysis", model="ProsusAI/finBERT-tone")
            
            # Phase 3 MVP: Use default sentiment model
            logger.warning("FinBERT checkpoint not available; using default sentiment-analysis")
            sentiment_fn = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        else:
            logger.info(f"Loading {model_name} model...")
            sentiment_fn = pipeline("sentiment-analysis")
        
        return sentiment_fn
    
    except Exception as e:
        logger.error(f"Error loading sentiment model: {str(e)}")
        return None


def extract_sentiment_score(text, sentiment_model):
    """
    Extract sentiment score from text (0-1 scale).
    
    Args:
        text (str): Input text (e.g., aggregated headlines)
        sentiment_model: Loaded sentiment pipeline
    
    Returns:
        dict: {score: float [0, 1], label: str (POSITIVE/NEGATIVE/NEUTRAL)}
    """
    if not text or pd.isna(text) or len(str(text).strip()) == 0:
        return {"score": 0.5, "label": "NEUTRAL", "confidence": 0.5}
    
    try:
        # Truncate to avoid token limit (max 512 for BERT)
        text = str(text)[:512]
        
        result = sentiment_model(text)[0]
        
        # Normalize label to 0-1 score
        label = result["label"].upper()
        confidence = result["score"]
        
        if label == "POSITIVE":
            score = confidence
        elif label == "NEGATIVE":
            score = 1 - confidence
        else:
            score = 0.5
        
        return {
            "score": score,
            "label": label,
            "confidence": confidence,
        }
    
    except Exception as e:
        logger.warning(f"Error extracting sentiment: {str(e)}")
        return {"score": 0.5, "label": "NEUTRAL", "confidence": 0.0}


def extract_sentiment_features(unified_df, sentiment_model=None):
    """
    Extract sentiment features from aggregated texts.
    
    Phase 3 deliverable: This adds sentiment columns to the dataset.
    
    Args:
        unified_df (pd.DataFrame): Output from Phase 2 (pipeline.py)
        sentiment_model: Loaded model, default: load default
    
    Returns:
        pd.DataFrame: Dataset with sentiment features added
    """
    if sentiment_model is None:
        sentiment_model = load_sentiment_model()
    
    if sentiment_model is None:
        logger.error("Could not load sentiment model!")
        return unified_df
    
    logger.info(f"Extracting sentiment from {len(unified_df)} rows...")
    
    # Extract sentiment for each row
    sentiments = []
    for idx, row in unified_df.iterrows():
        text = row.get("aggregated_text", "")
        sentiment = extract_sentiment_score(text, sentiment_model)
        sentiments.append(sentiment)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(unified_df)}")
    
    sentiment_df = pd.DataFrame(sentiments)
    
    # Add to dataset
    result_df = pd.concat([unified_df.reset_index(drop=True), sentiment_df], axis=1)
    
    logger.info(f"✓ Extracted sentiment features for {len(result_df)} rows")
    
    return result_df


def create_rolling_sentiment_aggregates(df, windows=None):
    """
    Create rolling averages of sentiment scores by symbol.
    
    Phase 3 feature engineering: Rolling aggregates capture sentiment trends.
    
    Args:
        df (pd.DataFrame): Dataset with sentiment_score column
        windows (list): Rolling window sizes (days)
    
    Returns:
        pd.DataFrame: Dataset with rolling sentiment columns
    """
    if windows is None:
        windows = SENTIMENT_CONFIG.get("aggregation_windows", [1, 5, 20])
    
    logger.info(f"Creating rolling sentiment aggregates: {windows}")
    
    result_df = df.copy()
    
    for window in windows:
        col_name = f"sentiment_ma_{window}"
        
        # Group by symbol, sort by date, compute rolling mean
        result_df[col_name] = result_df.groupby("symbol")["score"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    logger.info(f"✓ Created {len(windows)} rolling sentiment features")
    
    return result_df


def augment_with_sentiment(input_file="unified_training_dataset.csv", output_file="sentiment_augmented_dataset.csv"):
    """
    Main Phase 3 function: Load Phase 2 output, add sentiment features, save.
    
    Args:
        input_file (str): Phase 2 output filename
        output_file (str): Phase 3 output filename
    
    Returns:
        pd.DataFrame: Augmented dataset
    """
    logger.info("=" * 80)
    logger.info("PHASE 3: SENTIMENT FEATURE EXTRACTION")
    logger.info("=" * 80)
    
    # Load Phase 2 output
    input_path = Path(ARTIFACT_PATHS["datasets_dir"]) / input_file
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run Phase 2 first: python backend/pipeline.py")
        return None
    
    logger.info(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Load sentiment model
    logger.info("\nLoading sentiment model...")
    sentiment_model = load_sentiment_model(SENTIMENT_CONFIG.get("model_class", "FinBERT"))
    
    if sentiment_model is None:
        logger.error("Could not load sentiment model")
        return None
    
    # Extract sentiment scores
    logger.info("\nExtracting sentiment scores...")
    df = extract_sentiment_features(df, sentiment_model)
    
    # Create rolling aggregates
    logger.info("\nCreating rolling aggregates...")
    df = create_rolling_sentiment_aggregates(df)
    
    # Save augmented dataset
    logger.info("\n" + "=" * 80)
    logger.info("SAVING AUGMENTED DATASET")
    logger.info("=" * 80)
    output_path = Path(ARTIFACT_PATHS["datasets_dir"]) / output_file
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to {output_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3 COMPLETE - SENTIMENT FEATURES READY FOR PHASE 4")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Phase 4: Feature engineering and model training")
    logger.info("2. Run: python backend/train.py")
    logger.info("=" * 80)
    
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run Phase 3
    augment_with_sentiment()
