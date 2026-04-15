"""
Model training for Sentiment-Based Market Movement Prediction.

This module handles:
1. Feature engineering (technical + sentiment)
2. Train/validation/test splitting
3. Model training (multiple candidates)
4. Model selection and serialization

Reference: Phase 4 (Week 3-4) - Prediction Model and Feature Engineering
"""

import logging
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import TRAINING_CONFIG, ARTIFACT_PATHS, CANDIDATE_MODELS

logger = logging.getLogger(__name__)

# Create directories
Path(ARTIFACT_PATHS["models_dir"]).mkdir(parents=True, exist_ok=True)


def load_dataset(filename="sentiment_augmented_dataset.csv"):
    """Load Phase 3 output."""
    filepath = Path(ARTIFACT_PATHS["datasets_dir"]) / filename
    if not filepath.exists():
        logger.error(f"Dataset not found: {filepath}")
        logger.info("Run Phase 3 first: python backend/ml_model/sentiment_extractor.py")
        return None
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def engineer_features(df):
    """
    Engineer features for modeling.
    
    Phase 4 core: Combines market, technical, and sentiment features.
    
    Args:
        df (pd.DataFrame): Sentiment-augmented dataset from Phase 3
    
    Returns:
        tuple: (features_df, target_series, feature_names_list)
    """
    logger.info("Engineering features...")
    
    features_dict = {}
    
    # Market features
    logger.info("  - Adding market features (price, volume)")
    features_dict["close_price"] = df["close"]
    features_dict["volume"] = df["volume"]
    
    # Price change (log return)
    features_dict["price_change"] = df.groupby("symbol")["close"].pct_change()
    
    # Technical indicators (phase 4 expansion)
    logger.info("  - Adding technical indicators (SMA, volatility)")
    for window in [5, 20]:
        features_dict[f"sma_{window}"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        features_dict[f"volatility_{window}"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Sentiment features (from Phase 3)
    logger.info("  - Adding sentiment features")
    for col in ["score", "sentiment_ma_1", "sentiment_ma_5", "sentiment_ma_20"]:
        if col in df.columns:
            features_dict[col] = df[col]
    
    # Headline count (signals confidence)
    if "headline_count" in df.columns:
        features_dict["headline_count"] = df["headline_count"]
    
    # Lag features (momentum)
    logger.info("  - Adding lag features (price momentum)")
    for lag in [1, 5]:
        features_dict[f"price_lag_{lag}"] = df.groupby("symbol")["close"].shift(lag)
    
    features_df = pd.DataFrame(features_dict)
    
    # Handle NaN values
    logger.info(f"  - Handling missing values ({features_df.isnull().sum().sum()} nulls)")
    features_df = features_df.fillna(method="bfill").fillna(method="ffill").fillna(0)
    
    logger.info(f"✓ Engineered {len(features_df.columns)} features")
    
    return features_df, df["label"], list(features_df.columns)


def split_data(features_df, target_series):
    """
    Split data into train/validation/test.
    
    Uses time-series aware split: no data leakage from future to past.
    
    Args:
        features_df (pd.DataFrame): Features
        target_series (pd.Series): Labels
    
    Returns:
        dict: Train/val/test splits with features and targets
    """
    logger.info("Splitting data (time-series aware)...")
    
    n = len(features_df)
    test_ratio = TRAINING_CONFIG["test_split_ratio"]
    val_ratio = TRAINING_CONFIG["validation_split_ratio"]
    
    # Split based on time order (first to last)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * (1 - val_ratio))
    
    train_idx = range(0, val_start)
    val_idx = range(val_start, test_start)
    test_idx = range(test_start, n)
    
    splits = {
        "train": {
            "X": features_df.iloc[train_idx],
            "y": target_series.iloc[train_idx],
        },
        "val": {
            "X": features_df.iloc[val_idx],
            "y": target_series.iloc[val_idx],
        },
        "test": {
            "X": features_df.iloc[test_idx],
            "y": target_series.iloc[test_idx],
        },
    }
    
    logger.info(f"  Train: {len(splits['train']['X'])} | Val: {len(splits['val']['X'])} | Test: {len(splits['test']['X'])}")
    
    return splits


def scale_features(splits):
    """Standardize features."""
    logger.info("Scaling features...")
    
    scaler = StandardScaler()
    
    # Fit on train, apply to all
    splits["train"]["X"] = scaler.fit_transform(splits["train"]["X"])
    splits["val"]["X"] = scaler.transform(splits["val"]["X"])
    splits["test"]["X"] = scaler.transform(splits["test"]["X"])
    
    return splits, scaler


def train_candidate_models(splits, candidates=None):
    """
    Train multiple candidate models and evaluate.
    
    Phase 4: Compare logistic regression, random forest, etc.
    
    Args:
        splits (dict): Train/val/test splits
        candidates (list): Model names to try
    
    Returns:
        dict: Results for each model
    """
    if candidates is None:
        candidates = ["logistic_regression", "random_forest"]
    
    logger.info(f"\nTraining {len(candidates)} candidate models...")
    results = {}
    models = {}
    
    # Logistic Regression
    if "logistic_regression" in candidates:
        logger.info("  [1/2] Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(splits["train"]["X"], splits["train"]["y"])
        
        lr_pred = lr_model.predict(splits["val"]["X"])
        lr_acc = accuracy_score(splits["val"]["y"], lr_pred)
        
        logger.info(f"    Validation accuracy: {lr_acc:.3f}")
        results["logistic_regression"] = lr_acc
        models["logistic_regression"] = lr_model
    
    # Random Forest
    if "random_forest" in candidates:
        logger.info("  [2/2] Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(splits["train"]["X"], splits["train"]["y"])
        
        rf_pred = rf_model.predict(splits["val"]["X"])
        rf_acc = accuracy_score(splits["val"]["y"], rf_pred)
        
        logger.info(f"    Validation accuracy: {rf_acc:.3f}")
        results["random_forest"] = rf_acc
        models["random_forest"] = rf_model
    
    return results, models


def select_best_model(results, models):
    """Select best performing model."""
    best_name = max(results, key=results.get)
    best_score = results[best_name]
    
    logger.info(f"\n✓ Best model: {best_name} (val accuracy: {best_score:.3f})")
    
    return best_name, models[best_name]


def evaluate_model(model, splits, model_name=""):
    """Evaluate model on test set."""
    logger.info(f"\nEvaluating {model_name} on test set...")
    
    y_pred = model.predict(splits["test"]["X"])
    
    acc = accuracy_score(splits["test"]["y"], y_pred)
    prec = precision_score(splits["test"]["y"], y_pred, zero_division=0)
    rec = recall_score(splits["test"]["y"], y_pred, zero_division=0)
    f1 = f1_score(splits["test"]["y"], y_pred, zero_division=0)
    
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
    
    logger.info(f"  Accuracy:  {acc:.3f}")
    logger.info(f"  Precision: {prec:.3f}")
    logger.info(f"  Recall:    {rec:.3f}")
    logger.info(f"  F1 Score:  {f1:.3f}")
    
    return results


def save_model(model, model_name, scaler):
    """Save trained model and scaler."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f"{model_name}_{timestamp}.pkl"
    scaler_file = f"scaler_{timestamp}.pkl"
    
    model_path = Path(ARTIFACT_PATHS["models_dir"]) / model_file
    scaler_path = Path(ARTIFACT_PATHS["models_dir"]) / scaler_file
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"✓ Model saved: {model_path}")
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"✓ Scaler saved: {scaler_path}")
    
    return model_path, scaler_path


def main():
    """Run Phase 4 end-to-end training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("=" * 80)
    logger.info("PHASE 4: MODEL TRAINING & FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    # Load Phase 3 output
    logger.info("\n[Step 1/6] Loading sentiment-augmented dataset...")
    df = load_dataset()
    if df is None:
        logger.error("Failed to load dataset")
        return False
    
    # Engineer features
    logger.info("\n[Step 2/6] Engineering features...")
    features_df, target_series, feature_names = engineer_features(df)
    
    # Split data
    logger.info("\n[Step 3/6] Splitting data...")
    splits = split_data(features_df, target_series)
    
    # Scale features
    logger.info("\n[Step 4/6] Scaling features...")
    splits, scaler = scale_features(splits)
    
    # Train candidates
    logger.info("\n[Step 5/6] Training candidate models...")
    results, models = train_candidate_models(splits)
    
    # Select best
    logger.info("\n[Step 6/6] Selecting best model...")
    best_name, best_model = select_best_model(results, models)
    
    # Evaluate on test
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)
    test_results = evaluate_model(best_model, splits, best_name)
    
    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)
    model_path, scaler_path = save_model(best_model, best_name, scaler)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4 COMPLETE - MODEL TRAINED AND SAVED")
    logger.info("=" * 80)
    logger.info(f"Best model: {best_name}")
    logger.info(f"Test accuracy: {test_results['accuracy']:.1%}")
    logger.info(f"Model path: {model_path}")
    logger.info("\nNext steps:")
    logger.info("1. Phase 5: Validation & backtesting")
    logger.info("2. Run: python backend/ml_model/evaluate.py")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
