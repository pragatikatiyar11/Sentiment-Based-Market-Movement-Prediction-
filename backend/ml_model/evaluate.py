"""
Backtesting and evaluation for Sentiment-Based Market Movement Prediction.

This module handles:
1. Walk-forward backtesting (time-series validation)
2. Trading metrics calculation (returns, drawdown, Sharpe ratio)
3. ML metrics reporting (accuracy, precision, recall)
4. Results visualization and export

Reference: Phase 5 (Week 4-5) - Validation and Backtesting
"""

import sys
import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ARTIFACT_PATHS, SUCCESS_CRITERIA

logger = logging.getLogger(__name__)


def load_trained_model(model_file=None):
    """
    Load trained model and scaler from Phase 4.
    
    Args:
        model_file (str): Model filename, default: latest in trained_models/
    
    Returns:
        tuple: (model, scaler)
    """
    models_dir = Path(ARTIFACT_PATHS["models_dir"])
    
    if model_file is None:
        # Find latest model file
        pkl_files = sorted(models_dir.glob("*.pkl"))
        if not pkl_files:
            logger.error("No trained models found")
            return None, None
        
        # Get latest model (not scaler)
        model_files = [f for f in pkl_files if "scaler" not in f.name]
        if not model_files:
            logger.error("No model files found (only scalers)")
            return None, None
        
        latest_model_file = model_files[-1]
        model_file = latest_model_file.name
    
    model_path = models_dir / model_file
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None, None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model: {model_file}")
    
    # Find corresponding scaler
    timestamp = model_file.split("_")[-1].replace(".pkl", "")
    scaler_file = f"scaler_{timestamp}.pkl"
    scaler_path = models_dir / scaler_file
    
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler: {scaler_file}")
    else:
        logger.warning(f"Scaler not found: {scaler_file}")
    
    return model, scaler


def run_walk_forward_backtest(dataset, model, scaler, window_size=252, step_size=21):
    """
    Run walk-forward backtest: train window slides through time.
    
    Phase 5 core: Time-series validation preventing lookahead bias.
    
    Args:
        dataset (pd.DataFrame): Full dataset with features and labels
        model: Trained model from Phase 4
        scaler: Feature scaler
        window_size (int): Training window (days)
        step_size (int): Slide window by N days
    
    Returns:
        pd.DataFrame: Backtest results (predictions, actuals, returns)
    """
    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD BACKTEST")
    logger.info("=" * 80)
    
    results = []
    n = len(dataset)
    
    # Phase 5: Placeholder implementation (full dataset test)
    # Phase 7: Implement true walk-forward with retraining
    
    logger.warning("Walk-forward backtesting not yet fully implemented (Phase 5)")
    logger.info("Running full-dataset test instead...")
    
    # For now, use the full dataset as test
    logger.info(f"Dataset size: {n} rows")
    
    # Generate predictions on full dataset
    # Note: This is simplified; Phase 5 production should use walk-forward folds
    
    return None


def calculate_trading_metrics(predictions, actuals, prices):
    """
    Calculate trading-oriented metrics.
    
    Phase 5: Metrics for trading strategy evaluation.
    
    Args:
        predictions (np.array): Model predictions [0, 1]
        actuals (np.array): Actual labels [0, 1]
        prices (np.array): Price series for return calculation
    
    Returns:
        dict: Trading metrics
    """
    logger.info("\nCalculating trading metrics...")
    
    # Simplified metrics (Phase 5 MVP)
    accuracy = (predictions == actuals).mean()
    
    # Calculate returns
    correct_predictions = predictions == actuals
    returns = np.where(correct_predictions, 0.01, -0.005)  # +1% for correct, -0.5% for wrong
    total_return = returns.sum() / len(returns)
    
    # Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (simplified)
    excess_returns = returns - 0.0  # Assume 0% risk-free rate
    sharpe = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": correct_predictions.mean(),
        "trade_count": len(predictions),
    }
    
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.3f}")
    
    return metrics


def calculate_ml_metrics(predictions, actuals):
    """
    Calculate ML metrics (accuracy, precision, recall, F1).
    
    Args:
        predictions (np.array): Model predictions
        actuals (np.array): Actual labels
    
    Returns:
        dict: ML metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    
    logger.info("\nCalculating ML metrics...")
    
    metrics = {
        "accuracy": accuracy_score(actuals, predictions),
        "precision": precision_score(actuals, predictions, zero_division=0),
        "recall": recall_score(actuals, predictions, zero_division=0),
        "f1": f1_score(actuals, predictions, zero_division=0),
    }
    
    tn, fp, fn, tp = confusion_matrix(actuals, predictions).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.3f}")
    
    return metrics


def validate_success_criteria(trading_metrics, ml_metrics):
    """
    Check if model meets Phase 7 success criteria.
    
    Args:
        trading_metrics (dict): From calculate_trading_metrics()
        ml_metrics (dict): From calculate_ml_metrics()
    
    Returns:
        dict: Validation results
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS CRITERIA VALIDATION")
    logger.info("=" * 80)
    
    criteria = {
        "accuracy_threshold": ml_metrics["accuracy"] > SUCCESS_CRITERIA["minimum_accuracy_over_baseline"],
        "drawdown_threshold": abs(trading_metrics["max_drawdown"]) < SUCCESS_CRITERIA["acceptable_max_drawdown"],
        "sample_size": trading_metrics["trade_count"] >= SUCCESS_CRITERIA["minimum_test_samples"],
    }
    
    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {criterion}")
    
    all_passed = all(criteria.values())
    logger.info(f"\n{'✓' if all_passed else '✗'} All criteria met: {all_passed}")
    
    return criteria


def generate_report(dataset, model, scaler):
    """
    Generate comprehensive Phase 5 evaluation report.
    
    Args:
        dataset (pd.DataFrame): Full dataset
        model: Trained model
        scaler: Feature scaler
    
    Returns:
        dict: Complete evaluation report
    """
    logger.info("=" * 80)
    logger.info("PHASE 5: VALIDATION & BACKTESTING")
    logger.info("=" * 80)
    
    # Placeholder implementation
    logger.warning("Full Phase 5 evaluation not yet implemented")
    logger.info("Phase 5 is scheduled for Week 4-5")
    logger.info("\nPhase 5 TODO:")
    logger.info("1. Implement walk-forward backtesting with retraining")
    logger.info("2. Calculate Sharpe ratio, max drawdown, win rate")
    logger.info("3. Analyze model performance by market regime")
    logger.info("4. Generate visualizations (return curves, drawdown)")
    logger.info("5. Produce final report for presentation")
    
    report = {
        "phase": 5,
        "status": "NOT_YET_IMPLEMENTED",
        "planned_metrics": [
            "accuracy", "precision", "recall", "f1",
            "sharpe_ratio", "max_drawdown", "total_return", "win_rate"
        ],
    }
    
    return report


def main():
    """Run Phase 5 evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("=" * 80)
    logger.info("PHASE 5: VALIDATION & BACKTESTING")
    logger.info("=" * 80)
    
    # Load model and scaler
    logger.info("\n[Step 1/5] Loading trained model...")
    model, scaler = load_trained_model()
    
    if model is None:
        logger.error("Could not load model")
        logger.info("Run Phase 4 first: python backend/ml_model/train.py")
        return False
    
    # Load dataset
    logger.info("\n[Step 2/5] Loading dataset...")
    dataset_path = Path(ARTIFACT_PATHS["datasets_dir"]) / "sentiment_augmented_dataset.csv"
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    dataset = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(dataset)} rows")
    
    # Run evaluation (placeholder)
    logger.info("\n[Step 3/5] Running evaluation...")
    report = generate_report(dataset, model, scaler)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5 SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Status: {report['status']}")
    logger.info(f"Planned metrics: {', '.join(report['planned_metrics'])}")
    logger.info("\nNext steps:")
    logger.info("1. Implement walk-forward backtesting")
    logger.info("2. Calculate trading metrics")
    logger.info("3. Validate success criteria")
    logger.info("4. Move to Phase 6: API hardening")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
