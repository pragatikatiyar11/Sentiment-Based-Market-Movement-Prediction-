"""
Project configuration for Sentiment-Based Market Movement Prediction.

This module centralizes all project-level decisions: prediction target, data sources,
feature engineering parameters, and API contracts. All phases depend on these definitions.

Reference: Phase 1 (Week 1) - Problem Framing and Setup
"""

# ============================================================================
# PREDICTION TARGET DEFINITION
# ============================================================================
# Binary classification: Does the stock price move UP or DOWN next trading day?
# Label = 1 if close_price[t+1] > close_price[t], else 0

PREDICTION_TARGET = "binary_direction"  # Options: "binary_direction", "buy_sell_hold"
LABEL_WINDOW_DAYS = 1  # Predict next N trading days' price movement
DIRECTION_THRESHOLD = 0.0  # Percentage change threshold (0.0 = any positive change is UP)

# ============================================================================
# STOCK UNIVERSE
# ============================================================================
# Start with a small, liquid basket for MVP, expand later

STOCK_UNIVERSE = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "TSLA",  # Tesla
    "AMZN",  # Amazon
    "NVDA",  # NVIDIA
    "META",  # Meta
    "JPM",   # JP Morgan
    "BAC",   # Bank of America
    "XOM",   # ExxonMobil
]

# ============================================================================
# DATA SOURCES
# ============================================================================
# Phase 2 will implement ingestion for each

DATA_SOURCES = {
    "market_data": {
        "provider": "yfinance",
        "data_types": ["OHLCV"],  # Open, High, Low, Close, Volume
        "history_window_days": 252 * 3,  # 3 years of training data
        "update_frequency": "daily",
    },
    "sentiment_text": {
        "sources": [
            {"name": "newsapi", "type": "news_headlines", "priority": 1},
            # {"name": "twitter_api", "type": "social_media", "priority": 2},  # Phase 2 extension
        ],
        "lookback_days": 1,  # Include news from past N days for alignment
        "update_frequency": "daily",
    },
}

# ============================================================================
# SENTIMENT FEATURE ENGINEERING
# ============================================================================
# Phase 3 will implement FinBERT extraction; these define aggregation strategy

SENTIMENT_CONFIG = {
    "model_class": "FinBERT",  # Options: "FinBERT", "DistilBERT_financial", "custom_fine_tuned"
    "aggregation_windows": [1, 5, 20],  # Rolling averages (days): today, 1 week, 1 month
    "score_normalization": "min_max",  # Options: "min_max", "z_score", "raw"
    "confidence_threshold": 0.5,  # Minimum model confidence to include in aggregation
}

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
# Phase 4 will generate features; these define the feature set

FEATURES = {
    "technical_indicators": {
        "enabled": True,
        "indicators": [
            {"name": "SMA", "windows": [5, 20, 50]},
            {"name": "RSI", "window": 14},
            {"name": "MACD", "fast": 12, "slow": 26, "signal": 9},
            {"name": "ATR", "window": 14},  # Volatility
            {"name": "volume_ratio", "window": 20},
        ],
    },
    "sentiment_features": {
        "enabled": True,
        "features": [
            "sentiment_score_today",
            "sentiment_score_5day_ma",
            "sentiment_score_20day_ma",
            "sentiment_direction_change",  # Today vs yesterday
            "confidence_weighted_sentiment",
        ],
    },
    "lag_features": {
        "enabled": True,
        "lags": [1, 5, 10],  # Include past N days as features
    },
}

# ============================================================================
# MODEL TRAINING PARAMETERS
# ============================================================================
# Phase 4 will use these for train/validation/test splits and tuning

TRAINING_CONFIG = {
    "test_split_ratio": 0.2,  # Last 20% of data for final holdout test
    "validation_split_ratio": 0.15,  # 15% for hyperparameter tuning
    "walk_forward_window": 252,  # 1 year of training data per walk-forward fold
    "walk_forward_step": 21,  # Slide by 1 month at a time
    "random_seed": 42,
}

# ============================================================================
# MODEL SELECTION
# ============================================================================
# Phase 4 will train these; pick best performer in validation

CANDIDATE_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "neural_network",
]

# ============================================================================
# API CONTRACT
# ============================================================================
# Phase 6 will implement endpoints with these exact input/output shapes

API_ENDPOINTS = {
    "predict": {
        "method": "POST",
        "path": "/predict",
        "input": {
            "symbol": "str (e.g., 'AAPL')",
            "text": "str (news/sentiment text, optional)",
            "use_cached_model": "bool (default: True)",
        },
        "output": {
            "symbol": "str",
            "timestamp": "ISO 8601",
            "sentiment_score": "float [0, 1]",
            "predicted_direction": "int (0=DOWN, 1=UP)",
            "prediction_confidence": "float [0, 1]",
            "historical_price": "float",
            "features_used": "dict (feature values for transparency)",
        },
    },
    "backtest": {
        "method": "GET",
        "path": "/backtest",
        "input": {
            "symbol": "str",
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
        },
        "output": {
            "symbol": "str",
            "total_return": "float",
            "sharpe_ratio": "float",
            "max_drawdown": "float",
            "win_rate": "float",
            "trade_count": "int",
        },
    },
}

# ============================================================================
# LOGGING AND REPRODUCIBILITY
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    "log_file": "backend/logs/app.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

ARTIFACT_PATHS = {
    "models_dir": "backend/ml_model/trained_models/",
    "datasets_dir": "backend/data/processed/",
    "logs_dir": "backend/logs/",
    "cache_dir": "backend/cache/",
}

# ============================================================================
# SUCCESS CRITERIA (Phase 7)
# ============================================================================

SUCCESS_CRITERIA = {
    "minimum_accuracy_over_baseline": 0.55,  # Beat 50% random baseline by 5%
    "acceptable_max_drawdown": 0.15,  # Max 15% drawdown in backtest
    "minimum_test_samples": 50,  # At least 50 out-of-sample predictions
    "api_response_time_ms": 2000,  # Predict in under 2 seconds
}

# ============================================================================
# PRINT SUMMARY
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SENTIMENT-BASED MARKET MOVEMENT PREDICTION - PROJECT CONFIG")
    print("=" * 80)
    print(f"\n✓ Prediction Target: {PREDICTION_TARGET} ({LABEL_WINDOW_DAYS}-day horizon)")
    print(f"✓ Stock Universe: {len(STOCK_UNIVERSE)} symbols ({', '.join(STOCK_UNIVERSE[:3])}...)")
    print(f"✓ Data Sources: {', '.join([s for s in DATA_SOURCES.keys()])}")
    print(f"✓ Sentiment Model: {SENTIMENT_CONFIG['model_class']}")
    print(f"✓ Candidate Models: {', '.join(CANDIDATE_MODELS)}")
    print(f"✓ Test Accuracy Baseline: > {SUCCESS_CRITERIA['minimum_accuracy_over_baseline']*100:.0f}%")
    print("=" * 80)
