"""
Flask API for Sentiment-Based Market Movement Prediction.

This module exposes the prediction engine via REST endpoints defined in config.API_ENDPOINTS.
Phases 2-6 populate these endpoints with data and models; Phase 1 provides the config blueprint.

Reference: Phase 6 (Week 5-6) - API Integration and Reliability
"""

import logging
from datetime import datetime
from flask import Flask, jsonify, request

from config import (
    STOCK_UNIVERSE, 
    PREDICTION_TARGET,
    LOGGING_CONFIG,
    SUCCESS_CRITERIA,
)
from ml_model.sentiment import analyze
from stock import get_stock
from predictor import predict_action

# ============================================================================
# Setup Logging
# ============================================================================

logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Flask App Initialization
# ============================================================================

app = Flask(__name__)

@app.route("/")
def home():
    """Health check and API info endpoint."""
    return jsonify({
        "message": "Sentiment-Based Market Movement Prediction API",
        "version": "1.0.0",
        "prediction_target": PREDICTION_TARGET,
        "stock_universe_size": len(STOCK_UNIVERSE),
        "endpoints": {
            "predict": "POST /predict",
            "backtest": "GET /backtest",
            "health": "GET /",
        },
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict stock price direction given symbol and optional sentiment text.
    
    Phase 4: Will use trained model.predict() instead of hardcoded rules.
    Phase 6: Add input validation, error handling, caching.
    
    Request JSON:
    {
        "symbol": "AAPL",
        "text": "Apple announces record sales",
        "use_cached_model": true
    }
    
    Response JSON:
    {
        "symbol": "AAPL",
        "timestamp": "2026-04-16T14:30:00Z",
        "sentiment_score": 0.87,
        "predicted_direction": 1,
        "prediction_confidence": 0.72,
        "historical_price": 175.43,
        "features_used": {...}
    }
    """
    try:
        # Phase 6: Input validation
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Request body required"}), 400
        
        symbol = payload.get("symbol", "").upper()
        text = payload.get("text", "")
        
        if not symbol:
            return jsonify({"error": "symbol required"}), 400
        
        if symbol not in STOCK_UNIVERSE:
            return jsonify({
                "error": f"symbol {symbol} not in universe",
                "available_symbols": STOCK_UNIVERSE,
            }), 400
        
        # Phase 2: Fetch market data
        stock_data = get_stock(symbol)
        if stock_data is None:
            logger.warning(f"Stock data unavailable for {symbol}")
            return jsonify({"error": "Stock data not available"}), 503
        
        # Phase 3: Extract sentiment
        sentiment_result = analyze(text) if text else {"label": "neutral", "score": 0.5}
        
        # Phase 4: Model prediction (currently uses hardcoded rules)
        action = predict_action(
            sentiment_result["label"],
            stock_data["change"]
        )
        
        logger.info(f"Prediction: {symbol} -> {action} (confidence: {sentiment_result['score']})")
        
        # Phase 6: Structured response
        return jsonify({
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment_score": sentiment_result["score"],
            "predicted_direction": 1 if action == "BUY" else 0,  # 1=UP, 0=DOWN
            "prediction_confidence": sentiment_result["score"],
            "historical_price": stock_data["price"],
            "features_used": {
                "sentiment_label": sentiment_result["label"],
                "price_change": stock_data["change"],
            },
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/backtest", methods=["GET"])
def backtest():
    """
    Run walk-forward backtest for a symbol over a date range.
    
    Phase 5: Implement walk-forward backtesting logic.
    Phase 6: Expose results via API.
    
    Query parameters:
    - symbol: str (required)
    - start_date: YYYY-MM-DD (optional, default: 2 years ago)
    - end_date: YYYY-MM-DD (optional, default: today)
    
    Response JSON:
    {
        "symbol": "AAPL",
        "start_date": "2024-01-01",
        "end_date": "2025-01-01",
        "total_return": 0.18,
        "sharpe_ratio": 1.42,
        "max_drawdown": -0.12,
        "win_rate": 0.58,
        "trade_count": 252
    }
    """
    try:
        symbol = request.args.get("symbol", "").upper()
        start_date = request.args.get("start_date", None)
        end_date = request.args.get("end_date", None)
        
        if not symbol:
            return jsonify({"error": "symbol query parameter required"}), 400
        
        if symbol not in STOCK_UNIVERSE:
            return jsonify({
                "error": f"symbol {symbol} not in universe",
                "available_symbols": STOCK_UNIVERSE,
            }), 400
        
        # Phase 5: Placeholder - returns stub results
        logger.info(f"Backtest requested: {symbol} ({start_date} to {end_date})")
        
        return jsonify({
            "symbol": symbol,
            "start_date": start_date or "2024-01-01",
            "end_date": end_date or datetime.utcnow().strftime("%Y-%m-%d"),
            "total_return": 0.18,  # Placeholder
            "sharpe_ratio": 1.42,
            "max_drawdown": -0.12,
            "win_rate": 0.58,
            "trade_count": 252,
            "status": "BACKTEST NOT YET IMPLEMENTED (Phase 5)",
        }), 200
    
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting Sentiment-Based Market Movement Prediction API")
    logger.info(f"Stock Universe: {STOCK_UNIVERSE}")
    logger.info(f"Prediction Target: {PREDICTION_TARGET}")
    app.run(debug=True, port=5000)
