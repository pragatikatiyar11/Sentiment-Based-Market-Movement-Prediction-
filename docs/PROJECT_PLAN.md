# Project Plan: Sentiment-Based Market Movement Prediction

**Status**: Phase 1 (Week 1) - Problem Framing and Setup  
**Last Updated**: April 16, 2026  
**Team**: Development Team

---

## 1. Executive Summary

This document captures all Phase 1 decisions that unblock downstream phases. It defines:
- **What we predict**: Binary next-day stock price direction (UP/DOWN)
- **What we use**: Market OHLCV data + news sentiment + technical indicators
- **Which stocks**: A curated basket of 10 large-cap, liquid symbols
- **Success bar**: Beat 50% baseline with <15% max drawdown in backtest

All downstream phases reference decisions captured here.

---

## 2. Prediction Target Definition

### Decision: Binary Direction Classification

**Target**: Does the stock close higher tomorrow than today?
- **Label = 1** if `close[t+1] > close[t]` (price goes UP)
- **Label = 0** if `close[t+1] <= close[t]` (price goes DOWN)
- **Horizon**: Predict for next 1 trading day
- **Threshold**: Any positive change counts as UP (no minimum % change required initially)

**Rationale**:
- Binary is simpler than 3-class (BUY/SELL/HOLD) and converges faster for MVP
- 1-day horizon is tractable: achieves prediction within news/sentiment relevance window
- Upgrading to ternary or multi-day later is non-breaking if this foundation succeeds

---

## 3. Stock Universe

### Decision: 10 Large-Cap Liquid Stocks

```
Primary Universe (Phase 2-5):
  AAPL, MSFT, GOOGL, TSLA, AMZN (Tech)
  NVDA, META (Tech/Mega)
  JPM, BAC (Finance)
  XOM (Energy)
```

**Rationale**:
- High liquidity → reliable OHLCV data and news coverage
- Sector diversity → test if model generalizes across industries
- MVP scope → 10 symbols is tractable for manual validation
- Expansion path clear → add 20+ symbols in Phase 7 if needed

---

## 4. Data Sources & Ingestion Strategy

### Market Data

**Source**: Yahoo Finance (`yfinance` library)
- **Data types**: Open, High, Low, Close, Volume (OHLCV)
- **History window**: 3 years (252 trading days × 3)
- **Update frequency**: Daily (Phase 2)
- **Reliability**: Yahoo Finance is stable for MVP; production might use paid alternatives (Bloomberg, Refinitiv)

### Sentiment Text

**Primary Source** (Phase 2): NewsAPI
- **Data**: English news headlines about our stock symbols
- **Lookback window**: Past 1 day (align with market close)
- **Update frequency**: Daily

**Secondary Source** (Phase 2+ extension): Twitter API (if available)
- Only added if Phase 2 proves newsapi sufficient; don't block MVP on this

---

## 5. Feature Engineering Parameters

### Technical Indicators (Phase 4)

- **SMA** (Simple Moving Average): 5, 20, 50-day periods
- **RSI** (Relative Strength Index): 14-day
- **MACD** (Moving Average Convergence/Divergence): 12/26/9
- **ATR** (Average True Range): 14-day (volatility)
- **Volume Ratio**: 20-day MA of volume / long-term MA

### Sentiment Features (Phase 3)

- **Sentiment Score**: FinBERT model output [0, 1] for each headline
- **Rolling Averages**: 1-day, 5-day, 20-day MA of sentiment
- **Sentiment Direction Change**: Score today vs. yesterday
- **Confidence Weighting**: Only include model predictions with confidence > 0.5

### Lag Features (Phase 4)

- Use past 1, 5, 10-day values as predictors (capture momentum)

---

## 6. Model Training & Validation Strategy

### Train/Test Split (Phase 4-5)

- **Historical window**: 3 years of data per stock
- **Test set**: Last 20% (roughly 150 trading days)
- **Validation set**: 15% for hyperparameter tuning
- **Walk-forward backtesting**: 1-year training window, slide by 1 month

### Candidate Models (Phase 4)

1. Logistic Regression (baseline)
2. Random Forest (tree ensemble)
3. XGBoost (gradient boosting)
4. LightGBM (fast gradient boosting)
5. Simple Neural Network (if time permits)

**Selection criterion**: Best accuracy on validation set; tie-break on inference speed.

---

## 7. API Contract (Phase 6)

### Endpoint: `/predict` (POST)

**Request**:
```json
{
  "symbol": "AAPL",
  "text": "Apple announces new iPhone sales records",
  "use_cached_model": true
}
```

**Response**:
```json
{
  "symbol": "AAPL",
  "timestamp": "2026-04-16T14:30:00Z",
  "sentiment_score": 0.87,
  "predicted_direction": 1,
  "prediction_confidence": 0.72,
  "historical_price": 175.43,
  "features_used": {
    "sma_5": 175.1,
    "rsi_14": 62.5,
    "sentiment_5day_ma": 0.65
  }
}
```

### Endpoint: `/backtest` (GET)

**Query**:
```
GET /backtest?symbol=AAPL&start_date=2024-01-01&end_date=2025-01-01
```

**Response**:
```json
{
  "symbol": "AAPL",
  "total_return": 0.18,
  "sharpe_ratio": 1.42,
  "max_drawdown": -0.12,
  "win_rate": 0.58,
  "trade_count": 252
}
```

---

## 8. Success Criteria (Acceptance for Phase 7)

### Model Quality

- **Accuracy**: > 55% on holdout test (beat 50% random by 5%)
- **Precision/Recall**: Balanced; minimize false positives (avoid whipsaw trades)
- **Max Drawdown**: ≤ 15% in walk-forward backtest

### API Quality

- **Response Time**: < 2 seconds per prediction
- **Uptime**: 99% availability (during trading hours)
- **Input Validation**: Reject invalid symbols gracefully

### Dataset Quality

- **Completeness**: 100% coverage for stock/date/features (no NaN in train set)
- **Alignment**: News timestamps aligned within ±1 trading day of market close
- **Sample Size**: ≥ 50 out-of-sample predictions per stock

---

## 9. Project File Structure (Updated)

```
backend/
├── app.py                    # Flask API (Phase 6)
├── config.py                 # ← NEW: Centralized config (Phase 1)
├── predictor.py              # Prediction engine (Phase 4)
├── stock.py                  # Market data (Phase 2)
├── project_plan.md           # ← NEW: This file
├── data/
│   ├── raw/                  # Raw OHLCV + news (Phase 2)
│   └── processed/            # Engineered features (Phase 4)
├── ml_model/
│   ├── __init__.py
│   ├── sentiment.py          # Sentiment extraction (Phase 3)
│   ├── train.py              # ← NEW: Model training (Phase 4)
│   ├── features.py           # ← NEW: Feature engineering (Phase 4)
│   ├── evaluate.py           # ← NEW: Backtesting (Phase 5)
│   └── trained_models/       # Serialized models
├── logs/
└── cache/

requirements.txt              # ← UPDATED: All dependencies (Phase 1)
PHASE_CHECKLIST.md           # ← NEW: Phase tracking (Phase 1)
```

---

## 10. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| **News API rate limits** | Medium | Implement local caching; fallback to dummy news if API unavailable |
| **Low signal-to-noise ratio** | High | Validate hypothesis early (Phase 2): do UP-day announcements have +sentiment? |
| **Data leakage** (future data in features) | Medium | Strict temporal validation: features at time t cannot use data from t+1 |
| **Model overfitting** | High | Walk-forward validation; test on 2024+ data not seen during training |
| **yfinance discontinuation** | Low | Document fallback to pandas_datareader or Alpha Vantage API |

---

## 11. Phase 1 Deliverables Checklist

- [x] Prediction target finalized: binary direction, 1-day horizon
- [x] Stock universe chosen: 10 large-cap symbols
- [x] Data sources confirmed: yfinance + NewsAPI
- [x] Feature strategy documented: technical + sentiment + lag
- [x] API contract defined: `/predict` and `/backtest` signatures
- [x] Model candidates listed: logistic regression → LightGBM
- [x] Success criteria pinned: 55% accuracy, 15% max drawdown
- [x] Project config file created: `backend/config.py`
- [x] Requirements updated with all dependencies
- [x] File structure planned for all 7 phases

---

## 12. Next Steps (Phase 2 - Week 1-2)

1. **Implement data ingestion**:
   - `backend/data_collector.py`: Fetch OHLCV from yfinance
   - `backend/news_fetcher.py`: Fetch headlines from NewsAPI
   - Validate dataset integrity (schema, nulls, duplicates)

2. **Create training dataset**:
   - Join market data + news by timestamp
   - Label each day as UP/DOWN
   - Export as CSV/Parquet for Phase 3

3. **Verify data quality**:
   - Check class balance (% UP vs % DOWN)
   - Confirm no time leakage in labels
   - Sample 10 rows and validate by hand

---

**Document Owner**: Development Team  
**Approval**: Ready for Phase 2 - Data Ingestion & Alignment
