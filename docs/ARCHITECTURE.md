# Architecture & Phase Reference Guide

## Project Phases Overview

This document maps each phase to its deliverables, files, and execution flow.

---

## Phase-to-File Mapping

| Phase | Timeline | Purpose | Key Files | Input | Output |
|-------|----------|---------|-----------|-------|--------|
| **Phase 1** | Week 1 | Problem framing | `config.py`, `PROJECT_PLAN.md` | — | Config decisions |
| **Phase 2** | Week 1-2 | Data ingestion | `data_collector.py`, `sentiment_collector.py`, `pipeline.py` | — | `unified_training_dataset.csv` |
| **Phase 3** | Week 2-3 | Sentiment extraction | `ml_model/sentiment_extractor.py` | Phase 2 output | `sentiment_augmented_dataset.csv` |
| **Phase 4** | Week 3-4 | Model training | `ml_model/train.py` | Phase 3 output | Trained model + scaler |
| **Phase 5** | Week 4-5 | Backtesting | `ml_model/evaluate.py` | Phase 4 output | Evaluation report |
| **Phase 6** | Week 5-6 | API hardening | `app.py` (updated) | Phase 4 model | Production API |
| **Phase 7** | Week 6+ | Finalization | README, docs, summary | All phases | Presentation package |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Config & Planning                                  │
│ config.py, PROJECT_PLAN.md                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Data Collection                                    │
│ data_collector.py → OHLCV + labels                          │
│ sentiment_collector.py → Headlines                          │
│ pipeline.py → Unified dataset                              │
│ OUTPUT: unified_training_dataset.csv                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Sentiment Features                                 │
│ ml_model/sentiment_extractor.py                             │
│ • Load FinBERT / sentiment model                            │
│ • Extract sentiment scores from headlines                   │
│ • Create rolling aggregates                                │
│ OUTPUT: sentiment_augmented_dataset.csv                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: Model Training                                     │
│ ml_model/train.py                                           │
│ • Engineer features (market + technical + sentiment)        │
│ • Split train/val/test (time-aware)                        │
│ • Train candidate models (LR, RF, XGBoost, LightGBM)       │
│ • Select best on validation                                 │
│ OUTPUT: trained_models/model_*.pkl, scaler_*.pkl           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: Validation & Backtesting                           │
│ ml_model/evaluate.py                                        │
│ • Walk-forward backtest                                     │
│ • Calculate returns, Sharpe, drawdown                       │
│ • Report ML + trading metrics                              │
│ • Validate success criteria                                 │
│ OUTPUT: Evaluation report + recommendations                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: API Deployment                                     │
│ app.py (updated)                                            │
│ • Load trained model + scaler                              │
│ • Serve /predict and /backtest endpoints                   │
│ • Add validation, caching, logging                         │
│ OUTPUT: Production API on localhost:5000                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 7: Finalization                                       │
│ • Freeze model artifacts                                   │
│ • Prepare reproducible demo                                │
│ • Write final summary + limitations                        │
│ OUTPUT: Presentation package                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Running the Full Pipeline

### Sequential Execution

```bash
# Phase 1: Review config
cd backend
python config.py
cat PROJECT_PLAN.md

# Phase 2: Collect data
python pipeline.py
# Output: data/processed/unified_training_dataset.csv

# Phase 3: Extract sentiment features
python ml_model/sentiment_extractor.py
# Output: data/processed/sentiment_augmented_dataset.csv

# Phase 4: Train model
python ml_model/train.py
# Output: ml_model/trained_models/model_*.pkl

# Phase 5: Backtest & evaluate
python ml_model/evaluate.py
# Output: Evaluation report

# Phase 6: Start API server
python app.py
# Runs on http://localhost:5000

# Test API
# curl -X POST http://localhost:5000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"symbol": "AAPL", "text": "Apple reports strong earnings"}'
```

---

## Feature Engineering Stages

### Phase 2: Market Foundation
- OHLCV: Open, High, Low, Close, Volume
- Labels: Binary direction (UP/DOWN)
- Base: Raw market data + aggregated headlines

### Phase 3: Sentiment Layer
- Sentiment score: [0, 1] from headlines
- Rolling sentiment: 1-day, 5-day, 20-day MA
- Confidence: Model confidence in sentiment prediction

### Phase 4: Full Feature Set
- **Market**: close, volume
- **Technical**: SMA(5,20), volatility(5,20), price_change
- **Sentiment**: score, MA(1,5,20), headline_count
- **Momentum**: price_lag(1,5)
- **Total**: 18 features

---

## Model Candidates

### Phase 4 MVP
- Logistic Regression (baseline, fast)
- Random Forest (ensemble, interpretable)

### Phase 4+ Extension (Optional)
- XGBoost (gradient boosting, fast, accurate)
- LightGBM (ultra-fast gradient boosting)
- Neural Network (if time permits)

**Selection**: Best validation accuracy. Tie-break on inference speed.

---

## Key Configurations

All settings centralized in `backend/config.py`:

```python
# Prediction target
PREDICTION_TARGET = "binary_direction"  # UP/DOWN next day

# Stock universe
STOCK_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "JPM", "BAC", "XOM"]

# Data sources
DATA_SOURCES = {
    "market_data": {"provider": "yfinance", "history_window_days": 252*3},
    "sentiment_text": {"sources": [{"name": "newsapi"}], "lookback_days": 1},
}

# Sentiment model
SENTIMENT_CONFIG = {
    "model_class": "FinBERT",
    "aggregation_windows": [1, 5, 20],
}

# Training split
TRAINING_CONFIG = {
    "test_split_ratio": 0.2,
    "validation_split_ratio": 0.15,
}

# Success criteria
SUCCESS_CRITERIA = {
    "minimum_accuracy_over_baseline": 0.55,  # > 55%
    "acceptable_max_drawdown": 0.15,         # < 15%
}
```

---

## Expected Results by Phase

| Phase | Deliverable | Target | Status |
|-------|-------------|--------|--------|
| 1 | Config document | ✓ Decisions locked | ✅ Complete |
| 2 | Unified dataset | 7500+ rows, 10 stocks | 🚧 In Progress |
| 3 | Sentiment features | Scores + rolling MA | 📋 Planned |
| 4 | Trained model | Accuracy > 50% | 📋 Planned |
| 5 | Backtest report | Sharpe > 1.0, DD < 15% | 📋 Planned |
| 6 | API endpoints | < 2s response time | 📋 Planned |
| 7 | Presentation | Docs + demo + results | 📋 Planned |

---

## File Structure After All Phases

```
backend/
├── app.py                          # API (Phase 6 updated)
├── config.py                       # Config (Phase 1)
├── PROJECT_PLAN.md                 # Phase 1 decisions
├── PHASE2_README.md                # Phase 2 guide
├── PHASE3_README.md                # Phase 3 guide
├── PHASE4_README.md                # Phase 4 guide
├── ARCHITECTURE.md                 # This file
│
├── data_collector.py               # Phase 2: Market data
├── sentiment_collector.py          # Phase 2: News headlines
├── pipeline.py                     # Phase 2: Unified pipeline
│
├── data/
│   └── processed/
│       ├── training_dataset.csv                    # Phase 2
│       ├── sentiment_texts.csv                     # Phase 2
│       ├── unified_training_dataset.csv            # Phase 2
│       └── sentiment_augmented_dataset.csv         # Phase 3
│
├── ml_model/
│   ├── __init__.py
│   ├── sentiment.py                # Original (Phase 1)
│   ├── sentiment_extractor.py      # Phase 3
│   ├── train.py                    # Phase 4
│   ├── evaluate.py                 # Phase 5
│   ├── trained_models/
│   │   ├── random_forest_20260416_123456.pkl
│   │   └── scaler_20260416_123456.pkl
│   └── features.py                 # Phase 4 (optional)
│
├── logs/                           # Phase 6 logs
├── cache/                          # Phase 6 cache
├── stock.py                        # Original (Phase 1)
└── predictor.py                    # Original (Phase 1, upgraded Phase 4)

PHASE_CHECKLIST.md                  # Tracking
```

---

## Command Reference

```bash
# Setup
pip install -r requirements.txt

# Phase 1: Review config
python backend/config.py
cat backend/PROJECT_PLAN.md

# Phase 2: Data pipeline
python backend/pipeline.py

# Phase 3: Sentiment extraction
python backend/ml_model/sentiment_extractor.py

# Phase 4: Train model
python backend/ml_model/train.py

# Phase 5: Backtest
python backend/ml_model/evaluate.py

# Phase 6: API server
python backend/app.py

# Phase 7: (Documentation & presentation assembly)
```

---

## Success Metrics (Phase 7)

- ✓ Binary direction prediction > 55% accuracy
- ✓ Backtest Sharpe ratio > 1.0
- ✓ Max drawdown < 15%
- ✓ API response time < 2 seconds
- ✓ Reproducible end-to-end pipeline
- ✓ Clear documentation and limitations

---

## Known Limitations & Future Work

### Phase 2-3
- Mock news data (real NewsAPI requires key setup)
- Single-day sentiment window (could expand to multi-day)
- No cross-stock sentiment correlation

### Phase 4-5
- Linear models only (MVP); XGBoost/LightGBM deferred
- No hyperparameter tuning (fixed seeds)
- No regime switching or market condition analysis

### Phase 6-7
- No persistent caching (models reload each request)
- No auth/rate limiting (unsafe for public API)
- No model monitoring/retraining automation
- Frontend dashboard deferred to future

---

**Last Updated**: April 16, 2026  
**Phase Status**: Phase 2 In Progress → Phase 3-7 Planned
