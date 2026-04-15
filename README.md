# Sentiment-Based Market Movement Prediction

## 🚀 Overview

This project predicts stock market movements using sentiment analysis from news headlines combined with technical indicators. It uses a machine learning pipeline to classify next-day price direction (UP/DOWN) based on market data and text sentiment.

**Status**: Phase 1 Complete (Problem Framing & Setup) | Phases 2-7 In Progress

---

## 🧠 Features

- ✅ **Market Data Pipeline**: Fetch OHLCV data from Yahoo Finance
- ✅ **News Sentiment Integration**: Aggregated sentiment from financial news
- ✅ **Finance-Tuned Models**: Logistic Regression → LightGBM candidate models
- ✅ **API Endpoints**: RESTful prediction and backtesting interfaces
- 🔄 **Backtesting Framework**: Walk-forward validation with trading metrics
- 📊 **Flask Dashboard**: (Coming in Phase 7)

---

## 📋 Project Phases

| Phase | Timeline | Focus | Status |
|-------|----------|-------|--------|
| **Phase 1** | Week 1 | Problem framing, config, API contract | ✅ Complete |
| **Phase 2** | Week 1-2 | Data ingestion (OHLCV + news) | 🚧 In Progress |
| **Phase 3** | Week 2-3 | Finance sentiment extraction (FinBERT) | 📋 Planned |
| **Phase 4** | Week 3-4 | Feature engineering & model training | 📋 Planned |
| **Phase 5** | Week 4-5 | Validation & backtesting | 📋 Planned |
| **Phase 6** | Week 5-6 | API hardening & reliability | 📋 Planned |
| **Phase 7** | End of Week 6 | Final packaging & presentation | 📋 Planned |

See [PHASE_CHECKLIST.md](PHASE_CHECKLIST.md) for a detailed week-wise breakdown.

---

## 📂 Project Structure

```
.
├── README.md                         # This file
├── requirements.txt                  # Pinned dependencies
├── PHASE_CHECKLIST.md                # Phase tracking
│
└── backend/
    ├── config.py                     # Centralized config (Phase 1)
    ├── app.py                        # Flask API (Phase 1, 6)
    ├── PROJECT_PLAN.md               # Phase 1 detailed decisions
    ├── ARCHITECTURE.md               # Phase-to-file mapping
    ├── PHASE2_README.md              # Phase 2 guide
    ├── PHASE3_README.md              # Phase 3 guide
    ├── PHASE4_README.md              # Phase 4 guide
    │
    ├── data_collector.py             # OHLCV ingestion (Phase 2)
    ├── sentiment_collector.py        # News aggregation (Phase 2)
    ├── pipeline.py                   # Unified pipeline (Phase 2)
    │
    ├── stock.py                      # Market helpers
    ├── predictor.py                  # Prediction logic
    │
    ├── ml_model/
    │   ├── __init__.py
    │   ├── sentiment.py              # Sentiment extraction (Phase 1)
    │   ├── sentiment_extractor.py    # Finance sentiment (Phase 3)
    │   ├── train.py                  # Model training (Phase 4)
    │   ├── evaluate.py               # Backtesting (Phase 5)
    │   └── trained_models/           # Serialized models & scalers
    │
    ├── data/
    │   ├── raw/                      # Raw OHLCV + news
    │   └── processed/                # Engineered features
    │
    ├── logs/                         # Structured logs
    └── cache/                        # Model cache
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| **API** | Flask, Werkzeug |
| **ML** | scikit-learn, XGBoost, LightGBM |
| **NLP/Sentiment** | Transformers (FinBERT), PyTorch |
| **Data** | Pandas, NumPy, yfinance |
| **Features** | TA (technical analysis) |
| **Backtesting** | backtrader |

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone <repo>
cd Sentiment-Based-Market-Movement-Prediction-
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Review Phase 1 Config

All project decisions are in `backend/config.py`:

```bash
cd backend
python config.py  # Print configuration summary
cat PROJECT_PLAN.md  # Review detailed Phase 1 decisions
```

**Key decisions**:
- Prediction target: Binary next-day direction (UP/DOWN)
- Stock universe: 10 large-cap symbols (AAPL, MSFT, GOOGL, TSLA, AMZN, NVDA, META, JPM, BAC, XOM)
- Success criteria: > 55% accuracy, < 15% max drawdown

### 3. Run Full Pipeline (Phase 2-5)

**Option A: Sequential Steps**
```bash
# Phase 2: Collect market data + sentiment texts → unified dataset
python pipeline.py

# Phase 3: Extract sentiment features from headlines
python ml_model/sentiment_extractor.py

# Phase 4: Train prediction model
python ml_model/train.py

# Phase 5: Backtest and evaluate
python ml_model/evaluate.py
```

**Option B: Individual Phases**
```bash
# Phase 2 only
python data_collector.py           # Market OHLCV + labels
python sentiment_collector.py      # News headlines
python pipeline.py                 # Join everything

# Phase 3 only
python ml_model/sentiment_extractor.py

# Phase 4 only
python ml_model/train.py

# Phase 5 only
python ml_model/evaluate.py
```

### 4. Output Files

After each phase, check:

```
backend/data/processed/
├── training_dataset.csv                   # Phase 2
├── sentiment_texts.csv                    # Phase 2
├── unified_training_dataset.csv           # Phase 2
└── sentiment_augmented_dataset.csv        # Phase 3

backend/ml_model/trained_models/
├── random_forest_20260416_123456.pkl      # Phase 4
└── scaler_20260416_123456.pkl             # Phase 4
```

### 5. Start API Server (Phase 6)

```bash
python app.py
```

Server runs on `http://localhost:5000`

**Test endpoints**:
```bash
# Predict stock direction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "text": "Apple reports strong earnings"}'

# Backtest a symbol
curl "http://localhost:5000/backtest?symbol=AAPL&start_date=2024-01-01"
```

### 6. Documentation

- [PHASE_CHECKLIST.md](PHASE_CHECKLIST.md) — Phase-wise task tracking
- [backend/ARCHITECTURE.md](backend/ARCHITECTURE.md) — Phase-to-file mapping
- [backend/PROJECT_PLAN.md](backend/PROJECT_PLAN.md) — Phase 1 detailed plan
- [backend/PHASE2_README.md](backend/PHASE2_README.md) — Phase 2 guide
- [backend/PHASE3_README.md](backend/PHASE3_README.md) — Phase 3 guide
- [backend/PHASE4_README.md](backend/PHASE4_README.md) — Phase 4 guide

---

## 📊 Expected Results (Phase 7)

- **Model Accuracy**: > 55% on holdout test (binary direction)
- **Sharpe Ratio**: ≥ 1.0 in walk-forward backtest
- **Max Drawdown**: ≤ 15%
- **API Response Time**: < 2 seconds per prediction

---

## 🔄 Development Workflow

1. **Pull latest**: `git pull origin main`
2. **Review current phase**: Check `PHASE_CHECKLIST.md`
3. **Run phase tests**: Each phase has validation checks
4. **Commit & document**: Update checklist and phase READMEs as you progress

---

## 📚 Documentation

- **[PHASE_CHECKLIST.md](PHASE_CHECKLIST.md)** — High-level phase tracker
- **[backend/PROJECT_PLAN.md](backend/PROJECT_PLAN.md)** — Detailed Phase 1 decisions
- **[backend/config.py](backend/config.py)** — Centralized configuration with inline docs

---

## ⚠️ Known Limitations

- **Data**: Currently uses free tier APIs (yfinance, NewsAPI); production should use paid providers
- **Sentiment**: Generic transformers model until Phase 3 FinBERT integration
- **Prediction**: Hardcoded rules until Phase 4 model training
- **Frontend**: Dashboard coming in Phase 7

---

## 👥 Contributors

- **Original Author**: Pragati Katiyar
- **Phase 1 Architect**: Development Team (April 2026)

---

## 📝 License

(Specify your license here)

---

## 🤝 How to Contribute

1. Check [PHASE_CHECKLIST.md](PHASE_CHECKLIST.md) for current phase tasks
2. Pick a task, mark as in-progress
3. Follow the phase README and config.py guidelines
4. Submit PR with phase validation checks passing
5. Update checklist when complete

---

**Next**: Start Phase 2 by running `python backend/data_collector.py`
