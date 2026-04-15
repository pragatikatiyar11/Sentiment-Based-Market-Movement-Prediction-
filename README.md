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
├── README.md                              # This file
├── requirements.txt                       # All dependencies (pinned versions)
├── PHASE_CHECKLIST.md                     # Phase tracking checklist
│
├── backend/
│   ├── app.py                             # Flask API (Phases 1, 6)
│   ├── config.py                          # Centralized project config (Phase 1)
│   ├── PROJECT_PLAN.md                    # Detailed Phase 1 decisions
│   ├── data_collector.py                  # Data ingestion pipeline (Phase 2)
│   ├── stock.py                           # Market data helpers
│   ├── predictor.py                       # Prediction logic (to upgrade in Phase 4)
│   │
│   ├── ml_model/
│   │   ├── __init__.py
│   │   ├── sentiment.py                   # Sentiment extraction (Phase 3)
│   │   ├── train.py                       # Model training (Phase 4)
│   │   ├── features.py                    # Feature engineering (Phase 4)
│   │   ├── evaluate.py                    # Backtesting (Phase 5)
│   │   └── trained_models/                # Serialized model artifacts
│   │
│   ├── data/
│   │   ├── raw/                           # Raw OHLCV + news (Phase 2)
│   │   └── processed/                     # Engineered features (Phase 4)
│   │
│   ├── logs/                              # Structured logs (Phase 6)
│   └── cache/                             # Model cache (Phase 6)
│
└── [frontend/]                            # React dashboard (Phase 7, optional)
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

### 2. Phase 1: Review Project Configuration

All core decisions are centralized in `backend/config.py`:

```bash
cd backend
python config.py  # Print configuration summary
cat PROJECT_PLAN.md  # Review Phase 1 decisions
```

**Key decisions**:
- **Prediction Target**: Binary direction (UP/DOWN) for next trading day
- **Stock Universe**: AAPL, MSFT, GOOGL, TSLA, AMZN, NVDA, META, JPM, BAC, XOM
- **Data Sources**: Yahoo Finance (OHLCV) + NewsAPI (headlines)
- **Success Criteria**: > 55% accuracy, < 15% max drawdown

### 3. Phase 2: Fetch and Prepare Data

```bash
cd backend
python data_collector.py
```

This will:
- Fetch 3 years of OHLCV for all 10 stocks
- Create binary labels (UP/DOWN)
- Generate `backend/data/processed/training_dataset.csv`
- Validate dataset integrity

### 4. Phase 3-4: (Coming Soon)

Model training and feature engineering notebooks will be provided.

### 5. Start API Server

```bash
cd backend
python app.py
```

API will run on `http://localhost:5000`

**Endpoints** (Phase 6 full implementation):
- `POST /predict` — Predict stock direction given symbol + text
- `GET /backtest` — Run backtest for symbol over date range
- `GET /` — Health check and API info

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
