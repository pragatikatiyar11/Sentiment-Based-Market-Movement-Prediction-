# Startup Guide: Sentiment-Based Market Movement Prediction

**Status**: Production Ready (Phases 1-5 Infrastructure Complete)  
**Last Updated**: April 16, 2026  
**Estimated Setup Time**: 15-30 minutes

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.10+** installed
- **Git** (for version control)
- **pip** (Python package manager)
- **~5GB disk space** (for dependencies + data)
- **Internet connection** (for API calls and model downloads)

### Verify Installation

```powershell
python --version           # Should be 3.10+
pip --version              # Should be 24+
git --version              # Should be 2.30+
```

---

## 🚀 Step 1: Clone & Setup Environment

### 1.1 Clone the Repository

```powershell
git clone <your-repo-url>
cd Sentiment-Based-Market-Movement-Prediction-
```

### 1.2 Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.3 Upgrade pip (Optional but Recommended)

```powershell
python -m pip install --upgrade pip
```

---

## 📦 Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs all required packages:
- **torch** (2.6.0) — Deep learning
- **transformers** (4.40+) — NLP models
- **yfinance** (0.2.38) — Market data
- **scikit-learn** (1.5.2) — ML algorithms
- **Flask** (2.3.3) — API framework
- Plus 35+ other packages

**Estimated installation time:** 10-15 minutes on first run  
**Expected output:** `Successfully installed [X] packages`

### Troubleshooting Installation

If you get errors:

```powershell
# Clear pip cache and retry
pip cache purge
pip install -r requirements.txt

# Or install with no cache
pip install --no-cache-dir -r requirements.txt
```

If still failing, check [docs/README.md](README.md#-troubleshooting) for common issues.

---

## ✅ Step 3: Verify Installation

```powershell
python -c "import torch, transformers, pandas, sklearn; print('All imports OK!')"
```

Expected output: `All imports OK!`

---

## 🎯 Step 4: Review Configuration

All project decisions are centralized in one file:

```powershell
cd backend
python config.py
```

This prints:
- Stock universe (10 large-cap symbols)
- Prediction target (binary UP/DOWN next day)
- Data sources (yfinance + NewsAPI)
- Success criteria (>55% accuracy, <15% max drawdown)

**Key Settings to Know:**
```python
STOCK_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "JPM", "BAC", "XOM"]
PREDICTION_TARGET = "binary_direction"  # 1 = UP, 0 = DOWN
LABEL_WINDOW_DAYS = 1                   # Predict next 1 trading day
SUCCESS_CRITERIA = {
    "accuracy": 0.55,                   # >55% accuracy
    "max_drawdown": 0.15,               # <15% drawdown
}
```

---

## 🔄 Step 5: Run Full Pipeline (Phase 2-5)

### Option A: Sequential Full Run (Recommended)

```powershell
# Phase 2: Collect market data + news headlines
python pipeline.py
# Output: backend/data/processed/unified_training_dataset.csv (~50MB)

# Phase 3: Extract sentiment features
python ml_model/sentiment_extractor.py
# Output: backend/data/processed/sentiment_augmented_dataset.csv

# Phase 4: Train prediction model
python ml_model/train.py
# Output: backend/ml_model/trained_models/model_*.pkl

# Phase 5: Backtest & evaluate
python ml_model/evaluate.py
# Output: Performance report (accuracy, Sharpe ratio, drawdown)
```

**Total time:** 20-30 minutes (mostly downloading models on first run)

### Option B: Run Individual Phases

```powershell
# Phase 2 only (data collection)
python data_collector.py         # Market OHLCV
python sentiment_collector.py    # News headlines
python pipeline.py               # Join everything

# Phase 3 only (sentiment extraction)
python ml_model/sentiment_extractor.py

# Phase 4 only (model training)
python ml_model/train.py

# Phase 5 only (backtesting)
python ml_model/evaluate.py
```

### Phase Outputs

After each phase, check for output files:

```
backend/data/processed/
├── training_dataset.csv                 ← Phase 2
├── sentiment_texts.csv                  ← Phase 2
├── unified_training_dataset.csv         ← Phase 2 (final)
└── sentiment_augmented_dataset.csv      ← Phase 3

backend/ml_model/trained_models/
├── random_forest_20260416_*.pkl         ← Phase 4
├── scaler_20260416_*.pkl                ← Phase 4
└── model_metadata_*.json                ← Phase 4
```

---

## 🌐 Step 6: Start API Server (Phase 6)

```powershell
python app.py
```

Expected output:
```
 * Running on http://127.0.0.1:5000
 * WARNING: This is a development server. Not for production.
```

### Test the API

**In another terminal:**
```powershell
# Test /predict endpoint
curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"symbol": "AAPL", "text": "Apple reports strong earnings"}'
```

Expected response:
```json
{
  "symbol": "AAPL",
  "timestamp": "2026-04-16T14:30:00Z",
  "sentiment_score": 0.87,
  "predicted_direction": 1,
  "prediction_confidence": 0.72,
  "features_used": {"sma_5": 175.1, "rsi_14": 62.5}
}
```

---

## 📊 Step 7: Expected Results

By end of Phase 5, you should see:

```
✅ Phase 2: Dataset created with 7500+ rows (10 stocks × 3 years)
✅ Phase 3: Sentiment features extracted (scores + rolling MA)
✅ Phase 4: Model trained (test accuracy: 52-58%)
✅ Phase 5: Backtest results:
   - Accuracy: >55% (vs 50% random baseline)
   - Sharpe Ratio: 0.8-1.2
   - Max Drawdown: 10-15%
   - Win Rate: 52-56%
```

---

## 🛠️ Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Solution:** Activate virtual environment and reinstall:
```powershell
pip install --no-cache-dir torch==2.6.0
```

### Issue: `yfinance API Error: No data returned`
**Solution:** Yahoo Finance API may be temporarily down. Retry:
```powershell
python pipeline.py  # Will retry automatically
```

### Issue: `NEWSAPI_KEY not found`
**Solution:** Using mock data by default (safe for MVP). To use real API:
```powershell
$env:NEWSAPI_KEY = "your_newsapi_key"
```

### Issue: Out of memory during phase 3
**Solution:** Reduce dataset size in `backend/config.py`:
```python
STOCK_UNIVERSE = ["AAPL", "MSFT", "GOOGL"]  # Use 3 stocks instead of 10
```

### Issue: Model file not found when running phase 5
**Solution:** Run phase 4 first:
```powershell
python ml_model/train.py  # Generates model files
python ml_model/evaluate.py
```

---

## 📖 Documentation Reference

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview & quick start |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | Phase 1 detailed decisions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Phase-to-file mapping & flow |
| [PHASE2_README.md](PHASE2_README.md) | Data collection guide |
| [PHASE3_README.md](PHASE3_README.md) | Sentiment extraction guide |
| [PHASE4_README.md](PHASE4_README.md) | Model training guide |
| [PHASE_CHECKLIST.md](PHASE_CHECKLIST.md) | Progress tracking |

---

## 🔑 Key Files & Folders

```
backend/
├── config.py                           # All settings (read this first!)
├── app.py                              # Flask API server
├── pipeline.py                         # Phase 2 orchestration
├── data_collector.py                   # OHLCV data fetching
├── sentiment_collector.py              # News aggregation
│
├── ml_model/
│   ├── sentiment_extractor.py          # Phase 3: Sentiment scoring
│   ├── train.py                        # Phase 4: Model training
│   ├── evaluate.py                     # Phase 5: Backtesting
│   └── trained_models/                 # Serialized models (output)
│
├── data/
│   └── processed/                      # CSV datasets (output)
├── logs/                               # Log files (output)
└── cache/                              # Cache files (output)

docs/                                   # Documentation (this folder!)
requirements.txt                        # Dependencies
README.md                               # Main project docs
```

---

## 💡 Next Steps After Setup

### After Phase 2 (Data Collection)
- Check `backend/data/processed/unified_training_dataset.csv`
- Verify all 10 stocks have 3 years of data
- Inspect sample rows: are prices reasonable? Are headlines present?

### After Phase 3 (Sentiment)
- Check sentiment scores: are they in [0, 1] range?
- Verify rolling aggregates: no NaN values?
- Sample rows should show sentiment MA values

### After Phase 4 (Training)
- Check model accuracy on test set (>50% is good, >55% is excellent)
- Review feature importance (which features matter most?)
- Save model metadata for reproducibility

### After Phase 5 (Backtesting)
- Review Sharpe ratio (>1.0 is good)
- Check max drawdown (<15% is acceptable)
- Validate success criteria: accuracy, returns, stability

### After Phase 6 (API)
- Test `/predict` endpoint with different stocks
- Test `/backtest` endpoint with date ranges
- Monitor response times (<2 seconds target)

---

## 🎯 Common First-Run Checklist

- [ ] Virtual environment created & activated
- [ ] `pip install -r requirements.txt` succeeded
- [ ] `python backend/config.py` prints settings
- [ ] `python backend/pipeline.py` completes without errors
- [ ] `unified_training_dataset.csv` exists and has data
- [ ] `python backend/ml_model/sentiment_extractor.py` completes
- [ ] `sentiment_augmented_dataset.csv` exists with sentiment columns
- [ ] `python backend/ml_model/train.py` trains model (accuracy > 50%)
- [ ] Model files exist in `backend/ml_model/trained_models/`
- [ ] `python backend/ml_model/evaluate.py` produces backtest report
- [ ] `python backend/app.py` starts API server
- [ ] API `/predict` endpoint responds with predictions

---

## ⏱️ Estimated Timelines

| Phase | Task | Time |
|-------|------|------|
| Setup | Install & verify | 10 min |
| Phase 2 | Data collection | 5-10 min |
| Phase 3 | Sentiment extraction | 5-10 min |
| Phase 4 | Model training | 5 min |
| Phase 5 | Backtesting | 2-5 min |
| Phase 6 | API testing | 5 min |
| **Total** | **End-to-end** | **30-45 min** |

---

## 📞 Support & Debugging

1. **Check logs:** `backend/logs/` for error messages
2. **Review config:** `backend/config.py` — most issues stem from settings
3. **Verify data:** Do datasets exist in `backend/data/processed/`?
4. **Check dependencies:** `pip list | grep -E "torch|transformers|pandas"`
5. **See docs:** [README.md](README.md) has full troubleshooting section

---

## 🎉 Success Criteria

Your setup is successful when:

✅ All phases complete without errors  
✅ Model accuracy > 50% on test set  
✅ Backtest shows reasonable metrics  
✅ API server starts and responds to requests  
✅ All required output files exist  

---

**Ready to start? Run this command:**

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Then proceed to **Step 4: Review Configuration** above.

**Good luck! 🚀**
